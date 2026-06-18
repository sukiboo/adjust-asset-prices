from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import date
from typing import Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from ..constants import OPTIONS_INTERNALS
from ..schemas import AssetType, OSIContract, Predecessor
from ..utils import (
    build_target_index,
    describe_adjusted_options,
    drop_implausible_timestamps,
    fetch_splits,
    format_osi_ticker,
    load_options_data,
    parse_osi_ticker,
    underlying_matches,
)
from .assets import AssetPrices

OptionSide = Literal["call", "put"]


class OptionsPrices:
    """Options companion pass for an underlying: load → split-unify → backfill its OSI
    contracts. Built from the `AssetPrices` sibling purely to share its validated `data_dir`
    (so `check_data_dir` runs once). The underlying itself is retrieved separately via the
    standard `AssetPrices.get_prices` path; the structural gate compares contracts against it.
    """

    def __init__(self, asset: AssetPrices) -> None:
        self.data_dir = asset.data_dir

    def get_options(
        self,
        underlying: str,
        date_start: str | None = None,
        date_end: str | None = None,
        predecessors: Sequence[Predecessor] = (),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieve `underlying`'s option contracts as `(calls, puts)`: load → rename-unify →
        split-unify → backfill. The options-side mirror of `AssetPrices.get_prices`; the underlying
        series the gate compares against is retrieved separately (split-only, via `get_prices`).
        `predecessors` are former ticker symbols (from the stock-side rename auto-stitch) whose
        OSI contracts are loaded and rewritten to the live root, so series spanning a rename load
        continuous.
        """
        calls, puts = self.load_and_adjust(underlying, date_start, date_end, predecessors)
        calls, puts = self.backfill_options(calls, puts)
        describe_adjusted_options(calls, puts, underlying)
        return calls, puts

    def load_and_adjust(
        self,
        underlying: str,
        date_start: str | None = None,
        date_end: str | None = None,
        predecessors: Sequence[Predecessor] = (),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load + rename-unify + split-unify (no backfill) → raw adjusted `(calls, puts)`, every bar
        a real print. `Prices.process` gates these and streams the backfill to disk (never holding
        the full RAM-exceeding frame); `get_options` is the eager variant that also backfills.
        """
        calls, puts = self.load_options(underlying, date_start, date_end, predecessors)
        print("⚙️  Adjusting options contracts...")
        calls, puts = self.unify_rename_symbols(calls, puts, underlying, predecessors)
        calls, puts = self.adjust_options_splits(calls, puts, underlying)
        return calls, puts

    def load_options(
        self,
        underlying: str,
        date_start: str | None = None,
        date_end: str | None = None,
        predecessors: Sequence[Predecessor] = (),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw option bars for `underlying`, partitioned into (calls, puts), each
        multi-indexed on `(timestamp_utc, ticker)` with a `close` column. I/O + reshape only
        (backfill / adjust / gate are downstream); either side may be empty. Each `predecessor`'s
        `O:<symbol>…` contracts are loaded too, bounded to its span so a reused ticker can't leak.
        """
        print(f"\n⛏️  Loading options for {underlying}...")
        frames = [load_options_data(self.data_dir, underlying, date_start, date_end)]
        for pred in predecessors:
            try:
                frames.append(
                    load_options_data(
                        self.data_dir, pred.symbol, pred.start.isoformat(), pred.end.isoformat()
                    )
                )
                print(f"⛏️  Loaded predecessor {pred.symbol} options ({pred.start} -- {pred.end})")
            except ValueError:
                print(f"🪏  No predecessor {pred.symbol} options in {pred.start} -- {pred.end}")
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        df["timestamp_utc"] = pd.to_datetime(df["window_start"], unit="ns", utc=True)
        df = df[["timestamp_utc", "ticker", "close"]]
        # Use 64-bit string offsets: at large-underlying scale (e.g. full-range QQQ) the combined
        # ticker column exceeds pyarrow `string`'s int32 offset cap (~2 GB) and overflows when the
        # boolean call/put split below materializes a subset. `large_string` lifts the ceiling;
        # no-op cost for the small loads that already worked.
        df["ticker"] = df["ticker"].astype(pd.ArrowDtype(pa.large_string()))

        # After the large_string cast so the boolean mask's `take` doesn't overflow at QQQ scale.
        df = drop_implausible_timestamps(df, "option")

        contract_type = {t: parse_osi_ticker(t).option_type for t in df["ticker"].unique()}
        is_call = df["ticker"].map(contract_type) == "C"
        calls = (
            df[is_call]
            .sort_values(["timestamp_utc", "ticker"])
            .set_index(["timestamp_utc", "ticker"])
        )
        puts = (
            df[~is_call]
            .sort_values(["timestamp_utc", "ticker"])
            .set_index(["timestamp_utc", "ticker"])
        )

        n_calls = calls.index.get_level_values("ticker").nunique() if not calls.empty else 0
        n_puts = puts.index.get_level_values("ticker").nunique() if not puts.empty else 0
        print(f"🗑️  Loaded {len(calls):,} call records across {n_calls:,} contracts")
        print(f"🗑️  Loaded {len(puts):,} put records across {n_puts:,} contracts")

        return calls, puts

    def backfill_options(
        self, calls: pd.DataFrame, puts: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Backfill each contract onto its RTH window `[first_bar, min(expiry, window_end)]`.
        Options are illiquid, so gaps (and the run-out to expiry) are filled with the last
        traded price held flat (ffill/bfill) — not interpolated — to avoid inventing a price
        path that never traded. The end runs to expiry (not last trade), so a contract is held
        flat to expiry rather than vanishing mid-life (matters across splits, where an
        OCC-adjusted contract may not trade post-split). Adds an `is_real` column flagging the
        genuine prints (vs synthetic fill) so the structural gate can score real bars only.
        """
        return self._backfill_contracts(calls, "call"), self._backfill_contracts(puts, "put")

    def _backfill_contracts(self, df: pd.DataFrame, side: OptionSide) -> pd.DataFrame:
        if df.empty:
            return df
        # sort_index restores the canonical (timestamp, ticker) order; the streaming caller keeps
        # the (ticker, timestamp) emission order (a global timestamp sort needs the full frame).
        result = pd.concat(self._backfill_batches(df)).sort_index()
        n_contracts = result.index.get_level_values("ticker").nunique()
        print(f"🔧 Backfilled to {len(result):,} rows across {n_contracts:,} {side} contracts")
        return result

    def backfill_option_batches(
        self, df: pd.DataFrame, batch_rows: int = int(OPTIONS_INTERNALS["save_batch_rows"])
    ) -> Iterator[pd.DataFrame]:
        """Yield the backfilled `(close, is_real)` frame in row-bounded batches (~`batch_rows` each)
        so a large universe streams to disk without holding the full frame; row-count batching (not
        per-contract) bounds long-dated LEAPS."""
        yield from self._backfill_batches(df, batch_rows)

    def _backfill_batches(
        self, df: pd.DataFrame, batch_rows: int | None = None
    ) -> Iterator[pd.DataFrame]:
        # Build the RTH index once, slice per contract via searchsorted. Contracts come sorted-ticker
        # and each timestamp-sorted, so a batch is (ticker, timestamp)-ordered without a sort.
        # batch_rows set → flush at that size (streaming); None → one batch (eager caller re-sorts).
        ts_level = cast(pd.DatetimeIndex, df.index.get_level_values("timestamp_utc"))
        window_start = cast(pd.Timestamp, ts_level.min()).tz_convert("America/New_York")
        window_end = cast(pd.Timestamp, ts_level.max()).tz_convert("America/New_York")
        window_end_date = window_end.date()
        full_index = build_target_index(window_start, window_end, AssetType.OPTIONS)
        full_et = cast(pd.DatetimeIndex, full_index.tz_convert("America/New_York"))

        batch: list[pd.DataFrame] = []
        batch_len = 0
        for ticker, group in df.groupby(level="ticker", sort=True):
            frame = self._backfill_one(
                cast(str, ticker), group, full_index, full_et, window_end_date
            )
            if frame is None:
                continue
            batch.append(frame)
            batch_len += len(frame)
            if batch_rows is not None and batch_len >= batch_rows:
                yield pd.concat(batch)
                batch, batch_len = [], 0
        if batch:
            yield pd.concat(batch)

    def _backfill_one(
        self,
        ticker: str,
        group: pd.DataFrame,
        full_index: pd.DatetimeIndex,
        full_et: pd.DatetimeIndex,
        window_end_date: date,
    ) -> pd.DataFrame | None:
        """Reindex one contract onto `[first_bar, min(expiry, window_end)]` and flat-fill
        (ffill/bfill, no interpolation — options are illiquid). None if it first prints after expiry.
        """
        series = group.droplevel("ticker")["close"]
        expiry = parse_osi_ticker(ticker).expiry
        first_date = cast(pd.Timestamp, series.index[0]).tz_convert("America/New_York").date()
        end_date = min(expiry, window_end_date)
        if end_date < first_date:
            return None
        start_ts = pd.Timestamp(first_date, tz="America/New_York")
        end_ts = pd.Timestamp(end_date, tz="America/New_York") + pd.Timedelta(days=1)
        start_idx = full_et.searchsorted(start_ts, side="left")
        end_idx = full_et.searchsorted(end_ts, side="left")
        target_index = full_index[start_idx:end_idx]
        reindexed = series.reindex(target_index)
        is_real = reindexed.notna().to_numpy()
        filled = reindexed.ffill().bfill().to_numpy()
        mi = pd.MultiIndex.from_product([target_index, [ticker]], names=["timestamp_utc", "ticker"])
        return pd.DataFrame({"close": filled, "is_real": is_real}, index=mi)

    def adjust_options_splits(
        self, calls: pd.DataFrame, puts: pd.DataFrame, underlying: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split-unify each side: per split (chronological) on `underlying`, rescale pre-split
        premiums by 1/ratio and rewrite their symbols to the post-split successor, so each
        contract is one continuous series across the split. Only integer (x:1) or 1/integer
        (1:x) ratios are handled — other ratios (spinoffs / distributions) are skipped.
        See `_successor_for` for the successor rules and CLAUDE.md for the full rationale.
        """
        if calls.empty and puts.empty:
            return calls, puts

        starts = [
            cast(pd.Timestamp, df.index.get_level_values("timestamp_utc").min())
            for df in (calls, puts)
            if not df.empty
        ]
        splits = fetch_splits(underlying, min(starts))
        if splits.empty:
            print(f"🪚  No splits to apply for {underlying}")
            return calls, puts
        splits = splits.sort_index()
        return (
            self._unify_split_symbols(calls, splits, underlying, "call"),
            self._unify_split_symbols(puts, splits, underlying, "put"),
        )

    def _unify_split_symbols(
        self, df: pd.DataFrame, splits: pd.Series, underlying: str, side: OptionSide
    ) -> pd.DataFrame:
        """Rewrite every pre-split contract into post-split currency, one split at a time."""
        if df.empty:
            return df
        for split_ts, ratio in splits.items():
            split_ts = cast(pd.Timestamp, split_ts)
            if not self._is_handled_split_ratio(ratio):
                print(
                    f"⛓️‍💥  Skipping {ratio:.4f}-ratio event on {split_ts.date()}: "
                    "likely a spinoff / distribution, not a stock split (no OCC support)"
                )
                continue
            df = self._apply_split(df, split_ts, ratio, underlying, side)
        return df

    def _apply_split(
        self,
        df: pd.DataFrame,
        split_ts: pd.Timestamp,
        ratio: float,
        underlying: str,
        side: OptionSide,
    ) -> pd.DataFrame:
        """Rescale pre-split bars by 1/ratio, relabel to successor symbols, merge with the
        post-split bars (deduping the base/suffixed twins the raw feed emits).
        """
        ts_level = df.index.get_level_values("timestamp_utc")
        pre_mask = ts_level < split_ts
        if not pre_mask.any():
            return df

        rewrites, counts = self._successor_symbols(df, split_ts, ratio, underlying)
        deduped, n_twins = self._relabel_and_merge(df, pre_mask, rewrites, scale=ratio)
        print(
            f"🪚  {ratio:g}-for-1 split on {split_ts.date()}: rewrote {len(rewrites):,}"
            f" {side} contracts / {int(pre_mask.sum()):,} rows"
            + (f", {counts['matched']:,} matched" if counts["matched"] else "")
            + (
                f", {counts['standalone']:,} standalone (no OCC successor)"
                if counts["standalone"]
                else ""
            )
            + (f", deduped {n_twins:,} twin rows" if n_twins else "")
        )
        return deduped

    def _relabel_and_merge(
        self, df: pd.DataFrame, mask: np.ndarray, rewrites: dict[str, str], scale: float = 1.0
    ) -> tuple[pd.DataFrame, int]:
        """Relabel the `mask`-selected rows' ticker level via `rewrites` (rescaling their `close`
        by 1/`scale`), merge back with the rest, and dedup any (ts, ticker) twins the raw feed
        emits on a transition day — the unmasked/native rows win. Shared by the split rewrite
        (scale=ratio, mask=pre-split bars) and the rename rewrite (scale=1, mask=predecessor-root
        bars). Returns the deduped frame and the twin-row count dropped.
        """
        sel, rest = df[mask].copy(), df[~mask]
        if scale != 1.0:
            sel["close"] /= scale
        sel.index = pd.MultiIndex.from_arrays(
            [
                sel.index.get_level_values("timestamp_utc"),
                sel.index.get_level_values("ticker").map(rewrites),
            ],
            names=["timestamp_utc", "ticker"],
        )
        merged = pd.concat([rest, sel]).sort_index()
        deduped = merged[~merged.index.duplicated(keep="first")]
        return deduped, len(merged) - len(deduped)

    def unify_rename_symbols(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        underlying: str,
        predecessors: Sequence[Predecessor],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Rewrite predecessor-root contracts (e.g. `O:FB…`) to the live root (`O:META…`) so a
        contract spanning a ticker rename loads as one continuous series. A rename is a degenerate
        split — same strike/expiry, premium unchanged, only the root differs — so it reuses the
        split relabel+merge primitive with no premium scaling. Runs before `adjust_options_splits`
        so the split-unifier sees a single root namespace. No-op when there are no predecessors.
        """
        if not predecessors:
            return calls, puts
        return (
            self._unify_rename_side(calls, underlying, predecessors, "call"),
            self._unify_rename_side(puts, underlying, predecessors, "put"),
        )

    def _unify_rename_side(
        self,
        df: pd.DataFrame,
        underlying: str,
        predecessors: Sequence[Predecessor],
        side: OptionSide,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        rewrites: dict[str, str] = {}
        for t in df.index.get_level_values("ticker").unique():
            new = self._rename_ticker(cast(str, t), underlying, predecessors)
            if new is not None:
                rewrites[cast(str, t)] = new
        if not rewrites:
            return df
        mask = df.index.get_level_values("ticker").isin(list(rewrites))
        deduped, n_twins = self._relabel_and_merge(df, mask, rewrites)
        print(
            f"🔗 Rename-unified {len(rewrites):,} predecessor {side} contracts to {underlying}"
            + (f", deduped {n_twins:,} twin rows" if n_twins else "")
        )
        return deduped

    def _rename_ticker(
        self, ticker: str, underlying: str, predecessors: Sequence[Predecessor]
    ) -> str | None:
        """The live-root symbol a predecessor-root `ticker` rewrites to (root swapped, OCC numeric
        suffix preserved, strike/expiry/type intact); None if `ticker` is not on a predecessor root.
        """
        parsed = parse_osi_ticker(ticker)
        for pred in predecessors:
            if underlying_matches(parsed.underlying, pred.symbol):
                new_root = underlying + parsed.underlying[len(pred.symbol) :]  # keep OCC suffix
                return format_osi_ticker(
                    OSIContract(new_root, parsed.expiry, parsed.option_type, parsed.strike)
                )
        return None

    def _successor_symbols(
        self, df: pd.DataFrame, split_ts: pd.Timestamp, ratio: float, underlying: str
    ) -> tuple[dict[str, str], Counter[str]]:
        """Map every pre-split ticker — any root, incl. already-suffixed AAPL7 — to its
        post-split successor symbol, plus a Counter of successor categories (matched/standalone)
        for the per-split summary line.
        """
        ts_level = df.index.get_level_values("timestamp_utc")
        ticker_level = df.index.get_level_values("ticker")
        candidates = self._successor_candidates(ticker_level[ts_level >= split_ts], underlying)
        split_date = split_ts.date()
        rewrites: dict[str, str] = {}
        counts: Counter[str] = Counter()
        for t in ticker_level[ts_level < split_ts].unique():
            successor, kind = self._successor_for(cast(str, t), ratio, split_date, candidates)
            rewrites[cast(str, t)] = successor
            counts[kind] += 1
        return rewrites, counts

    def _successor_for(
        self,
        ticker: str,
        ratio: float,
        split_date: date,
        candidates: dict[tuple[date, str], list[tuple[str, float, bool]]],
    ) -> tuple[str, str]:
        """Successor symbol for `ticker` (strike ÷ ratio) + category ("direct"/"matched"/
        "standalone"). Clean/expired strikes resolve directly; a non-clean spanning contract matches
        the closest OCC successor within 1¢, suffixed root before base (feed dual-emits both).
        """
        parsed = parse_osi_ticker(ticker)
        new_strike = parsed.strike / ratio
        direct = format_osi_ticker(
            OSIContract(parsed.underlying, parsed.expiry, parsed.option_type, new_strike)
        )
        scale = OPTIONS_INTERNALS["strike_scale"]
        clean = (
            abs(new_strike * scale - round(new_strike * scale)) < OPTIONS_INTERNALS["integer_tol"]
        )
        if clean or parsed.expiry < split_date:
            return direct, "direct"

        pool = candidates.get((parsed.expiry, parsed.option_type), [])
        tol = OPTIONS_INTERNALS["successor_strike_tol"]
        for want_suffixed in (True, False):  # suffixed root is canonical; base root is the fallback
            sub = [c for c in pool if c[2] is want_suffixed]
            best = min(sub, key=lambda c: abs(c[1] - new_strike), default=None)
            if best is not None and abs(best[1] - new_strike) <= tol:
                return best[0], "matched"
        return direct, "standalone"

    def _successor_candidates(
        self, post_tickers: pd.Index, underlying: str
    ) -> dict[tuple[date, str], list[tuple[str, float, bool]]]:
        """Post-split contracts indexed by (expiry, type) as `(ticker, strike, is_suffixed)`, across
        both OCC conventions: suffixed roots (AAPL7) and base-root cent-rounded strikes (TQQQ 3:1).
        """
        candidates: dict[tuple[date, str], list[tuple[str, float, bool]]] = {}
        for t in post_tickers.unique():
            parsed = parse_osi_ticker(cast(str, t))
            suffix = parsed.underlying[len(underlying) :]
            base = parsed.underlying == underlying
            suffixed = parsed.underlying.startswith(underlying) and suffix.isdigit()
            if not (base or suffixed):
                continue
            key = (parsed.expiry, parsed.option_type)
            candidates.setdefault(key, []).append((cast(str, t), parsed.strike, suffixed))
        return candidates

    def _is_handled_split_ratio(self, ratio: float) -> bool:
        """True iff `ratio` is x or 1/x for integer x >= 2 (real x:1 / 1:x split); skips spinoffs."""
        return ratio > 0 and any(
            abs(x - round(x)) < OPTIONS_INTERNALS["integer_tol"]
            and round(x) >= OPTIONS_INTERNALS["min_split_factor"]
            for x in (ratio, 1 / ratio)
        )
