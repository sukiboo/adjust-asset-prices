from datetime import date
from typing import Literal, cast

import pandas as pd

from ..constants import OPTIONS_INTERNALS
from ..schemas import AssetType, OSIContract
from ..utils import (
    build_target_index,
    describe_adjusted_options,
    fetch_splits,
    format_osi_ticker,
    load_options_data,
    parse_osi_ticker,
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
        self, underlying: str, date_start: str | None = None, date_end: str | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieve `underlying`'s option contracts as `(calls, puts)`: load → split-unify →
        backfill. The options-side mirror of `AssetPrices.get_prices`; the underlying series the
        gate compares against is retrieved separately (split-only, via `get_prices`).
        """
        calls, puts = self.load_options(underlying, date_start, date_end)
        print("⚙️  Adjusting options contracts...")
        calls, puts = self.adjust_options_splits(calls, puts, underlying)
        calls, puts = self.backfill_options(calls, puts)
        describe_adjusted_options(calls, puts, underlying)
        return calls, puts

    def load_options(
        self, underlying: str, date_start: str | None = None, date_end: str | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw option bars for `underlying`, partitioned into (calls, puts), each
        multi-indexed on `(timestamp_utc, ticker)` with a `close` column. I/O + reshape only
        (backfill / adjust / gate are downstream); either side may be empty.
        """
        print(f"\n⛏️  Loading options for {underlying}...")
        df = load_options_data(self.data_dir, underlying, date_start, date_end)
        df["timestamp_utc"] = pd.to_datetime(df["window_start"], unit="ns", utc=True)
        df = df[["timestamp_utc", "ticker", "close"]]

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

        # Build the full RTH target index once and slice it per contract via searchsorted,
        # rather than a per-contract mcal.schedule call (the old hotspot: ~20min → seconds).
        ts_level = cast(pd.DatetimeIndex, df.index.get_level_values("timestamp_utc"))
        window_start = cast(pd.Timestamp, ts_level.min()).tz_convert("America/New_York")
        window_end = cast(pd.Timestamp, ts_level.max()).tz_convert("America/New_York")
        window_end_date = window_end.date()
        full_index = build_target_index(window_start, window_end, AssetType.OPTIONS)
        full_et = cast(pd.DatetimeIndex, full_index.tz_convert("America/New_York"))

        backfilled = []
        for ticker, group in df.groupby(level="ticker", sort=False):
            series = group.droplevel("ticker")["close"]
            expiry = parse_osi_ticker(cast(str, ticker)).expiry
            first_date = cast(pd.Timestamp, series.index[0]).tz_convert("America/New_York").date()
            end_date = min(expiry, window_end_date)
            if end_date < first_date:
                continue  # contract first prints after expiry — upstream bug, drop
            start_ts = pd.Timestamp(first_date, tz="America/New_York")
            end_ts = pd.Timestamp(end_date, tz="America/New_York") + pd.Timedelta(days=1)
            start_idx = full_et.searchsorted(start_ts, side="left")
            end_idx = full_et.searchsorted(end_ts, side="left")
            target_index = full_index[start_idx:end_idx]
            reindexed = series.reindex(target_index)
            is_real = reindexed.notna().to_numpy()
            filled = reindexed.ffill().bfill().to_numpy()
            mi = pd.MultiIndex.from_product(
                [target_index, [ticker]], names=["timestamp_utc", "ticker"]
            )
            backfilled.append(pd.DataFrame({"close": filled, "is_real": is_real}, index=mi))

        result = pd.concat(backfilled).sort_index()
        n_contracts = result.index.get_level_values("ticker").nunique()
        print(f"🔧 Backfilled to {len(result):,} rows across {n_contracts:,} {side} contracts")
        return result

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

        rewrites = self._successor_symbols(df, split_ts, ratio, underlying)
        pre, post = df[pre_mask].copy(), df[~pre_mask]
        pre["close"] /= ratio
        pre.index = pd.MultiIndex.from_arrays(
            [
                pre.index.get_level_values("timestamp_utc"),
                pre.index.get_level_values("ticker").map(rewrites),
            ],
            names=["timestamp_utc", "ticker"],
        )
        # The raw feed double-emits a contract under both its base and a suffixed OCC root on
        # the transition day; after rewriting, the twins collide on one (ts, ticker) — keep one.
        merged = pd.concat([post, pre]).sort_index()
        deduped = merged[~merged.index.duplicated(keep="first")]

        n_twins = len(merged) - len(deduped)
        print(
            f"🪚  {ratio:g}-for-1 split on {split_ts.date()}: rewrote {len(rewrites):,}"
            f" {side} contracts / {int(pre_mask.sum()):,} rows"
            + (f", deduped {n_twins:,} twin rows" if n_twins else "")
        )
        return deduped

    def _successor_symbols(
        self, df: pd.DataFrame, split_ts: pd.Timestamp, ratio: float, underlying: str
    ) -> dict[str, str]:
        """Map every pre-split ticker — any root, incl. already-suffixed AAPL7 — to its
        post-split successor symbol.
        """
        ts_level = df.index.get_level_values("timestamp_utc")
        ticker_level = df.index.get_level_values("ticker")
        candidates = self._suffixed_candidates(ticker_level[ts_level >= split_ts], underlying)
        split_date = split_ts.date()
        return {
            cast(str, t): self._successor_for(cast(str, t), ratio, split_date, candidates)
            for t in ticker_level[ts_level < split_ts].unique()
        }

    def _successor_for(
        self,
        ticker: str,
        ratio: float,
        split_date: date,
        candidates: dict[tuple[date, str], list[tuple[str, float]]],
    ) -> str:
        """The post-split symbol `ticker` rewrites to (strike ÷ ratio, own root). Clean strikes
        and expired-before-split contracts resolve directly; a non-clean spanning contract
        matches OCC's suffixed-root successor in the data (closest strike within 1¢), else
        falls back to the direct standalone symbol.
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
            return direct

        pool = candidates.get((parsed.expiry, parsed.option_type), [])
        best = min(pool, key=lambda c: abs(c[1] - new_strike), default=None)
        if (
            best is not None
            and abs(best[1] - new_strike) <= OPTIONS_INTERNALS["successor_strike_tol"]
        ):
            print(
                f"🔗  Suffix-match: {ticker} → {best[0]} "
                f"(${parsed.strike} ÷ {ratio:g} ≈ ${new_strike:.4f} ≈ ${best[1]})"
            )
            return best[0]
        tgt = f"closest ${best[1]:.4f}" if best is not None else "no candidate"
        print(
            f"⛓️‍💥  No successor for {ticker} ({ratio:g}-for-1 on {split_date}; {tgt}); "
            f"standalone scale → ${new_strike:.4f}"
        )
        return direct

    def _suffixed_candidates(
        self, post_tickers: pd.Index, underlying: str
    ) -> dict[tuple[date, str], list[tuple[str, float]]]:
        """Index post-split suffixed-root contracts (AAPL7, NVDA1, …) by (expiry, type) for the
        non-clean strike match. Base-root tickers merge via the clean strike-division path.
        """
        candidates: dict[tuple[date, str], list[tuple[str, float]]] = {}
        for t in post_tickers.unique():
            parsed = parse_osi_ticker(cast(str, t))
            suffix = parsed.underlying[len(underlying) :]
            if parsed.underlying == underlying or not (
                parsed.underlying.startswith(underlying) and suffix.isdigit()
            ):
                continue
            key = (parsed.expiry, parsed.option_type)
            candidates.setdefault(key, []).append((cast(str, t), parsed.strike))
        return candidates

    def _is_handled_split_ratio(self, ratio: float) -> bool:
        """True iff `ratio` is x or 1/x for integer x >= 2 (real x:1 / 1:x split); skips spinoffs."""
        return ratio > 0 and any(
            abs(x - round(x)) < OPTIONS_INTERNALS["integer_tol"]
            and round(x) >= OPTIONS_INTERNALS["min_split_factor"]
            for x in (ratio, 1 / ratio)
        )
