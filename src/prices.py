from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from .schemas import AssetType, OSIContract
from .utils import (
    build_target_index,
    check_data_dir,
    fetch_dividends,
    fetch_splits,
    fetch_yf_closes,
    format_osi_ticker,
    load_options_data,
    load_ticker_data,
    parse_date,
    parse_osi_ticker,
    resolve_index_bound,
)


def _is_handled_split_ratio(ratio: float, tol: float = 1e-6) -> bool:
    """True iff `ratio` is an integer ≥ 2 (forward split) or 1/integer with integer ≥ 2
    (reverse split). Anything else is skipped — empirically spinoffs / distributions /
    non-stock-split corporate actions in modern data, not true stock splits.
    """
    if abs(ratio - round(ratio)) < tol and round(ratio) >= 2:
        return True
    if ratio > 0 and abs(1 / ratio - round(1 / ratio)) < tol and round(1 / ratio) >= 2:
        return True
    return False


class Prices:
    """Class to load prices for a given ticker and date range.
    Raw prices are loaded and backfilled to the start and end dates.
    For stocks and options adjusted prices are calculated using the
    backfilled prices and the split and dividend data.
    """

    def __init__(self, data_dir: str, debug: bool = False) -> None:
        self.debug = debug
        self.data_dir, self.asset_types = check_data_dir(data_dir)

    def get_prices(
        self, ticker: str, date_start: str | None = None, date_end: str | None = None
    ) -> tuple[pd.DataFrame, AssetType]:
        """Get prices for a given ticker and date range, paired with the detected asset type."""
        df, asset_type = self.load_prices(ticker, date_start, date_end)
        df = self.adjust_prices(df, asset_type, date_start, date_end)
        return df, asset_type

    def load_prices(
        self, ticker: str, date_start: str | None = None, date_end: str | None = None
    ) -> tuple[pd.DataFrame, AssetType]:
        """Load prices for a given ticker and date range."""
        print(f"⛏️  Loading {ticker} price data...")
        df, asset_type = load_ticker_data(
            self.data_dir, self.asset_types, ticker, date_start, date_end
        )
        if "window_start" not in df.columns:
            raise ValueError(f"No window_start column found in data for ticker: `{ticker}`")

        df["timestamp_utc"] = pd.to_datetime(df["window_start"], unit="ns", utc=True)
        df = df.sort_values("timestamp_utc").set_index("timestamp_utc")
        df = pd.DataFrame(df[["close"]]).rename(columns={"close": ticker})

        if self.debug:
            start_date = parse_date(cast(pd.Timestamp, df.index[0]))
            end_date = parse_date(cast(pd.Timestamp, df.index[-1]))
            print(
                f"🗑️  Loaded {len(df):,} price records for {ticker} "
                f"from {start_date} to {end_date}:"
                f"\n{'-' * 32}\n{df.head(5)}\n{'-' * 32}"
            )

        return df, asset_type

    def load_options(
        self, underlying: str, date_start: str | None = None, date_end: str | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw option bars for all contracts on `underlying` in the date range,
        partitioned into (calls, puts). Each frame is multi-indexed on
        `(timestamp_utc, ticker)` with a single `close` column. Either side may be
        empty if the underlying only had calls (or only puts) in range.
        Backfill / adjustment / gating are deliberately downstream — this method is
        I/O + reshape only.
        """
        print(f"⛏️  Loading options for {underlying}...")
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

        if self.debug:
            n_calls = calls.index.get_level_values("ticker").nunique() if not calls.empty else 0
            n_puts = puts.index.get_level_values("ticker").nunique() if not puts.empty else 0
            print(f"🗑️  Loaded {len(calls):,} call bars across {n_calls:,} contracts")
            print(f"🗑️  Loaded {len(puts):,} put bars across {n_puts:,} contracts")

        return calls, puts

    def backfill_options(
        self, calls: pd.DataFrame, puts: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Backfill each contract independently onto its own RTH window:
        `[first_bar.date(), min(expiry, data_window_end)]`. The end runs to expiry, not
        to last trade, so a contract that stops printing before expiry is held flat
        (ffill) to expiry instead of vanishing mid-life — the holder still owns it, and
        a gap would otherwise read as a delisting. This matters across splits: an
        OCC-adjusted contract with no post-split trades (e.g. a pre-split strike
        re-struck to a thin post-split strike) would otherwise stop dead at the split
        rather than spanning it. Capped at the loaded data window's last date when
        expiry falls beyond it, since we have no market context past the data we loaded.
        The start stays at first_bar — we don't fabricate pre-listing history. Log-linear
        interpolation on interior gaps, ffill/bfill on the edges. Post-expiry raw bars
        (upstream bugs) are still dropped via the `min(..., expiry)` cap.
        """
        print("⚙️  Backfilling options contracts...")
        return self._backfill_contracts(calls), self._backfill_contracts(puts)

    def _backfill_contracts(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Pre-build one full RTH 1-min target index over the loaded data window so the
        # per-contract path becomes an O(log n) searchsorted slice instead of a fresh
        # mcal.schedule call. The per-contract mcal.schedule (~10-30ms each) was the
        # dominant cost; for busy underlyings (NVDA-style, 10k+ contracts) this turns
        # 20-min runs into ~10s.
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
            log_interp = cast(pd.Series, np.log(reindexed)).interpolate(method="linear")
            filled = cast(pd.Series, np.exp(log_interp)).ffill().bfill()
            filled.index = pd.MultiIndex.from_product(
                [target_index, [ticker]], names=["timestamp_utc", "ticker"]
            )
            backfilled.append(filled.to_frame("close"))

        result = pd.concat(backfilled).sort_index()
        if self.debug:
            n_contracts = result.index.get_level_values("ticker").nunique()
            print(f"🔧 Backfilled to {len(result):,} rows across {n_contracts:,} contracts")
        return result

    def adjust_options_splits(
        self, calls: pd.DataFrame, puts: pd.DataFrame, underlying: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scale every pre-split premium into post-split currency and rewrite its symbol
        to the split-adjusted strike — the options analog of the stocks split adjustment.
        For each split (chronological) on `underlying`, for every ticker with bars before
        split_date: divide pre-split bars by ratio and relabel under the strike/ratio
        successor symbol. Contracts that span the split merge by index with the natively-
        post-split-issued contract at the same successor symbol, producing one continuous
        series; contracts that expired before the split have no successor and stand alone
        (a synthetic split-adjusted strike that never traded, like back-adjusted stock
        prices). All pre-split contracts are adjusted regardless of expiry so the output
        is in uniform post-split currency — leaving expired-before-split contracts in
        pre-split currency breaks the structural gate's underlying comparison.

        Only handles ratios that are integers (forward splits like 2:1, 7:1, 10:1) or
        1/integer (reverse splits like 1:8, 1:20). Other ratios in yfinance.splits are
        empirically spinoffs / distributions / non-stock-split corporate actions
        (verified zero true 3:2-style splits in the top 400 option underlyings across
        2014-2025) — those get a one-line warning and skip. Spinoffs need OCC's
        deliverable-change adjustment which strike-divide-by-ratio doesn't model.
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
            if self.debug:
                print(f"🪚  No splits to apply for {underlying}")
            return calls, puts
        splits = splits.sort_index()
        return (
            self._unify_split_symbols(calls, splits, underlying),
            self._unify_split_symbols(puts, splits, underlying),
        )

    def _unify_split_symbols(
        self, df: pd.DataFrame, splits: pd.Series, underlying: str
    ) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        for split_ts, ratio in splits.items():
            split_ts_cast = cast(pd.Timestamp, split_ts)
            split_date = split_ts_cast.date()
            if not _is_handled_split_ratio(ratio):
                print(
                    f"⏭️  Skipping {ratio:.4f}-ratio event on {split_date}: "
                    f"non-integer ratio likely indicates spinoff / distribution, "
                    f"not a stock split — needs OCC deliverable-change handling we don't support"
                )
                continue
            ts_level = df.index.get_level_values("timestamp_utc")
            ticker_level = df.index.get_level_values("ticker")
            pre_mask = ts_level < split_ts_cast
            if not pre_mask.any():
                continue

            # Index suffixed-root post-split candidates by (expiry, type) for the non-clean
            # match path. Only contracts with bars on/after this split's date are eligible;
            # standard-root (no suffix) tickers aren't candidates here — those merge via the
            # clean strike-division path.
            post_mask = ts_level >= split_ts_cast
            suffixed_candidates: dict[tuple[date, str], list[tuple[str, float]]] = {}
            for t in ticker_level[post_mask].unique():
                parsed = parse_osi_ticker(cast(str, t))
                if parsed.underlying == underlying:
                    continue
                suffix = parsed.underlying[len(underlying) :]
                if not (parsed.underlying.startswith(underlying) and suffix.isdigit()):
                    continue
                key = (parsed.expiry, parsed.option_type)
                suffixed_candidates.setdefault(key, []).append((cast(str, t), parsed.strike))

            rewrites: dict[str, str] = {}
            for t in ticker_level[pre_mask].unique():
                parsed = parse_osi_ticker(cast(str, t))
                if parsed.underlying != underlying:
                    continue  # already a suffixed-root; don't re-rewrite at this split
                new_strike = parsed.strike / ratio
                clean = abs(new_strike * 1000 - round(new_strike * 1000)) < 1e-6
                # Clean strikes resolve to their successor symbol directly. So do
                # contracts that expired before the split: OCC never issued a successor
                # for them, so there's nothing to match/merge — they become standalone
                # split-adjusted series with a synthetic strike that never traded, exactly
                # as back-adjusted stock prices carry synthetic pre-split values. (A
                # non-clean expired strike rounds to OSI's native milli resolution.)
                if clean or parsed.expiry < split_date:
                    rewrites[cast(str, t)] = format_osi_ticker(
                        OSIContract(
                            parsed.underlying, parsed.expiry, parsed.option_type, new_strike
                        )
                    )
                    continue
                # Non-clean AND spanning the split: match OCC's suffixed-root successor
                # in the data (the real adjusted contract we want to merge with).
                candidates = suffixed_candidates.get((parsed.expiry, parsed.option_type), [])
                if not candidates:
                    if self.debug:
                        print(
                            f"⚠️  No suffixed-root candidate for {t} ({ratio:g}-for-1 "
                            f"on {split_date}): target ${new_strike:.4f}, "
                            f"no post-split contracts at same expiry/type"
                        )
                    continue
                best_ticker, best_strike = min(candidates, key=lambda c: abs(c[1] - new_strike))
                if abs(best_strike - new_strike) > 0.01:  # > 1¢: not the same OCC adjustment
                    if self.debug:
                        print(
                            f"⚠️  No close strike for {t} ({ratio:g}-for-1 on {split_date}): "
                            f"target ${new_strike:.4f}, closest ${best_strike:.4f}"
                        )
                    continue
                rewrites[cast(str, t)] = best_ticker
                if self.debug:
                    print(
                        f"🔗  Suffix-match: {t} → {best_ticker} "
                        f"(${parsed.strike} ÷ {ratio:g} ≈ ${new_strike:.4f} ≈ ${best_strike})"
                    )

            if not rewrites:
                continue

            rewrite_mask = pre_mask & ticker_level.isin(rewrites.keys())
            untouched = df[~rewrite_mask]
            to_rewrite = df[rewrite_mask].copy()
            to_rewrite["close"] /= ratio
            to_rewrite.index = pd.MultiIndex.from_arrays(
                [
                    to_rewrite.index.get_level_values("timestamp_utc"),
                    to_rewrite.index.get_level_values("ticker").map(rewrites),
                ],
                names=["timestamp_utc", "ticker"],
            )
            df = pd.concat([untouched, to_rewrite]).sort_index()
            if self.debug:
                print(
                    f"🪚  {ratio:g}-for-1 split on {split_date}: "
                    f"rewrote {len(rewrites)} contracts, {int(rewrite_mask.sum()):,} rows"
                )
        return df

    def adjust_prices(
        self,
        df: pd.DataFrame,
        asset_type: AssetType,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> pd.DataFrame:
        """Adjust prices for a given asset type via the following steps:
        1. Backfill the missing prices
        2. Adjust for splits (yfinance reports dividends in current-share-equivalent
           currency, so we put prices in that currency first)
        3. Adjust for dividends
        """
        print(f"⚙️  Adjusting {df.columns[0]} price data...")
        df = self.backfill_prices(df, asset_type, date_start, date_end)
        df = self.adjust_splits(df, asset_type)
        df = self.adjust_dividends(df, asset_type)
        return df

    def backfill_prices(
        self,
        df: pd.DataFrame,
        asset_type: AssetType,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> pd.DataFrame:
        """Backfill missing 1-minute rows over the appropriate trading calendar.
        Stocks use NYSE extended hours (04:00-19:59 ET on session days); options use NYSE
        regular hours (09:30-15:59 ET on session days). Half-days are handled by the
        calendar. Crypto and forex use a continuous 1-min grid. The output index spans
        every session bar over [date_start, date_end] when those are supplied (synthetic
        ffill/bfill prices on the first/last day if raw data does not reach the session
        edges); otherwise it falls back to the calendar over [df.index[0], df.index[-1]]
        with NYSE-asset timestamps converted to ET so .date() reflects the trading day.
        Interpolation runs in log space to preserve multiplicative behavior.
        """
        col, num_rows = df.columns[0], len(df)
        start_ts = resolve_index_bound(date_start, cast(pd.Timestamp, df.index[0]), asset_type)
        end_ts = resolve_index_bound(date_end, cast(pd.Timestamp, df.index[-1]), asset_type)
        target_index = build_target_index(start_ts, end_ts, asset_type)
        df = df.reindex(target_index)
        assert df.index.equals(target_index), "backfill produced unexpected index"
        df[col] = df[col].apply(np.log).interpolate(method="linear").apply(np.exp).ffill().bfill()
        if self.debug and len(df) > num_rows:
            print(f"🔧 Backfilled {len(df) - num_rows:,} new rows")
        df.index.name = "timestamp_utc"
        return df

    def adjust_splits(self, df: pd.DataFrame, asset_type: AssetType) -> pd.DataFrame:
        """Back-adjust historical prices for stock splits using yfinance split data.
        For each split with ex-date D and ratio r, prices at timestamps < D are divided
        by r. Options need contract-level adjustment (strike + multiplier) and are skipped.
        """
        if asset_type != AssetType.STOCKS:
            if self.debug:
                print(f"🪚  Not adjusting {asset_type} assets for splits")
            return df

        ticker = df.columns[0]
        splits = fetch_splits(ticker, cast(pd.Timestamp, df.index[0]))
        if splits.empty:
            if self.debug:
                print(f"🪚  No splits to apply for {ticker}")
            return df

        df = df.copy()
        for split_date, ratio in splits.items():
            ts = cast(pd.Timestamp, split_date)
            df.loc[df.index < ts, ticker] /= ratio
            if self.debug:
                print(f"🪚  Applied {ratio:g}-for-1 split on {ts.date()} to {ticker}")
        return df

    def adjust_dividends(self, df: pd.DataFrame, asset_type: AssetType) -> pd.DataFrame:
        """Back-adjust historical prices for cash dividends using yfinance dividend data.
        For each dividend with ex-date D and amount d, prices at timestamps < D are scaled
        by (1 - d / c) where c is yfinance's official Close on the trading day before D.
        Must run after adjust_splits: yfinance dividends are reported in current-share-
        equivalent currency, so we apply them against already-split-adjusted prices.
        """
        if asset_type != AssetType.STOCKS:
            if self.debug:
                print(f"🔩 Not adjusting {asset_type} assets for dividends")
            return df

        ticker = df.columns[0]
        divs = fetch_dividends(ticker, cast(pd.Timestamp, df.index[0])).sort_index()
        if divs.empty:
            if self.debug:
                print(f"🔩 No dividends to apply for {ticker}")
            return df

        yf_closes = fetch_yf_closes(
            ticker,
            cast(pd.Timestamp, df.index[0]),
            cast(pd.Timestamp, divs.index[-1]) + pd.Timedelta(days=1),
        )
        df = df.copy()
        for ex_date, amount in divs.items():
            ts = cast(pd.Timestamp, ex_date)
            mask = df.index < ts
            if not mask.any():
                continue
            prev_closes = yf_closes[yf_closes.index < ts]
            if prev_closes.empty:
                if self.debug:
                    print(f"🔩 Skipping {ts.date()} dividend: no yfinance close before ex-date")
                continue
            prev_close = prev_closes.iloc[-1]
            factor = 1 - amount / prev_close
            df.loc[mask, ticker] *= factor
            if self.debug:
                print(
                    f"🔩 Applied ${amount:.4f} dividend on {ts.date()} "
                    f"(factor {factor:.6f}) to {ticker}"
                )
        return df
