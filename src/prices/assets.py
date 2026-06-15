from datetime import date, timedelta
from typing import cast

import numpy as np
import pandas as pd

from ..aliases import stitch_predecessors
from ..schemas import AssetType, Predecessor, PriceEvent
from ..utils import (
    build_target_index,
    check_data_dir,
    describe_adjusted_prices,
    fetch_dividends,
    fetch_splits,
    fetch_yf_closes,
    load_ticker_data,
    parse_date,
    resolve_index_bound,
)


def _rename_events(
    ticker: str, predecessors: list[Predecessor], first: date, last: date
) -> list[PriceEvent]:
    """Rename markers at each symbol changeover in the stitched timeline (predecessor eras
    bracketed by the ticker): interior QQQ→QQQQ→QQQ yields two, leading FB→META yields one."""
    segments: list[tuple[date, str]] = []
    cursor = first
    for p in sorted(predecessors, key=lambda x: x.start):
        if cursor < p.start:
            segments.append((cursor, ticker))
        segments.append((p.start, p.symbol))
        cursor = p.end + timedelta(days=1)
    if cursor <= last:
        segments.append((cursor, ticker))
    return [
        PriceEvent(start, f"{a_sym}→{b_sym}", "rename")
        for (_, a_sym), (start, b_sym) in zip(segments, segments[1:])
        if a_sym != b_sym
    ]


class AssetPrices:
    """Single-series pipeline (stocks, crypto, forex): load → adjust for splits → backfill
    → adjust for dividends. Sibling of `OptionsPrices`; both are composed by `Prices`.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir, self.asset_types = check_data_dir(data_dir)
        # Predecessors discovered by the last `load_prices` (ticker renames, stocks only); read by
        # `Prices.process` to stitch the same renames onto the options pass. Empty when none/non-stock.
        self.predecessors: list[Predecessor] = []
        # Corporate actions the last `get_prices` adjusted for (splits/renames/dividends), surfaced
        # on the verification plot via `Prices.process`. Reflects what was actually applied.
        self.events: list[PriceEvent] = []

    def get_prices(
        self,
        ticker: str,
        date_start: str | None = None,
        date_end: str | None = None,
        dividends: bool = False,
    ) -> tuple[pd.DataFrame, AssetType]:
        """Get prices for a given ticker and date range, paired with the detected asset type.
        `dividends` opts into cash-dividend back-adjustment (off by default — the saved series
        is then the actual split-adjusted traded price).
        """
        self.events = []
        df, asset_type = self.load_prices(ticker, date_start, date_end)
        if self.predecessors:
            first = cast(pd.Timestamp, df.index[0]).tz_convert("America/New_York").date()
            last = cast(pd.Timestamp, df.index[-1]).tz_convert("America/New_York").date()
            self.events += _rename_events(ticker, self.predecessors, first, last)
        df = self.adjust_prices(df, asset_type, date_start, date_end, dividends)
        describe_adjusted_prices(df, ticker)
        return df, asset_type

    def load_prices(
        self, ticker: str, date_start: str | None = None, date_end: str | None = None
    ) -> tuple[pd.DataFrame, AssetType]:
        """Load prices for a given ticker and date range."""
        print(f"\n⛏️  Loading {ticker} price data...")
        df, asset_type = load_ticker_data(
            self.data_dir, self.asset_types, ticker, date_start, date_end
        )
        if "window_start" not in df.columns:
            raise ValueError(f"No window_start column found in data for ticker: `{ticker}`")

        # Stitch in any predecessor symbol (ticker rename, e.g. QQQ ⇄ QQQQ) so a series split
        # across a rename loads continuous instead of leaving a synthetic interpolated gap. The
        # discovered predecessors are cached so the options pass can stitch the same renames.
        self.predecessors = []
        if asset_type == AssetType.STOCKS:
            df, self.predecessors = stitch_predecessors(
                self.data_dir, asset_type, ticker, df, date_start, date_end
            )

        df["timestamp_utc"] = pd.to_datetime(df["window_start"], unit="ns", utc=True)
        df = df.sort_values("timestamp_utc").set_index("timestamp_utc")
        df = pd.DataFrame(df[["close"]]).rename(columns={"close": ticker})

        start_date = parse_date(cast(pd.Timestamp, df.index[0]))
        end_date = parse_date(cast(pd.Timestamp, df.index[-1]))
        print(f"🗑️  Loaded {len(df):,} price records for {ticker} from {start_date} to {end_date}")

        return df, asset_type

    def adjust_prices(
        self,
        df: pd.DataFrame,
        asset_type: AssetType,
        date_start: str | None = None,
        date_end: str | None = None,
        dividends: bool = False,
    ) -> pd.DataFrame:
        """Adjust prices for a given asset type via the following steps:
        1. Adjust for splits — before backfill, so the large split discontinuity isn't
           interpolated across (mirrors the options pipeline's split-before-backfill order).
        2. Backfill the missing 1-minute rows.
        3. Adjust for dividends — only when `dividends` is set; after backfill (the ~1%
           dividend discontinuity is negligible to interpolate across). Off by default so the
           output stays in the actual traded (split-only) currency that options align with.
        """
        print(f"⚙️  Adjusting {df.columns[0]} price data...")
        df = self.adjust_for_splits(df, asset_type)
        df = self.backfill_prices(df, asset_type, date_start, date_end)
        df = self.adjust_for_dividends(df, asset_type, dividends)
        return df

    def adjust_for_splits(self, df: pd.DataFrame, asset_type: AssetType) -> pd.DataFrame:
        """Back-adjust historical prices for stock splits using yfinance split data.
        For each split with ex-date D and ratio r, prices at timestamps < D are divided
        by r. Options need contract-level adjustment (strike + multiplier) and are skipped.
        """
        if asset_type != AssetType.STOCKS:
            print(f"🪚  Not adjusting {asset_type} assets for splits")
            return df

        ticker = df.columns[0]
        splits = fetch_splits(ticker, cast(pd.Timestamp, df.index[0]))
        if splits.empty:
            print(f"🪚  No splits to apply for {ticker}")
            return df

        df = df.copy()
        for split_date, ratio in splits.items():
            # ET-midnight, same as adjust_for_dividends: yfinance >=1.x stamps the split index at
            # 09:30 ET, so a raw `< ts` would divide the ex-date's own pre-market bars by the ratio.
            ts = cast(pd.Timestamp, split_date).tz_convert("America/New_York").normalize()
            df.loc[df.index < ts, ticker] /= ratio
            self.events.append(PriceEvent(ts.date(), f"{ratio:g}-for-1 split", "split"))
            print(f"🪚  Applied {ratio:g}-for-1 split on {ts.date()} to {ticker}")
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
        if len(df) > num_rows:
            print(f"🔧 Backfilled {len(df) - num_rows:,} rows")
        df.index.name = "timestamp_utc"
        return df

    def adjust_for_dividends(
        self, df: pd.DataFrame, asset_type: AssetType, dividends: bool = False
    ) -> pd.DataFrame:
        """Back-adjust historical prices for cash dividends using yfinance dividend data.
        For each dividend with ex-date D and amount d, prices at timestamps < D are scaled
        by (1 - d / c) where c is yfinance's official Close on the trading day before D.
        Runs after adjust_for_splits: yfinance dividends are reported in current-share-
        equivalent currency, matching the split-adjusted prices.
        """
        if asset_type != AssetType.STOCKS:
            print(f"🔩 Not adjusting {asset_type} assets for dividends")
            return df

        ticker = df.columns[0]
        if not dividends:
            print(f"🔩 Dividend adjustment for {ticker} is not requested, skipping")
            return df

        divs = fetch_dividends(ticker, cast(pd.Timestamp, df.index[0])).sort_index()
        if divs.empty:
            print(f"🔩 No dividends to apply for {ticker}")
            return df

        yf_closes = fetch_yf_closes(
            ticker,
            cast(pd.Timestamp, df.index[0]),
            cast(pd.Timestamp, divs.index[-1]) + pd.Timedelta(days=1),
        )
        df = df.copy()
        for ex_date, amount in divs.items():
            # ET-midnight ex-date: yfinance >=1.x stamps the dividend index at 09:30 ET (not
            # midnight), so a raw `< ts` would pick the ex-date's own post-drop close as prev_close.
            ts = cast(pd.Timestamp, ex_date).tz_convert("America/New_York").normalize()
            mask = df.index < ts
            if not mask.any():
                continue
            prev_closes = yf_closes[yf_closes.index < ts]
            if prev_closes.empty:
                print(f"🔩 Skipping {ts.date()} dividend: no yfinance close before ex-date")
                continue
            prev_close = prev_closes.iloc[-1]
            factor = 1 - amount / prev_close
            df.loc[mask, ticker] *= factor
            self.events.append(PriceEvent(ts.date(), f"${amount:.2f} dividend", "dividend"))
            print(
                f"🔩 Applied ${amount:.4f} dividend on {ts.date()} "
                f"(factor {factor:.6f}) to {ticker}"
            )
        return df
