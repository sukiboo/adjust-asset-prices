from typing import cast

import numpy as np
import pandas as pd

from .schemas import AssetType
from .utils import (
    build_target_index,
    check_data_dir,
    fetch_dividends,
    fetch_splits,
    fetch_yf_closes,
    load_ticker_data,
    parse_date,
)


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
        df = self.adjust_prices(df, asset_type)
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

    def adjust_prices(self, df: pd.DataFrame, asset_type: AssetType) -> pd.DataFrame:
        """Adjust prices for a given asset type via the following steps:
        1. Backfill the missing prices
        2. Adjust for splits (yfinance reports dividends in current-share-equivalent
           currency, so we put prices in that currency first)
        3. Adjust for dividends
        """
        print(f"⚙️  Adjusting {df.columns[0]} price data...")
        df = self.backfill_prices(df, asset_type)
        df = self.adjust_splits(df, asset_type)
        df = self.adjust_dividends(df, asset_type)
        return df

    def backfill_prices(self, df: pd.DataFrame, asset_type: AssetType) -> pd.DataFrame:
        """Backfill missing 1-minute rows over the appropriate trading calendar.
        Stocks use NYSE extended hours (04:00-19:59 ET on session days); options use NYSE
        regular hours (09:30-15:59 ET on session days). Half-days are handled by the
        calendar. Crypto and forex use a continuous 1-min grid. Interpolation runs in log
        space (log -> linear interpolate -> exp) to preserve multiplicative behavior.
        """
        col, num_rows = df.columns[0], len(df)
        target_index = build_target_index(
            cast(pd.Timestamp, df.index[0]), cast(pd.Timestamp, df.index[-1]), asset_type
        )
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
