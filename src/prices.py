from typing import cast

import pandas as pd

from .schemas import AssetType
from .utils import check_data_dir, load_ticker_data, parse_date


class Prices:
    """Class to load prices for a given ticker and date range.
    Raw prices are loaded and backfilled to the start and end dates.
    For stocks and options adjusted prices are calculated using the
    backfilled prices and the split and dividend data.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir, self.asset_types = check_data_dir(data_dir)

    def get_prices(
        self, ticker: str, date_start: str | None = None, date_end: str | None = None
    ) -> pd.DataFrame:
        """Get prices for a given ticker and date range."""
        df, asset_type = self.load_prices(ticker, date_start, date_end)
        df = self.adjust_prices(df, asset_type)
        return df

    def load_prices(
        self, ticker: str, date_start: str | None = None, date_end: str | None = None
    ) -> tuple[pd.DataFrame, AssetType]:
        """Load prices for a given ticker and date range."""
        df, asset_type = load_ticker_data(
            self.data_dir, self.asset_types, ticker, date_start, date_end
        )
        if "window_start" not in df.columns:
            raise ValueError(f"No window_start column found in data for ticker: `{ticker}`")

        df["timestamp"] = pd.to_datetime(df["window_start"], unit="ns")
        df = df.sort_values("timestamp").set_index("timestamp")
        df = pd.DataFrame(df[["close"]]).rename(columns={"close": ticker})

        start_date = parse_date(cast(pd.Timestamp, df.index[0]))
        end_date = parse_date(cast(pd.Timestamp, df.index[-1]))
        print(
            f"Loaded {len(df)} price records for {ticker} "
            f"from {start_date} to {end_date}:\n{df.head()}"
        )

        return df, asset_type

    def adjust_prices(self, df: pd.DataFrame, asset_type: AssetType) -> pd.DataFrame:
        """Adjust prices for a given asset type via the following steps:
        1. Backfill the missing prices
        2. Adjust for splits
        3. Adjust for dividends
        """
        self.backfill_prices(df)
        self.adjust_splits(df, asset_type)
        self.adjust_dividends(df, asset_type)
        return df

    def backfill_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backfill the missing price values."""
        df = df.ffill().bfill()
        return df

    def adjust_splits(self, df: pd.DataFrame, asset_type: AssetType) -> pd.DataFrame:
        """Adjust for historical prices for stock splits/merges."""
        if asset_type not in [AssetType.STOCKS, AssetType.OPTIONS]:
            return df
        else:
            print(f"Adjusting for splits/merges is not yet implemented")
            return df

    def adjust_dividends(self, df: pd.DataFrame, asset_type: AssetType) -> pd.DataFrame:
        """Adjust for historical prices for stock dividends."""
        if asset_type != AssetType.STOCKS:
            return df
        else:
            print(f"Adjusting for dividends is not yet implemented")
            return df
