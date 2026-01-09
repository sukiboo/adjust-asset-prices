from typing import cast

import pandas as pd

from .utils import check_data_dir, load_ticker_data, parse_date


class Prices:
    """
    Class to load prices for a given ticker and date range.
    Raw prices are loaded and backfilled to the start and end dates.
    For stocks and optiosn adjusted prices are calculated using the backfilled prices and the split and dividend data.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir, self.asset_types = check_data_dir(data_dir)

    def load_prices(
        self, ticker: str, date_start: str | None = None, date_end: str | None = None
    ) -> pd.DataFrame:
        """Load prices for a given ticker and date range."""
        df = load_ticker_data(self.data_dir, self.asset_types, ticker, date_start, date_end)
        if "window_start" not in df.columns:
            raise ValueError(f"No window_start column found in data for ticker: `{ticker}`")

        df["timestamp"] = pd.to_datetime(df["window_start"], unit="ns")
        df = df.sort_values("timestamp").set_index("timestamp")

        start_date = parse_date(cast(pd.Timestamp, df.index[0]))
        end_date = parse_date(cast(pd.Timestamp, df.index[-1]))
        print(
            f"Loaded {len(df)} price records for {ticker} "
            f"from {start_date} to {end_date}:\n{df.head()}"
        )

        return df
