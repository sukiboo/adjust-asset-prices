from typing import cast

import pandas as pd

from src import Prices


def check_for_gaps(df: pd.DataFrame) -> None:
    """Check for gaps in the price data where adjacent timestamps are longer than 1 minute apart."""
    # Calculate time differences between consecutive rows
    # time_diffs[i] is the difference between df.index[i] and df.index[i-1]
    time_diffs = df.index.to_series().diff()

    # Filter gaps longer than 1 minute (exclude NaN from first row)
    gap_mask = time_diffs > pd.Timedelta(minutes=1)
    gaps_series = cast(pd.Series, time_diffs[gap_mask])

    if len(gaps_series) == 0:
        print("No gaps found in the price data (all intervals are 1 minute or less)")
        return

    # For each gap, gap_end is the timestamp where the gap ends (current row)
    # gap_start is the timestamp before the gap (previous row)
    # gap_duration is already in gaps_series.values
    gap_info = []
    for gap_end in gaps_series.index:
        gap_duration = gaps_series.loc[gap_end]
        gap_start = gap_end - gap_duration
        gap_info.append({"gap_start": gap_start, "gap_end": gap_end, "gap_duration": gap_duration})

    # Convert to DataFrame and sort by gap duration
    gap_df = pd.DataFrame(gap_info)
    gap_df = gap_df.sort_values("gap_duration", ascending=False).head(10)

    print(f"\nFound {len(gaps_series)} gaps. Largest gaps:")
    for idx, row in gap_df.iterrows():
        print(f"{row['gap_duration']}: {row['gap_start']} -> {row['gap_end']}")


if __name__ == "__main__":
    prices = Prices(data_dir="./data/files")
    df = prices.get_prices(ticker="BTC-USD", date_start="2025-01-01", date_end=None)
    check_for_gaps(df)
