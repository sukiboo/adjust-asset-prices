from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf

from .schemas import ChecksConfig

sns.set_theme(style="darkgrid", palette="muted", font="monospace", rc={"lines.linewidth": 2})


def check_prices(df: pd.DataFrame, config: ChecksConfig) -> bool:
    """Collection of sanity checks for the price data."""
    print(f"\n🔍 Checking {df.columns[0]} price data...")
    return all(
        [
            check_for_gaps(
                df,
                gap_threshold_mins=config["gap_threshold_mins"],
                num_gaps_display=config["num_gaps_display"],
            ),
            compare_to_yf(
                df,
                abs_rel_diff_pct_p50=config["abs_rel_diff_pct_p50"],
                abs_rel_diff_pct_p99=config["abs_rel_diff_pct_p99"],
                show_plot=config["show_plot"],
            ),
        ]
    )


def check_for_gaps(df: pd.DataFrame, gap_threshold_mins: int, num_gaps_display: int) -> bool:
    """Check for gaps in the price data where adjacent timestamps are longer
    than `gap_threshold_mins` minutes apart.
    """
    time_diffs = df.index.to_series().diff()
    gap_mask = time_diffs > pd.Timedelta(minutes=gap_threshold_mins)
    gaps_series = cast(pd.Series, time_diffs[gap_mask])

    if len(gaps_series) == 0:
        print("✔️  No gaps found in the price data")
        return True

    gap_info = []
    for gap_end in gaps_series.index:
        gap_duration = gaps_series.loc[gap_end]
        gap_start = gap_end - gap_duration
        gap_info.append({"gap_start": gap_start, "gap_end": gap_end, "gap_duration": gap_duration})

    gap_df = pd.DataFrame(gap_info)
    gap_df = gap_df.sort_values("gap_duration", ascending=False).head(num_gaps_display)

    print(f"❗ Found {len(gaps_series)} gaps larger than {gap_threshold_mins} minutes:")
    for idx, row in gap_df.iterrows():
        print(f"{row['gap_duration']}: {row['gap_start']} -> {row['gap_end']}")

    return False


def compare_to_yf(
    df: pd.DataFrame,
    abs_rel_diff_pct_p50: float,
    abs_rel_diff_pct_p99: float,
    show_plot: bool,
) -> bool:
    """Compare the price data to Yahoo Finance.
    Displays a plot of the price data and the difference between the two datasets.

    Both datasets are normalized to UTC day boundaries for comparison.
    """
    ticker = df.columns[0]
    start_date = cast(pd.Timestamp, df.index[0])
    end_date = cast(pd.Timestamp, df.index[-1])

    # Ensure our data is UTC-aware before resampling to get consistent day boundaries
    our_df = df.copy()
    if our_df.index.tz is None:  # type: ignore[attr-defined]
        our_df.index = our_df.index.tz_localize("UTC")  # type: ignore[attr-defined]
    else:
        our_df.index = our_df.index.tz_convert("UTC")  # type: ignore[attr-defined]

    # Resample to daily using UTC day boundaries (last price before midnight UTC)
    our_daily = our_df[ticker].resample("D").last().dropna()
    # Convert to timezone-naive dates for comparison
    our_daily.index = our_daily.index.tz_localize(None).normalize()  # type: ignore[attr-defined]

    try:
        yf_df = yf.Ticker(ticker).history(start=start_date, end=end_date + pd.Timedelta(days=1))
        if yf_df.empty:
            print(f"⚠️  Warning: No yfinance data found for {ticker}")
            return False

        yf_daily = yf_df["Close"].copy()
        # yfinance returns timezone-aware timestamps; convert to UTC then strip timezone
        yf_daily_index = pd.to_datetime(yf_daily.index)
        if yf_daily_index.tz is not None:
            yf_daily_index = yf_daily_index.tz_convert("UTC").tz_localize(None)
        yf_daily.index = yf_daily_index.normalize()  # type: ignore[attr-defined]

        # Align datasets
        comparison = pd.concat([our_daily, yf_daily], axis=1).dropna()
        comparison.columns = ["our_close", "yf_close"]
        if comparison.empty:
            print("⚠️  Warning: No overlapping dates")
            return False

        # Calculate differences
        diff_usd = comparison["our_close"] - comparison["yf_close"]
        diff_pct = 100 * (diff_usd / comparison["yf_close"])

        # Print summary
        p50_abs_diff = diff_pct.abs().median()
        p99_abs_diff = diff_pct.abs().quantile(0.99)
        if p50_abs_diff > abs_rel_diff_pct_p50 and p99_abs_diff > abs_rel_diff_pct_p99:
            status = "❗"
        elif p50_abs_diff > 0.5 * abs_rel_diff_pct_p50 or p99_abs_diff > 0.5 * abs_rel_diff_pct_p99:
            status = "❕"
        else:
            status = "✔️ "
        p01_pct, p50_pct, p99_pct = diff_pct.quantile([0.01, 0.50, 0.99])
        p01_usd, p50_usd, p99_usd = diff_usd.quantile([0.01, 0.50, 0.99])
        print(
            f"{status} Price comparison over {len(comparison)} days (p01/p50/p99):"
            f" {p01_pct:.3f}% / {p50_pct:.3f}% / {p99_pct:.3f}%"
            f" = ${p01_usd:.2f} / ${p50_usd:.2f} / ${p99_usd:.2f}"
        )

        # Plot the price comparison
        if show_plot:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(comparison.index, comparison["our_close"], alpha=0.9, label="Adjusted prices")
            ax1.plot(comparison.index, comparison["yf_close"], alpha=0.9, label="Yahoo Finance")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Price ($)", color="black")
            ax1.legend(loc="upper left")

            ax2 = ax1.twinx()
            ax2.plot(
                comparison.index,
                diff_pct,
                color="red",
                linestyle=":",
                alpha=0.4,
                label="Relative diff",
            )
            ax2.set_ylim(-abs_rel_diff_pct_p99, abs_rel_diff_pct_p99)
            ax2.set_ylabel("Relative price difference (%)", color="red")
            ax2.grid(False)
            ax2.legend(loc="upper right")

            plt.title(f"{ticker} price comparison")
            plt.tight_layout()
            plt.show()

        return status != "❗"

    except Exception as e:
        print(f"⚠️  Error comparing with yfinance: {e}")
        return False
