from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf

from .schemas import AssetType, ChecksConfig, PriceFileFormat
from .utils import build_target_index, save_prices, verify_saved_prices

sns.set_theme(style="darkgrid", palette="muted", font="monospace", rc={"lines.linewidth": 2})


def check_prices(df: pd.DataFrame, config: ChecksConfig, asset_type: AssetType) -> bool:
    """Collection of sanity checks for the price data."""
    print(f"\n🔍 Checking {df.columns[0]} price data...")
    return (
        _index_matches_calendar(df, asset_type)
        and _prices_are_valid(df)
        and compare_to_yf(
            df,
            asset_type=asset_type,
            abs_rel_diff_pct_p50=config["abs_rel_diff_pct_p50"],
            abs_rel_diff_pct_p99=config["abs_rel_diff_pct_p99"],
            show_plot=config["show_plot"],
        )
    )


def save_if_valid(
    df: pd.DataFrame,
    save_dir: str,
    format: PriceFileFormat,
    config: ChecksConfig,
    asset_type: AssetType,
) -> bool:
    """Run checks; on success, save to disk and verify the round-trip."""
    if not check_prices(df, config=config, asset_type=asset_type):
        print("\n❌ Some checks failed, not saving the price data!")
        return False
    print("\n🎉 All checks passed, saving the price data...")
    save_prices(df, save_dir=save_dir, format=format)
    verify_saved_prices(df, save_dir=save_dir, format=format)
    return True


def _index_matches_calendar(df: pd.DataFrame, asset_type: AssetType) -> bool:
    """Verify df.index matches the calendar-aware target index for this asset type."""
    ticker = df.columns[0]
    expected_index = build_target_index(
        cast(pd.Timestamp, df.index[0]), cast(pd.Timestamp, df.index[-1]), asset_type
    )
    if not df.index.equals(expected_index):
        print(
            f"❗ {ticker} index does not match expected calendar: "
            f"{len(df)} rows vs {len(expected_index)} expected"
        )
        return False
    print(f"✔️  {ticker} index matches expected calendar")
    return True


def _prices_are_valid(df: pd.DataFrame) -> bool:
    """Verify every price is finite and strictly positive (NaN > 0 is False, so this
    also catches any missed NaN from the backfill).
    """
    ticker = df.columns[0]
    if not (df[ticker] > 0).all():
        n_bad = int((~(df[ticker] > 0)).sum())
        print(f"❗ {ticker} contains {n_bad} invalid prices (NaN or non-positive)")
        return False
    print(f"✔️  {ticker} contains only valid prices")
    return True


def compare_to_yf(
    df: pd.DataFrame,
    asset_type: AssetType,
    abs_rel_diff_pct_p50: float,
    abs_rel_diff_pct_p99: float,
    show_plot: bool,
) -> bool:
    """Compare the price data to Yahoo Finance.
    Displays a plot of the price data and the difference between the two datasets.

    For stocks/options, our daily close is the 1-min bar at 15:59 ET (regular-session
    close) so it lines up with yfinance's 4 PM ET Close. For other asset types, we
    fall back to UTC day boundaries since yfinance reports those at midnight UTC.
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

    if asset_type in (AssetType.STOCKS, AssetType.OPTIONS):
        # Pick the bar whose window_start is 15:59 ET — its close is the 4 PM ET print
        et_index = our_df.index.tz_convert("America/New_York")  # type: ignore[attr-defined]
        mask = (et_index.hour == 15) & (et_index.minute == 59)
        our_daily = cast(pd.Series, our_df.loc[mask, ticker])
        our_daily.index = pd.DatetimeIndex(et_index[mask]).normalize().tz_localize(None)
    else:
        # Crypto/forex: yfinance reports daily Close at midnight UTC
        our_daily = our_df[ticker].resample("D").last().dropna()
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
            _, ax1 = plt.subplots(figsize=(12, 6))
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
