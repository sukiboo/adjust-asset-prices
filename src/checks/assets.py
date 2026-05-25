from typing import cast

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
import yfinance as yf

from ..schemas import AssetType, ChecksConfig, PriceFileFormat
from ..utils import build_target_index, save_prices, verify_saved_prices

sns.set_theme(style="darkgrid", palette="muted", font="monospace", rc={"lines.linewidth": 2})


def check_prices(
    df: pd.DataFrame,
    config: dict[AssetType, ChecksConfig],
    asset_type: AssetType,
    show_plot: bool,
    dividends_adjusted: bool,
) -> bool:
    """Collection of sanity checks for the price data. `dividends_adjusted` reflects whether
    `df` was dividend-adjusted, so `compare_to_yf` picks the matching yfinance convention.
    """
    print(f"\n🔍 Checking {df.columns[0]} price data...")
    thresholds = config[asset_type]
    return (
        _index_matches_calendar(df, asset_type)
        and _prices_are_valid(df)
        and compare_to_yf(
            df,
            asset_type=asset_type,
            abs_rel_diff_pct_p50=thresholds["abs_rel_diff_pct_p50"],
            abs_rel_diff_pct_p99=thresholds["abs_rel_diff_pct_p99"],
            show_plot=show_plot,
            dividends_adjusted=dividends_adjusted,
        )
    )


def save_if_valid(
    df: pd.DataFrame,
    save_dir: str,
    format: PriceFileFormat,
    config: dict[AssetType, ChecksConfig],
    asset_type: AssetType,
    show_plot: bool,
    dividends_adjusted: bool,
) -> bool:
    """Run checks; on success, save to disk and verify the round-trip."""
    if not check_prices(
        df,
        config=config,
        asset_type=asset_type,
        show_plot=show_plot,
        dividends_adjusted=dividends_adjusted,
    ):
        print("\n❌ Some checks failed, not saving the price data!")
        return False
    print("\n🎉 All checks passed, saving the price data...")
    save_prices(df, save_dir=save_dir, format=format)
    verify_saved_prices(df, save_dir=save_dir, format=format)
    return True


def _index_matches_calendar(df: pd.DataFrame, asset_type: AssetType) -> bool:
    """Verify df.index matches the calendar-aware target index for this asset type."""
    ticker = df.columns[0]
    start = cast(pd.Timestamp, df.index[0])
    end = cast(pd.Timestamp, df.index[-1])
    if asset_type in (AssetType.STOCKS, AssetType.OPTIONS):
        # Round-trip via ET so .date() inside build_target_index sees the trading day,
        # not the UTC day (last EST session bar lands at 00:59 UTC the next day).
        start = start.tz_convert("America/New_York")
        end = end.tz_convert("America/New_York")
    expected_index = build_target_index(start, end, asset_type)
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


def _our_daily_close(df: pd.DataFrame, asset_type: AssetType) -> pd.Series:
    """Reduce our 1-min bars to one daily close per calendar day, aligned to whichever
    boundary yfinance uses for that asset type's daily Close:
    - NYSE assets: the 15:59 ET bar (its close is the 4 PM ET regular-session print).
    - Crypto: the last 1-min bar per UTC day (yfinance reports at midnight UTC).
    - Forex: the bar at 00:00 Europe/London (yfinance's daily index labels each row
      with London midnight and reports Close == Open == that snapshot price; the
      systematic ~1h UTC offset matters on volatile pairs like USD-JPY).
    Returns a tz-naive Series indexed by normalized day.
    """
    ticker = df.columns[0]
    s = df[ticker].copy()
    idx = cast(pd.DatetimeIndex, s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
        s.index = idx

    if asset_type in (AssetType.STOCKS, AssetType.OPTIONS):
        et_idx = idx.tz_convert("America/New_York")
        mask = (et_idx.hour == 15) & (et_idx.minute == 59)
        daily = s.loc[mask].copy()
        daily.index = pd.DatetimeIndex(et_idx[mask]).normalize().tz_localize(None)
        return daily

    if asset_type == AssetType.CRYPTO:
        daily = s.resample("D").last().dropna()
        daily.index = daily.index.tz_localize(None).normalize()  # type: ignore[attr-defined]
        return daily

    if asset_type == AssetType.FOREX:
        london_idx = idx.tz_convert("Europe/London")
        mask = (london_idx.hour == 0) & (london_idx.minute == 0)
        daily = s.loc[mask].copy()
        # Index by London wall-clock date to match yfinance's row labels. Indexing by
        # UTC date instead would collapse Sun-GMT (UTC 00:00) and Mon-BST (UTC 23:00
        # prior day) onto the same UTC date around the spring DST transition.
        daily.index = pd.DatetimeIndex(london_idx[mask]).tz_localize(None).normalize()
        return daily

    raise ValueError(f"Unsupported asset type for daily-close reduction: {asset_type}")


def _yf_daily_close(
    ticker: str,
    asset_type: AssetType,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dividends_adjusted: bool,
) -> pd.Series | None:
    """Fetch yfinance's daily Close over [start, end] inclusive, normalized to a tz-naive
    day index. `dividends_adjusted` picks the convention to match our output:
    `auto_adjust=True` → dividend-adjusted Close, `auto_adjust=False` → split-adjusted but
    dividend-unadjusted Close. Translates `EUR-USD` → `EURUSD=X` for forex (yfinance returns
    404 on the hyphenated form). Returns None when yfinance has no data for the ticker.
    """
    yf_ticker = f"{ticker.replace('-', '')}=X" if asset_type == AssetType.FOREX else ticker
    yf_df = yf.Ticker(yf_ticker).history(
        start=start, end=end + pd.Timedelta(days=1), auto_adjust=dividends_adjusted
    )
    if yf_df.empty:
        return None
    daily = yf_df["Close"].copy()
    idx = pd.to_datetime(daily.index)
    # Forex rows are tz-aware Europe/London and their date *label* is the London date —
    # converting to UTC first would shift BST rows back one day. For other asset types
    # the UTC normalization is benign (NY midnight is already on the NY date in UTC).
    if asset_type == AssetType.FOREX:
        daily.index = idx.tz_localize(None).normalize()
    else:
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        daily.index = idx.normalize()
    return daily


def compare_to_yf(
    df: pd.DataFrame,
    asset_type: AssetType,
    abs_rel_diff_pct_p50: float,
    abs_rel_diff_pct_p99: float,
    show_plot: bool,
    dividends_adjusted: bool,
) -> bool:
    """Compare the price data to Yahoo Finance. `dividends_adjusted` must reflect whether
    `df` was dividend-adjusted, so we compare against the matching yfinance Close convention.
    Displays a plot of the price data and the difference between the two datasets.
    """
    ticker = df.columns[0]
    start_date = cast(pd.Timestamp, df.index[0])
    end_date = cast(pd.Timestamp, df.index[-1])

    try:
        yf_daily = _yf_daily_close(ticker, asset_type, start_date, end_date, dividends_adjusted)
        if yf_daily is None:
            print(f"⚠️ Warning: No yfinance data found for {ticker}")
            return False

        comparison = pd.concat([_our_daily_close(df, asset_type), yf_daily], axis=1).dropna()
        comparison.columns = ["our_close", "yf_close"]
        if comparison.empty:
            print("⚠️ Warning: No overlapping dates")
            return False

        diff_usd = comparison["our_close"] - comparison["yf_close"]
        diff_pct = 100 * (diff_usd / comparison["yf_close"])

        passed = _diff_passes_thresholds(
            diff_pct, diff_usd, abs_rel_diff_pct_p50, abs_rel_diff_pct_p99
        )
        if show_plot:
            _plot_comparison(comparison, diff_pct, ticker, abs_rel_diff_pct_p99)
        return passed

    except Exception as e:
        print(f"⚠️ Error comparing with yfinance: {e}")
        return False


def _diff_passes_thresholds(
    diff_pct: pd.Series,
    diff_usd: pd.Series,
    abs_rel_diff_pct_p50: float,
    abs_rel_diff_pct_p99: float,
) -> bool:
    # OR gate on |diff| quantiles; warn band at half threshold (also OR).
    abs_diff = diff_pct.abs()
    metrics = [
        ("abs_p50", abs_diff.median(), abs_rel_diff_pct_p50),
        ("abs_p99", abs_diff.quantile(0.99), abs_rel_diff_pct_p99),
    ]
    fail = any(v > t for _, v, t in metrics)
    warn = any(v > 0.5 * t for _, v, t in metrics)
    status = "❗" if fail else "❕" if warn else "✔️ "

    p01_pct, p50_pct, p99_pct = diff_pct.quantile([0.01, 0.50, 0.99])
    p01_usd, p50_usd, p99_usd = diff_usd.quantile([0.01, 0.50, 0.99])
    print(
        f"{status} Price comparison over {len(diff_pct)} days (p01/p50/p99):"
        f" {p01_pct:.3f}% / {p50_pct:.3f}% / {p99_pct:.3f}%"
        f" = ${p01_usd:.2f} / ${p50_usd:.2f} / ${p99_usd:.2f}"
    )
    if fail:
        msg = " and ".join(f"{n} = {v:.2f}% > {t:.2f}%" for n, v, t in metrics if v > t)
        print(f"‼️  Price differences violate the threshold: {msg}")
    return not fail


def _plot_comparison(
    comparison: pd.DataFrame, diff_pct: pd.Series, ticker: str, diff_ylim: float
) -> None:
    # Tailwind blue-500 / green-500 / red-400.
    price_adj, price_yf, diff_color = "#3b82f6", "#22c55e", "#f87171"

    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Both series share thickness and carry opacity so the overlap (a near-perfect
    # match) blends visibly instead of one line hiding the other.
    (adj_line,) = ax1.plot(
        comparison.index,
        comparison["our_close"],
        color=price_adj,
        linewidth=3,
        alpha=0.75,
        label="Adjusted prices",
        zorder=2,
    )
    (yf_line,) = ax1.plot(
        comparison.index,
        comparison["yf_close"],
        color=price_yf,
        linewidth=3,
        alpha=0.75,
        label="Yahoo Finance",
        zorder=1,
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price ($)")
    ax1.margins(x=0.01)

    ax2 = ax1.twinx()
    ax2.axhline(0, color=diff_color, linewidth=0.8, alpha=0.3)
    ax2.fill_between(comparison.index, diff_pct, 0, color=diff_color, alpha=0.1, zorder=0)
    (diff_line,) = ax2.plot(
        comparison.index,
        diff_pct,
        color=diff_color,
        linewidth=1.2,
        alpha=0.7,
        label="Relative diff",
    )
    ax2.set_ylim(-diff_ylim, diff_ylim)
    ax2.set_ylabel("Relative price difference", color=diff_color)
    ax2.tick_params(axis="y", colors=diff_color)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f%%"))
    ax2.grid(False)

    handles = [adj_line, yf_line, diff_line]
    ax1.legend(handles, [str(h.get_label()) for h in handles], loc="upper left", framealpha=0.9)

    ax1.set_title(f"{ticker} price comparison", fontsize=14, pad=12)
    fig.tight_layout()
    plt.show()
