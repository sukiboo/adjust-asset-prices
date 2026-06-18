import gc
from collections.abc import Callable, Sequence
from typing import cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
import yfinance as yf
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.typing import LineStyleType

from ..schemas import AssetType, ChecksConfig, PriceEvent, PriceFileFormat
from ..utils import build_target_index, save_prices, verify_saved_prices, yf_retry

sns.set_theme(style="darkgrid", palette="muted", font="monospace", rc={"lines.linewidth": 2})


def check_prices(
    df: pd.DataFrame,
    config: dict[AssetType, ChecksConfig],
    asset_type: AssetType,
    show_plot: bool,
    dividends_adjusted: bool,
    events: Sequence[PriceEvent] = (),
) -> bool:
    """Collection of sanity checks for the price data. `dividends_adjusted` reflects whether
    `df` was dividend-adjusted, so `compare_to_yf` picks the matching yfinance convention.
    `events` are corporate actions to mark on the verification plot (only used when `show_plot`).
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
            yf_max_missing_run_sessions=thresholds["yf_max_missing_run_sessions"],
            show_plot=show_plot,
            dividends_adjusted=dividends_adjusted,
            events=events,
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
    events: Sequence[PriceEvent] = (),
    confirm_on_fail: Callable[[], bool] | None = None,
) -> bool:
    """Run checks; on success, save to disk and verify the round-trip. When checks fail,
    `confirm_on_fail` (if given) is invoked and the data is saved anyway iff it returns True —
    the CLI's interactive override. Default `None` keeps the strict "never save on fail"
    behavior, so non-CLI callers (tests) are unaffected and never block on input. `events` are
    the corporate actions to mark on the verification plot (only used when `show_plot`)."""
    if not check_prices(
        df,
        config=config,
        asset_type=asset_type,
        show_plot=show_plot,
        dividends_adjusted=dividends_adjusted,
        events=events,
    ):
        if confirm_on_fail is None or not confirm_on_fail():
            print("\n❌ Some checks failed, not saving the price data!")
            return False
    else:
        print("\n🎉 All checks passed, saving the price data...")
    save_prices(
        df, save_dir=save_dir, format=format, asset_type=asset_type, dividends=dividends_adjusted
    )
    verify_saved_prices(
        df, save_dir=save_dir, format=format, asset_type=asset_type, dividends=dividends_adjusted
    )
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
    yf_df = yf_retry(
        lambda: yf.Ticker(yf_ticker).history(
            start=start, end=end + pd.Timedelta(days=1), auto_adjust=dividends_adjusted
        ),
        f"{yf_ticker} daily close",
        retry_empty=lambda df: df.empty,
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


def _max_missing_run(our_index: pd.Index, yf_index: pd.Index) -> int:
    """Longest contiguous run of our daily dates that yfinance has no data for. A large run means
    our series contains a span yfinance can't corroborate (a wrong stitch or ticker); small
    scattered gaps (yfinance hiccups, weekend forex) stay near zero."""
    yf_dates = set(yf_index)
    best = run = 0
    for d in sorted(our_index):
        run = run + 1 if d not in yf_dates else 0
        best = max(best, run)
    return best


def compare_to_yf(
    df: pd.DataFrame,
    asset_type: AssetType,
    abs_rel_diff_pct_p50: float,
    abs_rel_diff_pct_p99: float,
    yf_max_missing_run_sessions: int,
    show_plot: bool,
    dividends_adjusted: bool,
    events: Sequence[PriceEvent] = (),
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
            print(
                f"❌ {ticker}: yfinance has no data -- likely a retired/renamed ticker. "
                f"Request the current symbol instead (e.g. META rather than FB) to get the "
                f"full history; the rename's old-symbol bars are stitched in automatically."
            )
            return False

        our_daily = _our_daily_close(df, asset_type)
        # Stocks only: the check backstops the stock-only rename stitcher. Crypto/forex never
        # stitch, so an uncorroborated run is just yfinance's shorter history, not a wrong stitch.
        if asset_type == AssetType.STOCKS:
            missing_run = _max_missing_run(our_daily.index, yf_daily.index)
            if missing_run > yf_max_missing_run_sessions:
                print(
                    f"❌ {ticker}: yfinance is missing {missing_run} consecutive sessions our data "
                    f"covers -- that span is uncorroborated (likely a wrong stitch), not saving."
                )
                return False

        comparison = pd.concat([our_daily, yf_daily], axis=1).dropna()
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
            _plot_comparison(comparison, diff_pct, ticker, abs_rel_diff_pct_p99, events)
            gc.collect()
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


# Marker style per event kind: (color, linestyle). Tailwind purple-500 / amber-500 / teal-500.
_EVENT_STYLES: dict[str, tuple[str, LineStyleType]] = {
    "split": ("#a855f7", "-."),
    "rename": ("#f59e0b", "--"),
    "dividend": ("#14b8a6", ":"),
}
_EVENT_LEGEND = {"split": "Splits", "rename": "Rename", "dividend": "Dividends"}


def _mark_events(
    ax: Axes, events: Sequence[PriceEvent], xmin: pd.Timestamp, xmax: pd.Timestamp
) -> tuple[list[Line2D], list[str]]:
    """Draw a vertical line per corporate action within [xmin, xmax]; annotate splits/renames
    (few, informative) and leave dividends as faint unlabeled lines (often many). Returns one
    legend proxy + label per event kind present."""
    visible = [e for e in events if xmin <= pd.Timestamp(e.date) <= xmax]
    for e in visible:
        color, ls = _EVENT_STYLES[e.kind]
        x = float(mdates.date2num(e.date))
        ax.axvline(x, color=color, linestyle=ls, linewidth=1, alpha=0.6)
        if e.kind != "dividend":
            ax.annotate(
                e.label,
                xy=(x, 1),
                xycoords=("data", "axes fraction"),
                xytext=(3, -4),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
                color=color,
            )
    kinds = [k for k in _EVENT_STYLES if any(e.kind == k for e in visible)]
    return [
        Line2D([0], [0], color=_EVENT_STYLES[k][0], linestyle=_EVENT_STYLES[k][1], linewidth=1.5)
        for k in kinds
    ], [_EVENT_LEGEND[k] for k in kinds]


def _plot_comparison(
    comparison: pd.DataFrame,
    diff_pct: pd.Series,
    ticker: str,
    diff_threshold: float,
    events: Sequence[PriceEvent] = (),
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
        linewidth=2,
        alpha=0.75,
        label="Adjusted prices",
        zorder=2,
    )
    (yf_line,) = ax1.plot(
        comparison.index,
        comparison["yf_close"],
        color=price_yf,
        linewidth=2,
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
        linewidth=1,
        alpha=0.6,
        label="Relative diff",
    )
    # Scale the diff axis to 2x the p99 threshold so the tail spikes that legitimately exceed
    # the gate's p99 limit (only ~1% of points need to stay under it) are visible, not clipped.
    ax2.set_ylim(-2 * diff_threshold, 2 * diff_threshold)
    ax2.set_ylabel("Relative price difference", color=diff_color)
    ax2.tick_params(axis="y", colors=diff_color)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f%%"))
    ax2.grid(False)

    event_handles, event_labels = _mark_events(
        ax1, events, comparison.index[0], comparison.index[-1]
    )
    handles = [adj_line, yf_line, diff_line, *event_handles]
    labels = [str(h.get_label()) for h in (adj_line, yf_line, diff_line)] + event_labels
    ax1.legend(handles, labels, loc="lower right", framealpha=0.9)

    ax1.set_title(f"{ticker} price verification vs yfinance", fontsize=14, pad=12)
    fig.tight_layout()
    plt.show()
    plt.close(fig)  # drop pyplot's reference; caller forces the Tk-cycle collection (see below)
