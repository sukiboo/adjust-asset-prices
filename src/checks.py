from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

from .schemas import AssetType, ChecksConfig, OptionsChecksConfig, PriceFileFormat
from .utils import (
    build_target_index,
    parse_osi_ticker,
    save_options,
    save_prices,
    verify_saved_options,
    verify_saved_prices,
)

sns.set_theme(style="darkgrid", palette="muted", font="monospace", rc={"lines.linewidth": 2})


def check_prices(
    df: pd.DataFrame,
    config: dict[AssetType, ChecksConfig],
    asset_type: AssetType,
    show_plot: bool,
) -> bool:
    """Collection of sanity checks for the price data."""
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
        )
    )


def save_if_valid(
    df: pd.DataFrame,
    save_dir: str,
    format: PriceFileFormat,
    config: dict[AssetType, ChecksConfig],
    asset_type: AssetType,
    show_plot: bool,
) -> bool:
    """Run checks; on success, save to disk and verify the round-trip."""
    if not check_prices(df, config=config, asset_type=asset_type, show_plot=show_plot):
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
    ticker: str, asset_type: AssetType, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series | None:
    """Fetch yfinance's daily Close over [start, end] inclusive, normalized to a tz-naive
    day index. Translates `EUR-USD` → `EURUSD=X` for forex (yfinance returns 404 on the
    hyphenated form). Returns None when yfinance has no data for the ticker.
    """
    yf_ticker = f"{ticker.replace('-', '')}=X" if asset_type == AssetType.FOREX else ticker
    yf_df = yf.Ticker(yf_ticker).history(start=start, end=end + pd.Timedelta(days=1))
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
) -> bool:
    """Compare the price data to Yahoo Finance.
    Displays a plot of the price data and the difference between the two datasets.
    """
    ticker = df.columns[0]
    start_date = cast(pd.Timestamp, df.index[0])
    end_date = cast(pd.Timestamp, df.index[-1])

    try:
        yf_daily = _yf_daily_close(ticker, asset_type, start_date, end_date)
        if yf_daily is None:
            print(f"⚠️  Warning: No yfinance data found for {ticker}")
            return False

        comparison = pd.concat([_our_daily_close(df, asset_type), yf_daily], axis=1).dropna()
        comparison.columns = ["our_close", "yf_close"]
        if comparison.empty:
            print("⚠️  Warning: No overlapping dates")
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
        print(f"⚠️  Error comparing with yfinance: {e}")
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
    _, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(comparison.index, comparison["our_close"], alpha=0.9, label="Adjusted prices")
    ax1.plot(comparison.index, comparison["yf_close"], alpha=0.9, label="Yahoo Finance")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price ($)", color="black")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        comparison.index, diff_pct, color="red", linestyle=":", alpha=0.4, label="Relative diff"
    )
    ax2.set_ylim(-diff_ylim, diff_ylim)
    ax2.set_ylabel("Relative price difference (%)", color="red")
    ax2.grid(False)
    ax2.legend(loc="upper right")

    plt.title(f"{ticker} price comparison")
    plt.tight_layout()
    plt.show()


def check_options(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    underlying: str,
    underlying_df: pd.DataFrame,
    config: OptionsChecksConfig,
) -> bool:
    """Structural gate for the options output. yfinance has no historical per-contract
    series so external comparison is impossible — instead we check self-consistency
    against the underlying via four no-arb bounds:

      1. positivity (P > 0, C > 0) — strict, every bar
      2. no bars past `parse_osi_ticker(t).expiry` — strict, every bar
      3. upper bounds: P ≤ K and C ≤ S — relative-p99-gated, near-the-money/OTM bars
      4. intrinsic floors: P ≥ max(K - S, 0), C ≥ max(S - K, 0) — same gating

    Checks 3-4 skip deep-ITM bars (intrinsic > `deep_itm_moneyness_cap` × underlying):
    those contracts are stock proxies whose illiquid last-trade prints breach the
    bounds routinely without a real arb. See `OptionsChecksConfig` for the rationale.

    `underlying_df` must be the SPLIT-ADJUSTED-ONLY underlying — option prices and
    intrinsic value use the underlying as it traded (cash dividends are priced into
    the premium ahead of ex-date, not back-adjusted out). Passing a dividend-adjusted
    series here overstates intrinsic on pre-dividend bars and produces false alarms.
    """
    print(f"\n🔍 Checking {underlying} options...")
    if calls.empty and puts.empty:
        print("⚠️  Both calls and puts empty; nothing to check")
        return False

    threshold_rel = config["noarb_violation_p99_rel"]
    deep_itm_cap = config["deep_itm_moneyness_cap"]
    if underlying not in underlying_df.columns:
        raise KeyError(
            f"underlying_df must have a column named `{underlying}`; got {list(underlying_df.columns)}"
        )
    underlying_series = underlying_df[underlying]

    results: list[bool] = []
    for side, df in [("calls", calls), ("puts", puts)]:
        if df.empty:
            print(f"⚠️  {side} side empty; skipping {side} checks")
            continue
        results.extend(_run_side_checks(side, df, underlying_series, threshold_rel, deep_itm_cap))
    return all(results) if results else False


def _run_side_checks(
    side: str,
    df: pd.DataFrame,
    underlying_series: pd.Series,
    threshold_rel: float,
    deep_itm_cap: float,
) -> list[bool]:
    """Run the four checks on one side (calls or puts). Strikes, the underlying snapshot,
    intrinsic value, and the deep-ITM mask are computed once and shared across checks."""
    ts_level = cast(pd.DatetimeIndex, df.index.get_level_values("timestamp_utc"))
    ticker_level = df.index.get_level_values("ticker")
    parsed_by_ticker = {t: parse_osi_ticker(cast(str, t)) for t in ticker_level.unique()}
    strikes = np.array([parsed_by_ticker[t].strike for t in ticker_level])
    underlying_at_t = cast(pd.Series, underlying_series.reindex(ts_level)).to_numpy()
    close = df["close"].to_numpy()

    # Intrinsic and the shallow (= not-deep-ITM) mask are shared by both no-arb bound
    # checks. Deep-ITM bars (intrinsic > cap × underlying) are dropped from those checks
    # only — positivity and expiry still cover every bar.
    if side == "calls":
        intrinsic = np.maximum(underlying_at_t - strikes, 0.0)
    else:
        intrinsic = np.maximum(strikes - underlying_at_t, 0.0)
    shallow = intrinsic <= deep_itm_cap * underlying_at_t

    return [
        _check_positive(side, close, ts_level, ticker_level),
        _check_no_bars_past_expiry(side, ts_level, ticker_level, parsed_by_ticker),
        _check_upper_bound(side, close, strikes, underlying_at_t, shallow, threshold_rel),
        _check_intrinsic_floor(side, close, intrinsic, underlying_at_t, shallow, threshold_rel),
    ]


def _format_examples(
    ts: pd.DatetimeIndex, tickers: pd.Index, mask: np.ndarray, values: np.ndarray, n: int = 3
) -> str:
    idx = np.flatnonzero(mask)[:n]
    parts = [f"{tickers[i]} @ {ts[i]} = {values[i]:.4f}" for i in idx]
    return "; ".join(parts)


def _check_positive(side: str, close: np.ndarray, ts: pd.DatetimeIndex, tickers: pd.Index) -> bool:
    bad = ~(close > 0)
    n_bad = int(bad.sum())
    if n_bad:
        print(
            f"❗ {side}: {n_bad:,} bars are not strictly positive "
            f"(worst: {_format_examples(ts, tickers, bad, close)})"
        )
        return False
    print(f"✔️  {side} > 0: all {len(close):,} bars positive")
    return True


def _check_no_bars_past_expiry(
    side: str, ts: pd.DatetimeIndex, tickers: pd.Index, parsed_by_ticker: dict
) -> bool:
    bar_dates = np.asarray(ts.tz_convert("America/New_York").date)
    expiries = np.asarray([parsed_by_ticker[t].expiry for t in tickers])
    bad = bar_dates > expiries
    n_bad = int(bad.sum())
    if n_bad:
        # Inline rather than reuse _format_examples — values here are dates, not
        # floats, and `:.4f` on a date silently renders the literal `.4f` (strftime
        # treats it as a template with no directives).
        idx = np.flatnonzero(bad)[:3]
        ex = "; ".join(
            f"{tickers[i]} @ {ts[i]} (bar date {bar_dates[i]} > expiry {expiries[i]})" for i in idx
        )
        print(f"❗ {side}: {n_bad:,} bars past contract expiry (worst: {ex})")
        return False
    print(f"✔️  {side} bars all within contract expiry")
    return True


def _check_upper_bound(
    side: str,
    close: np.ndarray,
    strikes: np.ndarray,
    underlying_at_t: np.ndarray,
    shallow: np.ndarray,
    threshold_rel: float,
) -> bool:
    """Calls: close ≤ underlying. Puts: close ≤ strike. Evaluated over non-deep-ITM bars
    (`shallow`) only — see `_run_side_checks`. Percentile-gated on the breach RELATIVE to
    the underlying price: a contract's last print lags spot during a down-move and can
    breach the bound by a few % without an exploitable arb. Isolated stale prints leave
    p99 at zero; a systematic mis-scaling breaches a large fraction and pushes p99 past
    the band.
    """
    label = "underlying" if side == "calls" else "strike"
    upper = underlying_at_t if side == "calls" else strikes
    n_excl = int((~shallow).sum())
    if not shallow.any():
        print(
            f"✔️  {side} ≤ {label}: no near-the-money bars to check ({n_excl:,} deep-ITM excluded)"
        )
        return True
    rel_breach = np.maximum(close[shallow] - upper[shallow], 0.0) / underlying_at_t[shallow]
    p50 = float(np.percentile(rel_breach, 50))
    p99 = float(np.percentile(rel_breach, 99))
    worst = float(rel_breach.max())
    n_viol = int((rel_breach > 0).sum())
    status = "❗" if p99 > threshold_rel else "✔️ "
    print(
        f"{status} {side} ≤ {label}: rel breach p50/p99/max = "
        f"{p50:.2%}/{p99:.2%}/{worst:.2%} over {n_viol:,} violating bars "
        f"(excl {n_excl:,} deep-ITM; p99 threshold {threshold_rel:.2%})"
    )
    return p99 <= threshold_rel


def _check_intrinsic_floor(
    side: str,
    close: np.ndarray,
    intrinsic: np.ndarray,
    underlying_at_t: np.ndarray,
    shallow: np.ndarray,
    threshold_rel: float,
) -> bool:
    """Calls: close ≥ max(S - K, 0). Puts: close ≥ max(K - S, 0). Evaluated over
    non-deep-ITM bars (`shallow`) only — see `_run_side_checks`. Percentile-gated on the
    shortfall RELATIVE to the underlying price: stale/off-market prints on illiquid ITM
    contracts routinely sit a few % below intrinsic (the last trade lags spot during a
    move) without an exploitable arb, and the noise scales with the price level — so a
    relative gate ports across assets where an absolute dollar gate wouldn't. A missed
    split, by contrast, shows up as a shortfall that is a large fraction of the
    underlying, far above the noise band — and one that hits near-the-money contracts
    too, so it survives the deep-ITM exclusion.
    """
    n_excl = int((~shallow).sum())
    if not shallow.any():
        print(
            f"✔️  {side} intrinsic floor: no near-the-money bars to check ({n_excl:,} deep-ITM excluded)"
        )
        return True
    rel_shortfall = np.maximum(intrinsic[shallow] - close[shallow], 0.0) / underlying_at_t[shallow]
    p50 = float(np.percentile(rel_shortfall, 50))
    p99 = float(np.percentile(rel_shortfall, 99))
    worst = float(rel_shortfall.max())
    n_viol = int((rel_shortfall > 0).sum())
    status = "❗" if p99 > threshold_rel else "✔️ "
    print(
        f"{status} {side} intrinsic floor: rel shortfall p50/p99/max = "
        f"{p50:.2%}/{p99:.2%}/{worst:.2%} over {n_viol:,} violating bars "
        f"(excl {n_excl:,} deep-ITM; p99 threshold {threshold_rel:.2%})"
    )
    return p99 <= threshold_rel


def save_options_if_valid(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    underlying: str,
    underlying_df: pd.DataFrame,
    save_dir: str,
    format: PriceFileFormat,
    config: OptionsChecksConfig,
) -> bool:
    """Run the structural gate; on pass, save both sides and verify the round-trip."""
    if not check_options(calls, puts, underlying, underlying_df, config):
        print(f"\n❌ {underlying} options checks failed, not saving!")
        return False
    print(f"\n🎉 {underlying} options checks passed, saving...")
    save_options(calls, puts, underlying, save_dir=save_dir, format=format)
    return verify_saved_options(calls, puts, underlying, save_dir=save_dir, format=format)
