from collections.abc import Callable, Iterator
from typing import cast

import numpy as np
import pandas as pd

from ..schemas import OptionsChecksConfig, PriceFileFormat
from ..utils import parse_osi_ticker, stream_save_options


def check_options(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    underlying: str,
    underlying_df: pd.DataFrame,
    config: OptionsChecksConfig,
) -> bool:
    """Structural gate: four no-arb self-consistency checks against the underlying
    (yfinance has no historical per-contract series to compare to):
      1. positivity (P > 0, C > 0) — strict, every bar
      2. no bars past expiry — strict, every bar
      3. upper bounds (P ≤ K, C ≤ S) — relative-p99-gated, REAL & non-deep-ITM bars only
      4. intrinsic floors (P ≥ K - S, C ≥ S - K) — same gating
    The two no-arb bounds score REAL (traded) bars only via the `is_real` column: synthetic
    backfilled bars hold the last trade flat and don't track the underlying between prints, so
    they would breach the bounds without any arb. `underlying_df` must be SPLIT-ONLY (not
    dividend-adjusted): dividends are priced into the premium ahead of ex-date, so a
    dividend-adjusted series overstates intrinsic.
    """
    print(f"\n🔍 Checking {underlying} options...")
    if calls.empty and puts.empty:
        print("⚠️ Both calls and puts empty; nothing to check")
        return False

    threshold_pct = config["noarb_violation_pct_p99"]
    deep_itm_pct = config["deep_itm_intrinsic_pct"]
    if underlying not in underlying_df.columns:
        raise KeyError(
            f"underlying_df must have a column named `{underlying}`; got {list(underlying_df.columns)}"
        )
    underlying_series = underlying_df[underlying]

    results: list[bool] = []
    for side, df in [("calls", calls), ("puts", puts)]:
        if df.empty:
            print(f"⚠️ {side} side empty; skipping {side} checks")
            continue
        results.extend(_run_side_checks(side, df, underlying_series, threshold_pct, deep_itm_pct))
    return all(results) if results else False


def _run_side_checks(
    side: str,
    df: pd.DataFrame,
    underlying_series: pd.Series,
    threshold_pct: float,
    deep_itm_pct: float,
) -> list[bool]:
    """Run the four checks on one side (calls or puts). Strikes, the underlying snapshot,
    intrinsic value, and the evaluation mask are computed once and shared across checks."""
    ts_level = cast(pd.DatetimeIndex, df.index.get_level_values("timestamp_utc"))
    ticker_level = df.index.get_level_values("ticker")
    parsed_by_ticker = {t: parse_osi_ticker(cast(str, t)) for t in ticker_level.unique()}
    strikes = np.array([parsed_by_ticker[t].strike for t in ticker_level])
    underlying_at_t = cast(pd.Series, underlying_series.reindex(ts_level)).to_numpy()
    close = df["close"].to_numpy()
    # Raw adjusted frames (streaming save path) have no `is_real` — every bar is real, so absence
    # means all-real (equivalent to gating the backfilled output; synthetic bars are ffill copies).
    real = df["is_real"].to_numpy() if "is_real" in df.columns else np.ones(len(df), dtype=bool)

    # The two no-arb bound checks evaluate REAL (traded) bars that are not deep-ITM:
    #  - real only: synthetic backfilled bars hold the last trade flat and don't track the
    #    underlying between prints, so they breach the bounds by construction without any arb.
    #    The bounds are the *economic* check; positivity + expiry still cover every bar.
    #  - not deep-ITM (intrinsic > deep_itm_pct% of underlying): even real deep-ITM prints lag
    #    spot on illiquid contracts; a real split error is strike-independent and still hits
    #    the retained near-the-money bars.
    if side == "calls":
        intrinsic = np.maximum(underlying_at_t - strikes, 0.0)
    else:
        intrinsic = np.maximum(strikes - underlying_at_t, 0.0)
    shallow = intrinsic <= (deep_itm_pct / 100) * underlying_at_t
    eval_mask = real & shallow
    n_deep_itm = int((real & ~shallow).sum())  # real bars dropped for being deep-ITM (reported)

    return [
        _check_positive(side, close, ts_level, ticker_level),
        _check_no_bars_past_expiry(side, ts_level, ticker_level, parsed_by_ticker),
        _check_upper_bound(
            side, close, strikes, underlying_at_t, eval_mask, n_deep_itm, threshold_pct
        ),
        _check_intrinsic_floor(
            side, close, intrinsic, underlying_at_t, eval_mask, n_deep_itm, threshold_pct
        ),
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
    return True


def _check_upper_bound(
    side: str,
    close: np.ndarray,
    strikes: np.ndarray,
    underlying_at_t: np.ndarray,
    eval_mask: np.ndarray,
    n_deep_itm: int,
    threshold_pct: float,
) -> bool:
    """Calls: close ≤ underlying. Puts: close ≤ strike. A last print lagging spot can
    breach by a few % without an arb; systematic mis-scaling pushes p99 past the band.
    """
    label = "underlying" if side == "calls" else "strike"
    upper = underlying_at_t if side == "calls" else strikes
    breach_pct = 100 * np.maximum(close - upper, 0.0) / underlying_at_t
    return _gate_relative_violation(
        side, f"≤ {label}", breach_pct, eval_mask, n_deep_itm, threshold_pct, quiet=True
    )


def _check_intrinsic_floor(
    side: str,
    close: np.ndarray,
    intrinsic: np.ndarray,
    underlying_at_t: np.ndarray,
    eval_mask: np.ndarray,
    n_deep_itm: int,
    threshold_pct: float,
) -> bool:
    """Calls: close ≥ max(S - K, 0). Puts: close ≥ max(K - S, 0). On real near-the-money bars
    this holds tightly; a missed split shows up as a large shortfall that hits all moneyness
    (split adjustment is strike-independent), so it survives the deep-ITM exclusion.
    """
    shortfall_pct = 100 * np.maximum(intrinsic - close, 0.0) / underlying_at_t
    return _gate_relative_violation(
        side, "intrinsic floor", shortfall_pct, eval_mask, n_deep_itm, threshold_pct
    )


def _gate_relative_violation(
    side: str,
    desc: str,
    violation_pct: np.ndarray,
    eval_mask: np.ndarray,
    n_deep_itm: int,
    threshold_pct: float,
    quiet: bool = False,
) -> bool:
    """Percentile-gate a no-arb violation over the evaluation bars (`eval_mask` =
    real & not-deep-ITM). `violation_pct` is the per-bar breach/shortfall as a percent of the
    underlying; the gate fires when its p99 over those bars exceeds `threshold_pct` (also a
    percent). `n_deep_itm` is the count of real bars excluded for being deep-ITM (reported;
    synthetic bars are excluded too but not counted here). `quiet` suppresses the passing
    summary line (failures always print).
    """
    if not eval_mask.any():
        if not quiet:
            print(
                f"✔️  {side} {desc}: no real near-the-money bars to check "
                f"({n_deep_itm:,} real deep-ITM excluded)"
            )
        return True
    rel = violation_pct[eval_mask]
    p50, p90, p99 = (float(np.percentile(rel, q)) for q in (50, 90, 99))
    failed = p99 > threshold_pct
    if failed or not quiet:
        status = "❗" if failed else "✔️ "
        print(
            f"{status} {side} {desc} violation (p50/p90/p99): "
            f"{p50:.3f}% / {p90:.3f}% / {p99:.3f}% "
            f"(excl {n_deep_itm:,} real deep-ITM; threshold {threshold_pct:.2f}%)"
        )
    return p99 <= threshold_pct


def save_options_if_valid(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    underlying: str,
    underlying_df: pd.DataFrame,
    save_dir: str,
    format: PriceFileFormat,
    config: OptionsChecksConfig,
    backfill_side: Callable[[pd.DataFrame], Iterator[pd.DataFrame]],
) -> bool:
    """Gate the RAW adjusted contracts, then stream the backfill to disk and verify. Gating raw
    equals gating the backfilled output (synthetic bars are ffill copies). `backfill_side(df)` yields
    the backfill in row-bounded batches, so the full RAM-exceeding frame is never materialized.
    """
    if not check_options(calls, puts, underlying, underlying_df, config):
        print(f"\n❌ {underlying} options checks failed, not saving!")
        return False
    print(f"\n🎉 {underlying} options checks passed, saving the options data...")

    def get_batches(side: str) -> Iterator[pd.DataFrame]:
        return backfill_side(calls if side == "calls" else puts)

    return stream_save_options(get_batches, underlying, save_dir=save_dir, format=format)
