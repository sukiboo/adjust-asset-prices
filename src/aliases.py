import itertools
from datetime import date
from pathlib import Path
from statistics import median

import pandas as pd
import pandas_market_calendars as mcal

from .constants import ALIAS_INTERNALS
from .schemas import AssetType, Predecessor
from .utils import (
    get_files_in_range,
    load_symbol_rows,
    normalize_ticker,
    parse_date,
    read_gz,
)

NY_TZ = "America/New_York"


def _to_et(window_start: pd.Series) -> pd.Series:
    """Raw `window_start` (ns epoch) -> America/New_York tz-aware datetimes."""
    return pd.to_datetime(window_start, unit="ns", utc=True).dt.tz_convert(NY_TZ)


def _et_dates(df: pd.DataFrame) -> list[date]:
    """Sorted distinct NYSE trading dates present in raw `df`."""
    return sorted(set(_to_et(df["window_start"]).dt.date))


def _files_by_date(files: list[Path]) -> dict[date, Path]:
    out: dict[date, Path] = {}
    for f in files:
        try:
            out[parse_date(f.stem.replace(".csv", ""))] = f
        except (ValueError, TypeError):
            continue
    return out


def _daily_closes(path: Path) -> pd.Series:
    """Per-ticker regular-session close on this file's date (last bar before 16:00 ET, matching
    yfinance's Close convention). Indexed by (string) ticker."""
    df = read_gz(path, ["ticker", "close", "window_start"])
    df = df[_to_et(df["window_start"]).dt.hour < 16].sort_values("window_start")
    closes = df.groupby("ticker")["close"].last().astype("float64")
    closes.index = closes.index.map(str)
    return closes


def _tickers_on(path: Path) -> set[str]:
    """All ticker symbols that traded on this file's date (any session)."""
    return set(read_gz(path, ["ticker"])["ticker"].map(str))


def _bar_counts(path: Path) -> pd.Series:
    """Rows per ticker on this file's date — an intraday-liquidity proxy. Indexed by ticker."""
    return read_gz(path, ["ticker"])["ticker"].map(str).value_counts()


def _close_on_date(df: pd.DataFrame, d: date) -> float | None:
    """Regular-session close on ET date `d` for raw `df` — where the requested series resumes."""
    et = _to_et(df["window_start"])
    sub = df[(et.dt.date == d) & (et.dt.hour < 16)]
    if sub.empty:
        return None
    return float(sub.sort_values("window_start")["close"].iloc[-1])


def _find_large_gaps(
    have: list[date], min_gap: int, start_bound: date | None = None
) -> list[tuple[date, date]]:
    """Runs of NYSE sessions (longer than `min_gap`) for which we hold no raw bars — the signature of
    a former-symbol span. A `start_bound` predating `have[0]` also surfaces the leading pre-start
    gap (e.g. META, whose pre-rename history is under FB)."""
    if not have:
        return []
    sched_start = min(have[0], start_bound) if start_bound else have[0]
    if sched_start >= have[-1]:
        return []
    sched = mcal.get_calendar("NYSE").schedule(start_date=sched_start, end_date=have[-1])
    haveset = set(have)
    sessions = [ts.date() for ts in sched.index]
    gaps: list[tuple[date, date]] = []
    for present, group in itertools.groupby(sessions, key=lambda d: d in haveset):
        run = list(group)
        if not present and len(run) > min_gap:
            gaps.append((run[0], run[-1]))
    return gaps


def _survivors(
    files_by_date: dict[date, Path], resume_date: date, slop: int, window: int
) -> set[str]:
    """Tickers trading in the `window` sessions after the rename (skipping `slop` boundary-overlap
    sessions). Scanned contiguously, so a symbol absent from the whole window genuinely stopped at
    the boundary and may be the predecessor."""
    post = [d for d in sorted(files_by_date) if d >= resume_date][slop : slop + window]
    return set().union(*(_tickers_on(files_by_date[d]) for d in post)) if post else set()


def _find_predecessor(
    combined: pd.DataFrame,
    gap_start: date,
    gap_end: date,
    have: list[date],
    files_by_date: dict[date, Path],
    visited: set[str],
) -> str | None:
    """Identify the predecessor symbol that traded over `[gap_start, gap_end]`, from our files only
    (yfinance is reserved for the independent compare-to-yf gate): a stops-at-the-boundary filter,
    then a liquidity filter, then nearest resume-price (argmin) with a sanity floor. See the
    aliases.py bullet in CLAUDE.md for what each layer catches."""
    gap_dates = [d for d in files_by_date if gap_start <= d <= gap_end]
    resume_dates = [d for d in have if d > gap_end]
    if not gap_dates or not resume_dates:
        return None
    pre_date = max(gap_dates)  # last gap session we hold a file for (requested series absent)
    resume_date = resume_dates[0]
    target = _close_on_date(combined, resume_date)  # requested series' close where it resumes
    if target is None or target <= 0:
        return None

    slop = int(ALIAS_INTERNALS["boundary_slop_sessions"])
    window = int(ALIAS_INTERNALS["survivor_window_sessions"])
    survivors = _survivors(files_by_date, resume_date, slop, window)
    pre_closes = _daily_closes(files_by_date[pre_date])
    candidates = {t for t in pre_closes.index if t not in survivors and t not in visited}
    if not candidates:
        return None

    # Liquidity: the predecessor is the same instrument as the requested series, so near the rename
    # it traded with comparable activity. Compare median bars/day over the last `liq_window` gap
    # sessions (candidate side) against the first `liq_window` sessions back (requested side) — a
    # median over a short window, not a single day, so half-days / spikes don't sway it.
    liq_window = int(ALIAS_INTERNALS["liquidity_window_sessions"])
    back_dates = [d for d in have if d >= resume_date][:liq_window]
    day_counts = _to_et(combined["window_start"]).dt.date.value_counts()
    ref_bars = median(int(day_counts.get(d, 0)) for d in back_dates)
    min_bars = ALIAS_INTERNALS["liquidity_frac"] * ref_bars
    tail_dates = sorted(d for d in files_by_date if gap_start <= d <= gap_end)[-liq_window:]
    tail_counts = [_bar_counts(files_by_date[d]) for d in tail_dates]
    candidates = {t for t in candidates if median(c.get(t, 0) for c in tail_counts) >= min_bars}
    if not candidates:
        return None

    best = min(candidates, key=lambda t: abs(pre_closes[t] - target))
    splice = abs(pre_closes[best] - target) / target
    return best if splice <= ALIAS_INTERNALS["splice_sanity_pct"] / 100 else None


def _pregap_is_foreign(
    combined: pd.DataFrame,
    rows: pd.DataFrame,
    gap_start: date,
    have: list[date],
    sanity_frac: float,
) -> bool:
    """A genuine rename connects on BOTH gap edges — the predecessor abuts the resume side (matched
    by `_find_predecessor`) and the pre-gap base data. A reused ticker connects only on the resume
    side: the pre-gap segment is a different instrument under the same symbol (e.g. Meta Materials
    held META before Facebook). True when the pre-gap close jumps past `sanity_frac` to the
    predecessor's first gap close — i.e. that pre-gap segment is foreign and should be dropped."""
    prev_dates = [d for d in have if d < gap_start]
    pred_dates = _et_dates(rows)
    if not prev_dates or not pred_dates:
        return False
    prev_close = _close_on_date(combined, max(prev_dates))
    pred_close = _close_on_date(rows, pred_dates[0])
    if not prev_close or not pred_close:
        return False
    return abs(prev_close - pred_close) / pred_close > sanity_frac


def stitch_predecessors(
    data_dir: Path,
    asset_type: AssetType,
    ticker: str,
    base_raw: pd.DataFrame,
    date_start: str | None = None,
    date_end: str | None = None,
) -> tuple[pd.DataFrame, list[Predecessor]]:
    """Append predecessor-symbol rows to `base_raw` so a series split across ticker renames (e.g.
    QQQ⇄QQQQ) loads as one continuous instrument. Each large gap (interior or leading vs
    `date_start`) is resolved by `_find_predecessor` and spliced in; the scan repeats to unwind a
    multi-rename chain one link at a time, with a visited-set guarding symbol-reuse loops. Concat +
    window_start dedup keeps the series gap-free and unique; the ticker label is dropped downstream
    so output stays under the requested (live) ticker. Returns the merged frame and the list of
    spliced `Predecessor`s (former symbol + span) so the options pass can stitch the same renames
    onto its OSI contracts."""
    start = parse_date(date_start, "2000-01-01")
    end = parse_date(date_end)
    files_by_date = _files_by_date(get_files_in_range(Path(data_dir), asset_type.value, start, end))
    lower_bound = parse_date(date_start) if date_start else None
    min_gap = int(ALIAS_INTERNALS["min_gap_trading_days"])
    sanity = ALIAS_INTERNALS["splice_sanity_pct"] / 100

    visited = {normalize_ticker(ticker, asset_type)}
    combined = base_raw
    predecessors: list[Predecessor] = []
    while True:
        have = _et_dates(combined)
        gaps = _find_large_gaps(have, min_gap, lower_bound)
        if not gaps:
            break
        for gap_start, gap_end in gaps:
            pred = _find_predecessor(combined, gap_start, gap_end, have, files_by_date, visited)
            if pred is None:
                continue
            gap_files = [p for d, p in files_by_date.items() if gap_start <= d <= gap_end]
            rows = load_symbol_rows(gap_files, pred)
            if rows.empty:
                continue
            if _pregap_is_foreign(combined, rows, gap_start, have, sanity):
                kept = combined[_to_et(combined["window_start"]).dt.date > gap_end]
                print(
                    f"🧹 {ticker}: dropped {len(combined) - len(kept):,} rows before {gap_start} "
                    f"(reused ticker -- another company held {ticker} there, unrelated to {pred})"
                )
                combined = kept
                break  # re-derive gaps on the cleaned series
            pred_dates = _et_dates(rows)
            print(
                f"🔗  {ticker}: stitched predecessor {pred} over {gap_start} -- {gap_end} "
                f"({len(rows):,} rows)"
            )
            combined = pd.concat([combined, rows], ignore_index=True).drop_duplicates(
                subset="window_start", keep="first"
            )
            predecessors.append(Predecessor(pred, pred_dates[0], pred_dates[-1]))
            visited.add(pred)
            break  # recompute gaps from scratch — the stitch may expose an earlier link
        else:
            for gs, ge in gaps:
                print(f"⛓️‍💥  {ticker}: gap {gs} -- {ge} has no recoverable predecessor")
            break
    return combined, predecessors
