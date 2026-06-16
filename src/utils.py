import os
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Tuple, TypeVar, cast

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

from .constants import (
    DATA_SOURCE_URL,
    OPTIONS_INTERNALS,
    READ_MAX_WORKERS,
    YF_MAX_RETRIES,
    YF_RETRY_BACKOFF,
)
from .schemas import (
    ASSET_TYPE_CONFIG,
    ASSET_TYPES,
    AssetType,
    DateLike,
    OSIContract,
    PriceFileFormat,
)

T = TypeVar("T")


def check_data_dir(data_dir: str) -> Tuple[Path, list[str]]:
    """Check if data directory exists and contains expected asset type subfolders.
    Returns data_dir path and asset_types list."""
    data_dir_path = Path(data_dir)
    download_hint = (
        f"Download the raw daily price files into `{data_dir}/<asset_type>/` -- "
        f"files for all asset types (through 2026) are available at:\n  {DATA_SOURCE_URL}"
    )

    if not data_dir_path.exists() or not data_dir_path.is_dir():
        raise ValueError(f"Data directory does not exist: `{data_dir}`.\n{download_hint}")

    asset_types = sorted(
        list(
            item.name
            for item in data_dir_path.iterdir()
            if item.is_dir() and any(item.name == at for at in ASSET_TYPES)
        )
    )

    if not asset_types:
        raise ValueError(
            f"Data directory `{data_dir}` has no expected asset type subfolders.\n"
            f"Expected at least one of: {ASSET_TYPES} but found: "
            f"{[item.name for item in data_dir_path.iterdir() if item.is_dir()]}\n"
            f"{download_hint}"
        )

    return data_dir_path, asset_types


def parsable_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        pd.to_datetime(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unparsable date: {value!r}") from e
    return value


def parse_date(date_input: DateLike = None, default: DateLike = None) -> date:
    """Parse any date/datetime string, timestamp, or date object and return a date object."""
    date_input = date_input or default
    if date_input is None:
        return datetime.now().date()
    try:
        return cast(pd.Timestamp, pd.to_datetime(date_input)).date()
    except (ValueError, TypeError):
        if isinstance(date_input, str):
            return datetime.strptime(date_input, "%Y-%m-%d").date()
        raise


def resolve_index_bound(
    date_str: str | None, fallback: pd.Timestamp, asset_type: AssetType
) -> pd.Timestamp:
    """Resolve a build_target_index bound from optional user input + a fallback timestamp.
    If `date_str` is given, return a naive midnight timestamp on that date (its `.date()`
    is what build_target_index will use). Otherwise return `fallback`, tz-converted to
    America/New_York for NYSE assets so the trading day -- not the UTC day -- drives the
    round-trip in `_index_matches_calendar`.
    """
    if date_str:
        return pd.Timestamp(parse_date(date_str))
    if asset_type in (AssetType.STOCKS, AssetType.OPTIONS):
        return cast(pd.Timestamp, fallback.tz_convert("America/New_York"))
    return fallback


def build_target_index(
    start: pd.Timestamp, end: pd.Timestamp, asset_type: AssetType
) -> pd.DatetimeIndex:
    """Return the expected 1-min timestamp index for an asset type over the calendar
    days [start.date(), end.date()]. Bounds are inclusive at the session level:
    stocks/options return every session bar on those dates (no mid-session clipping),
    crypto/forex return a continuous grid from 00:00 UTC on start.date() through
    23:59 UTC on end.date(). Callers that pass NYSE-asset df.index timestamps for the
    round-trip case must convert to ET first so .date() reflects the trading day, not
    the UTC day (last bar of an EST session lands at 00:59 UTC the next calendar day).
    """
    if asset_type in (AssetType.CRYPTO, AssetType.FOREX):
        start_utc = pd.Timestamp(start.date(), tz="UTC")
        end_utc = pd.Timestamp(end.date(), tz="UTC") + pd.Timedelta(hours=23, minutes=59)
        return pd.date_range(start=start_utc, end=end_utc, freq="1min")

    nyse = mcal.get_calendar("NYSE")
    if asset_type == AssetType.STOCKS:
        sched = nyse.schedule(
            start_date=start.date(),
            end_date=end.date(),
            market_times=["pre", "market_open", "market_close", "post"],
        )
        sessions: set[Literal["pre", "RTH", "post"]] = {"pre", "RTH", "post"}
    elif asset_type == AssetType.OPTIONS:
        sched = nyse.schedule(start_date=start.date(), end_date=end.date())
        sessions = {"RTH"}
    else:
        raise ValueError(f"Unsupported asset type: {asset_type}")

    if sched.empty:
        raise ValueError(
            f"No NYSE sessions for {asset_type} between {start.date()} and {end.date()}"
        )
    return cast(
        pd.DatetimeIndex,
        mcal.date_range(
            sched, frequency="1min", closed="left", force_close=False, session=sessions
        ),
    )


def get_files_in_range(data_dir: Path, asset_type: str, start: date, end: date) -> list[Path]:
    """Get all files in date range for an asset type."""
    asset_dir = data_dir.joinpath(asset_type)
    files_in_range = []
    for f in asset_dir.glob("*.csv.gz"):
        try:
            date_val = datetime.strptime(f.stem.replace(".csv", ""), "%Y-%m-%d").date()
            if start <= date_val <= end:
                files_in_range.append(f)
        except ValueError:
            continue
    return sorted(
        files_in_range,
        key=lambda f: datetime.strptime(f.stem.replace(".csv", ""), "%Y-%m-%d").date(),
    )


def normalize_ticker(ticker: str, asset_type: AssetType) -> str:
    """Normalize ticker with appropriate prefix based on asset type."""
    prefix = ASSET_TYPE_CONFIG[asset_type]["prefix"]
    return f"{prefix}{ticker}" if prefix and not ticker.startswith(prefix) else ticker


def parse_osi_ticker(ticker: str) -> OSIContract:
    """Parse an OSI option ticker into (underlying, expiry, option_type, strike).
    Accepts either the Polygon-prefixed form (`O:SPY240315C00400000`) or unprefixed
    (`SPY240315C00400000`). Layout: variable-length root, then YYMMDD (always 20YY —
    OSI is post-2010), then `C`/`P`, then 8-digit strike scaled by 1000. Parses from
    the right so the variable-length root is unambiguous (1 to 6+ chars in practice).
    """
    body = ticker[2:] if ticker.startswith("O:") else ticker
    if len(body) < 16:
        raise ValueError(f"OSI ticker too short to parse: `{ticker}`")
    else:
        underlying = body[:-15]
        yy, mm, dd = body[-15:-13], body[-13:-11], body[-11:-9]
        option_char, strike_str = body[-9], body[-8:]

    if not (yy.isdigit() and mm.isdigit() and dd.isdigit() and strike_str.isdigit()):
        raise ValueError(f"Non-numeric expiry or strike in OSI ticker: `{ticker}`")
    if option_char not in ("C", "P"):
        raise ValueError(f"Expected `C` or `P`, got `{option_char}` in OSI ticker: `{ticker}`")

    try:
        expiry = date(2000 + int(yy), int(mm), int(dd))
    except ValueError as e:
        raise ValueError(f"Invalid expiry date in OSI ticker: `{ticker}`") from e

    return OSIContract(
        underlying=underlying,
        expiry=expiry,
        option_type=cast(Literal["C", "P"], option_char),
        strike=int(strike_str) / OPTIONS_INTERNALS["strike_scale"],
    )


def underlying_matches(parsed_underlying: str, target: str) -> bool:
    """Match a parsed OSI underlying to `target`, allowing OCC numeric-suffix roots
    (e.g. target=`AAPL` matches `AAPL`, `AAPL1`, `AAPL7`, …). Used by the loader to
    bundle OCC-adjusted suffixed-root contracts under the user-facing underlying.
    """
    if parsed_underlying == target:
        return True
    if parsed_underlying.startswith(target):
        suffix = parsed_underlying[len(target) :]
        return bool(suffix) and suffix.isdigit()
    return False


def format_osi_ticker(contract: OSIContract) -> str:
    """Inverse of `parse_osi_ticker` — reconstruct the Polygon-prefixed OSI symbol.
    Strike is rounded to the nearest 1/1000 dollar (OSI's native precision); callers
    that need to verify a strike divides cleanly under a corporate-action ratio should
    check that themselves before calling.
    """
    yy = contract.expiry.year % 100
    strike_milli = round(contract.strike * OPTIONS_INTERNALS["strike_scale"])
    return (
        f"O:{contract.underlying}"
        f"{yy:02d}{contract.expiry.month:02d}{contract.expiry.day:02d}"
        f"{contract.option_type}{strike_milli:08d}"
    )


def read_gz(path: Path, cols: list[str] | None = None) -> pd.DataFrame:
    """Read a gzipped daily CSV (pyarrow when available, else the default parser). `cols` limits
    the columns read; None reads all."""
    try:
        return pd.read_csv(
            path, compression="gzip", engine="pyarrow", dtype_backend="pyarrow", usecols=cols
        )
    except Exception:
        return pd.read_csv(path, compression="gzip", usecols=cols)


def _read_files_concat(
    files: list[Path], extract: Callable[[pd.DataFrame], pd.DataFrame]
) -> list[pd.DataFrame]:
    """Read each gzipped daily file and apply `extract` (a per-file row filter), in parallel,
    returning the non-empty results in file order. pyarrow's reader releases the GIL during the
    read (the bulk of load time), so a thread pool parallelizes it; `map` preserves file order.
    """
    with ThreadPoolExecutor(max_workers=READ_MAX_WORKERS) as ex:
        results = ex.map(lambda p: extract(read_gz(p)), files)
    return [df for df in results if not df.empty]


def determine_asset_type(
    data_dir: Path,
    asset_types: list[str],
    ticker: str,
    date_start: str | None = None,
    date_end: str | None = None,
) -> Tuple[AssetType, list[Path]]:
    """Determine asset type by checking folders for ticker existence.
    Returns asset_type and files in date range.
    """
    start = parse_date(date_start, "2000-01-01")
    end = parse_date(date_end, datetime.now().date().strftime("%Y-%m-%d"))

    for asset_type in asset_types:
        files_in_range = get_files_in_range(data_dir, asset_type, start, end)
        if not files_in_range:
            continue

        try:
            last_file = files_in_range[-1]
            asset_type_enum = AssetType(asset_type)
            normalized_ticker = normalize_ticker(ticker, asset_type_enum)
            df = read_gz(last_file)
            if "ticker" in df.columns and (df["ticker"] == normalized_ticker).any():
                return asset_type_enum, files_in_range
        except Exception:
            continue

    raise ValueError(f"Could not determine asset type for ticker: `{ticker}`")


def load_options_data(
    data_dir: Path,
    underlying: str,
    date_start: str | None = None,
    date_end: str | None = None,
) -> pd.DataFrame:
    """Load raw option bars for every contract on `underlying` over [date_start, date_end].
    Per-file pipeline: tight prefix prefilter (`^O:<underlying>\\d` — narrow enough to
    avoid the 1-char-root pathology where bare `O:A` would catch every A-prefixed
    underlying) then exact confirmation via `parse_osi_ticker(t).underlying == underlying`
    on the unique survivors (rules out e.g. `O:SPYG…` when underlying is `SPY`). Returns
    flat raw rows (raw schema preserved); reshaping into multi-indexed calls/puts frames
    is the caller's job.
    """
    start = parse_date(date_start, "2000-01-01")
    end = parse_date(date_end, datetime.now().date().strftime("%Y-%m-%d"))
    files_in_range = get_files_in_range(data_dir, AssetType.OPTIONS, start, end)
    if not files_in_range:
        raise ValueError(f"No options files in range {start}..{end}")

    prefilter = rf"O:{re.escape(underlying)}\d"

    def extract(df: pd.DataFrame) -> pd.DataFrame:
        candidates = df[df["ticker"].str.match(prefilter, na=False)]
        if candidates.empty:
            return candidates
        matching = {
            t
            for t in candidates["ticker"].unique()
            if underlying_matches(parse_osi_ticker(t).underlying, underlying)
        }
        return candidates[candidates["ticker"].isin(matching)]

    dfs = _read_files_concat(files_in_range, extract)
    if not dfs:
        raise ValueError(
            f"No option contracts found for underlying `{underlying}` in {start}..{end}"
        )
    return pd.concat(dfs, ignore_index=True)


def load_symbol_rows(files: list[Path], symbol: str) -> pd.DataFrame:
    """Concatenate the raw rows for one exact (already-normalized) ticker symbol across `files`.
    Returns an empty frame if the symbol appears in none of them — callers decide if that's an
    error. Shared by `load_ticker_data` and the alias stitcher (which loads predecessor symbols).
    """
    dfs = _read_files_concat(files, lambda df: df[df["ticker"] == symbol])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_ticker_data(
    data_dir: Path,
    asset_types: list[str],
    ticker: str,
    date_start: str | None = None,
    date_end: str | None = None,
) -> Tuple[pd.DataFrame, AssetType]:
    """Load and concatenate ticker data from files in range.
    Determines asset type and loads data.
    """
    asset_type, files_in_range = determine_asset_type(
        data_dir, asset_types, ticker, date_start, date_end
    )
    df = load_symbol_rows(files_in_range, normalize_ticker(ticker, asset_type))
    if df.empty:
        raise ValueError(f"No data found for `{asset_type}` ticker `{ticker}`")

    return df, asset_type


def yf_retry(
    fetch: Callable[[], T], label: str, retry_empty: Callable[[T], bool] = lambda _: False
) -> T:
    """Call `fetch`, retrying transient yfinance failures (an exception, or — when `retry_empty`
    says so — an empty result) with linear backoff. Returns the final result even if still empty,
    so a genuinely empty series is a valid answer."""
    for attempt in range(1, YF_MAX_RETRIES + 1):
        last = attempt == YF_MAX_RETRIES
        try:
            result = fetch()
            if last or not retry_empty(result):
                return result
            reason = "empty result"
        except Exception as e:
            if last:
                raise
            reason = type(e).__name__
        print(f"🔁 {label}: {reason} -- retrying ({attempt}/{YF_MAX_RETRIES - 1})")
        time.sleep(YF_RETRY_BACKOFF * attempt)
    raise RuntimeError("unreachable")  # loop always returns or raises on the final attempt


def fetch_splits(ticker: str, start: pd.Timestamp) -> pd.Series:
    """Fetch yfinance splits for `ticker` with ex-date >= `start`.
    No upper bound: events with ex-date *after* the data window still apply (the
    `index < ex_date` mask in adjust_for_splits covers all rows), matching yfinance's
    convention that historical prices reflect all known future events.
    """
    splits = yf_retry(lambda: yf.Ticker(ticker).splits, f"{ticker} splits")
    if splits.empty:
        return splits
    idx = pd.to_datetime(splits.index)
    splits.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    return splits.loc[start:]


def fetch_dividends(ticker: str, start: pd.Timestamp) -> pd.Series:
    """Fetch yfinance cash dividends for `ticker` with ex-date >= `start`.
    See `fetch_splits` for why there's no upper bound.
    """
    divs = yf_retry(lambda: yf.Ticker(ticker).dividends, f"{ticker} dividends")
    if divs.empty:
        return divs
    idx = pd.to_datetime(divs.index)
    divs.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    return divs.loc[start:]


def fetch_yf_closes(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch yfinance daily Close (split-adjusted, no dividend adjustment) for `ticker`
    over [start, end]. Used as the `prev_close` reference in dividend factor computation
    so adjusted prices match yfinance's Adj Close exactly — yfinance computes the factor
    against its official 16:00 ET close, not the last bar before midnight ET.
    """
    h = yf_retry(
        lambda: yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False),
        f"{ticker} closes",
        retry_empty=lambda df: df.empty,
    )
    if h.empty:
        return pd.Series(dtype=float)
    closes = h["Close"].copy()
    idx = pd.to_datetime(closes.index)
    closes.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    return closes


def describe_adjusted_prices(df: pd.DataFrame, ticker: str) -> None:
    """Print a one-line summary of a single-series price frame (bar count, index range,
    price range). `ticker` names the price column."""
    print(
        f"🔎 {ticker}: {len(df):,} values from {df.index[0]} to {df.index[-1]},"
        f" price ranges from ${df[ticker].min():,.2f} to ${df[ticker].max():,.2f}"
    )


def describe_adjusted_options(calls: pd.DataFrame, puts: pd.DataFrame, underlying: str) -> None:
    """Print a per-side (bars / contracts / time range) summary of `underlying`'s options."""
    print(f"🗃️  Option contracts for {underlying}:")
    for label, df in (("calls", calls), ("puts", puts)):
        if df.empty:
            print(f"   - {label}: 0 bars / 0 contracts")
            continue
        ts = df.index.get_level_values("timestamp_utc")
        n_contracts = df.index.get_level_values("ticker").nunique()
        print(
            f"   - {label}: {len(df):,} values / {n_contracts:,} contracts"
            f" from {ts.min()} to {ts.max()}"
        )


def _date_range_suffix(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """`YYYYMMDD_YYYYMMDD` spanning [start, end] as UTC dates, for self-describing filenames."""
    return f"{parse_date(start):%Y%m%d}_{parse_date(end):%Y%m%d}"


def _price_file_name(df: pd.DataFrame, format: PriceFileFormat, dividends: bool = False) -> str:
    """`<TICKER>[_dividends]_<start>_<end>.<format>`; the date range is taken from the frame's own
    index, and the `dividends` tag flags a dividend-adjusted (total-return) series so it isn't
    confused with the default split-only one."""
    idx = cast(pd.DatetimeIndex, df.index)
    suffix = _date_range_suffix(cast(pd.Timestamp, idx.min()), cast(pd.Timestamp, idx.max()))
    tag = "dividends_" if dividends else ""
    return f"{df.columns[0]}_{tag}{suffix}.{format}"


def save_prices(
    df: pd.DataFrame,
    save_dir: str = "./data/prices",
    format: PriceFileFormat = "parquet",
    dividends: bool = False,
) -> None:
    """Save the prices to `<save_dir>/<TICKER>[_dividends]_<start>_<end>.<format>` (date range
    from the frame's own index; `dividends` tags a dividend-adjusted series)."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, _price_file_name(df, format, dividends))
    if format == "csv":
        df.to_csv(save_path, index=True)
    elif format == "parquet":
        df.to_parquet(save_path)
    else:
        raise ValueError(f"Invalid format: {format}, must be `csv` or `parquet`")
    print(f"📀 Saved {df.columns[0]} prices to {save_path}")


def load_prices(file_name: str, load_dir: str = "./data/prices") -> pd.DataFrame:
    """Load prices from a CSV or Parquet file. Format is detected from file extension.

    Timestamps are assumed to be UTC. For CSV files (which don't preserve timezone info),
    the index is explicitly localized to UTC after loading.
    """
    file_path = os.path.join(load_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Price file not found: `{file_path}`")

    try:
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext == ".csv":
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            # CSV doesn't preserve timezone info, so localize to UTC
            if df.index.tz is None:  # type: ignore[attr-defined]
                df.index = df.index.tz_localize("UTC")  # type: ignore[attr-defined]
        elif file_ext == ".parquet":
            df = pd.read_parquet(file_path)
            # Parquet preserves timezone, but ensure it's UTC
            if df.index.tz is None:  # type: ignore[attr-defined]
                df.index = df.index.tz_localize("UTC")  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unsupported file format: `{file_ext}`, must be `csv` or `parquet`")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load prices from `{file_path}`: {e}") from e


def verify_saved_prices(
    df: pd.DataFrame, save_dir: str, format: PriceFileFormat, dividends: bool = False
) -> bool:
    """Reload the saved file and confirm it matches the in-memory data."""
    ticker = df.columns[0]
    loaded = load_prices(_price_file_name(df, format, dividends), load_dir=save_dir)
    values_match = np.allclose(df.values, loaded.values, equal_nan=True)
    index_match = df.index.equals(loaded.index)
    if not (values_match and index_match):
        print(f"❗ Saved file for {ticker} does not match in-memory data!")
        return False
    print(f"💿 Successfully loaded {ticker} data through {format}")
    return True


def _options_date_range_suffix(calls: pd.DataFrame, puts: pd.DataFrame) -> str:
    """One `YYYYMMDD_YYYYMMDD` shared across both sides (so a run's calls/puts files carry the
    same range), spanning the min/max bar of the non-empty sides. "" only if both are empty."""
    levels = [
        cast(pd.DatetimeIndex, df.index.get_level_values("timestamp_utc"))
        for df in (calls, puts)
        if not df.empty
    ]
    if not levels:
        return ""
    start = min(cast(pd.Timestamp, lv.min()) for lv in levels)
    end = max(cast(pd.Timestamp, lv.max()) for lv in levels)
    return _date_range_suffix(start, end)


def _options_file_name(underlying: str, suffix: str, side: str, format: PriceFileFormat) -> str:
    return f"{underlying}_{suffix}_{side}.{format}"


def save_options(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    underlying: str,
    save_dir: str = "./data/prices",
    format: PriceFileFormat = "parquet",
) -> None:
    """Save calls and puts to
    `<save_dir>/options/<UNDERLYING>_<start>_<end>_{calls,puts}.<format>` (one shared date
    range across both sides, from the frames' indices). Each frame is multi-indexed on
    `(timestamp_utc, ticker)` with a `close` column; empty frames are skipped (one side may
    legitimately have no contracts in range).
    """
    out_dir = os.path.join(save_dir, "options")
    os.makedirs(out_dir, exist_ok=True)
    suffix = _options_date_range_suffix(calls, puts)
    for side, df in (("calls", calls), ("puts", puts)):
        if df.empty:
            print(f"⚠️ {underlying} {side}: no contracts to save")
            continue
        save_path = os.path.join(out_dir, _options_file_name(underlying, suffix, side, format))
        if format == "csv":
            df.to_csv(save_path, index=True)
        elif format == "parquet":
            df.to_parquet(save_path)
        else:
            raise ValueError(f"Invalid format: {format}, must be `csv` or `parquet`")
        print(f"📀 Saved {underlying} {side} to {save_path}")


def load_options_file(file_name: str, load_dir: str = "./data/prices/options") -> pd.DataFrame:
    """Load a saved options file (multi-indexed on `(timestamp_utc, ticker)`).
    Parquet preserves the multi-index and tz natively; CSV is re-parsed and the
    `timestamp_utc` level is re-localized to UTC if it came back naive.
    """
    file_path = os.path.join(load_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Options file not found: `{file_path}`")
    file_ext = os.path.splitext(file_name)[1].lower()
    if file_ext == ".csv":
        df = pd.read_csv(file_path, index_col=[0, 1], parse_dates=[0])
    elif file_ext == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: `{file_ext}`, must be `csv` or `parquet`")

    ts_level = cast(pd.DatetimeIndex, df.index.get_level_values("timestamp_utc"))
    if ts_level.tz is None:
        df.index = df.index.set_levels(  # type: ignore[attr-defined]
            ts_level.tz_localize("UTC"), level="timestamp_utc"
        )
    return df


def verify_saved_options(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    underlying: str,
    save_dir: str,
    format: PriceFileFormat,
) -> bool:
    """Reload each saved side and confirm values + index match the in-memory frames."""
    options_dir = os.path.join(save_dir, "options")
    suffix = _options_date_range_suffix(calls, puts)
    ok = True
    for side, df in (("calls", calls), ("puts", puts)):
        if df.empty:
            continue
        loaded = load_options_file(
            _options_file_name(underlying, suffix, side, format), load_dir=options_dir
        )
        values_match = np.allclose(df.values, loaded.values, equal_nan=True)
        index_match = df.index.equals(loaded.index)
        if not (values_match and index_match):
            print(f"❗ Saved options file for {underlying} {side} does not match in-memory data!")
            ok = False
            continue
        print(f"💿 Successfully loaded {underlying} {side} through {format}")
    return ok


def _prompt_save_on_fail() -> bool:
    return input("\n❓ Checks failed. Save the data anyway? [y/N] ").strip().lower() in ("y", "yes")


def save_on_fail_decider(force: bool) -> Callable[[], bool] | None:
    # Policy for failed checks: --force saves unconditionally; an interactive terminal prompts;
    # a non-interactive run keeps the safe default (None → don't save).
    if force:
        return lambda: True
    if sys.stdin.isatty():
        return _prompt_save_on_fail
    return None
