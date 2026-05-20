import os
from datetime import date, datetime
from pathlib import Path
from typing import Tuple, cast

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

from .schemas import (
    ASSET_TYPE_CONFIG,
    ASSET_TYPES,
    AssetType,
    DateLike,
    PriceFileFormat,
)


def check_data_dir(data_dir: str) -> Tuple[Path, list[str]]:
    """Check if data directory exists and contains expected asset type subfolders.
    Returns data_dir path and asset_types list."""
    data_dir_path = Path(data_dir)

    if not data_dir_path.exists() or not data_dir_path.is_dir():
        raise ValueError(f"Data directory does not exist: `{data_dir}`")

    asset_types = sorted(
        list(
            item.name
            for item in data_dir_path.iterdir()
            if item.is_dir() and any(item.name == at for at in ASSET_TYPES)
        )
    )

    if not asset_types:
        raise ValueError(
            f"Data directory is empty or does not contain expected asset type subfolders.\n"
            f"Expected at least one of: {ASSET_TYPES} but found: "
            f"{[item.name for item in data_dir_path.iterdir() if item.is_dir()]}"
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


def build_target_index(
    start: pd.Timestamp, end: pd.Timestamp, asset_type: AssetType
) -> pd.DatetimeIndex:
    """Return the expected 1-min timestamp index for an asset type over [start, end].
    Crypto/forex use a continuous grid; stocks use NYSE extended hours (04:00-19:59 ET);
    options use NYSE regular hours (09:30-15:59 ET). Half-days come from the calendar.
    """
    if asset_type in (AssetType.CRYPTO, AssetType.FOREX):
        return pd.date_range(start=start, end=end, freq="1min")

    nyse = mcal.get_calendar("NYSE")
    if asset_type == AssetType.STOCKS:
        sched = nyse.schedule(
            start_date=start.date(), end_date=end.date(), market_times=["pre", "post"]
        )
    elif asset_type == AssetType.OPTIONS:
        sched = nyse.schedule(start_date=start.date(), end_date=end.date())
    else:
        raise ValueError(f"Unsupported asset type: {asset_type}")

    if sched.empty:
        raise ValueError(
            f"No NYSE sessions for {asset_type} between {start.date()} and {end.date()}"
        )
    idx = mcal.date_range(sched, frequency="1min", closed="left", force_close=False)
    return cast(pd.DatetimeIndex, idx[(idx >= start) & (idx <= end)])


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
            df = pd.read_csv(last_file, compression="gzip")
            if "ticker" in df.columns and (df["ticker"] == normalized_ticker).any():
                return asset_type_enum, files_in_range
        except Exception:
            continue

    raise ValueError(f"Could not determine asset type for ticker: `{ticker}`")


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
    normalized_ticker = normalize_ticker(ticker, asset_type)
    dfs = []
    for file_path in files_in_range:
        try:
            df = pd.read_csv(
                file_path, compression="gzip", engine="pyarrow", dtype_backend="pyarrow"
            )
        except Exception:
            df = pd.read_csv(file_path, compression="gzip")

        ticker_rows = df[df["ticker"] == normalized_ticker]
        if not ticker_rows.empty:
            dfs.append(ticker_rows)

    if not dfs:
        raise ValueError(f"No data found for `{asset_type}` ticker `{ticker}`")

    return pd.concat(dfs, ignore_index=True), asset_type


def fetch_splits(ticker: str, start: pd.Timestamp) -> pd.Series:
    """Fetch yfinance splits for `ticker` with ex-date >= `start`.
    No upper bound: events with ex-date *after* the data window still apply (the
    `index < ex_date` mask in adjust_splits covers all rows), matching yfinance's
    convention that historical prices reflect all known future events.
    """
    splits = yf.Ticker(ticker).splits
    if splits.empty:
        return splits
    idx = pd.to_datetime(splits.index)
    splits.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    return splits.loc[start:]


def fetch_dividends(ticker: str, start: pd.Timestamp) -> pd.Series:
    """Fetch yfinance cash dividends for `ticker` with ex-date >= `start`.
    See `fetch_splits` for why there's no upper bound.
    """
    divs = yf.Ticker(ticker).dividends
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
    h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    if h.empty:
        return pd.Series(dtype=float)
    closes = h["Close"].copy()
    idx = pd.to_datetime(closes.index)
    closes.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    return closes


def save_prices(
    df: pd.DataFrame, save_dir: str = "./data/prices", format: PriceFileFormat = "parquet"
) -> None:
    """Save the prices to a CSV or Parquet file."""
    ticker = df.columns[0]
    save_path = f"{save_dir}/{ticker}.{format}"
    os.makedirs(save_dir, exist_ok=True)
    if format == "csv":
        df.to_csv(save_path, index=True)
    elif format == "parquet":
        df.to_parquet(save_path)
    else:
        raise ValueError(f"Invalid format: {format}, must be `csv` or `parquet`")
    print(f"📀 Saved {ticker} prices to {save_path}")


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


def verify_saved_prices(df: pd.DataFrame, save_dir: str, format: PriceFileFormat) -> bool:
    """Reload the saved file and confirm it matches the in-memory data."""
    ticker = df.columns[0]
    loaded = load_prices(f"{ticker}.{format}", load_dir=save_dir)
    values_match = np.allclose(df.values, loaded.values, equal_nan=True)
    index_match = df.index.equals(loaded.index)
    if not (values_match and index_match):
        print(f"❗ Saved file for {ticker} does not match in-memory data!")
        return False
    print(f"💿 Successfully loaded {ticker} data through {format}")
    print(loaded)
    return True
