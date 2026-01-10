from datetime import date, datetime
from pathlib import Path
from typing import Tuple, cast

import pandas as pd

from .schemas import ASSET_TYPE_CONFIG, ASSET_TYPES, AssetType, DateLike


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
