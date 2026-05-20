from datetime import date, datetime
from enum import StrEnum
from typing import Literal, TypedDict

import pandas as pd

PriceFileFormat = Literal["parquet", "csv"]


class AssetType(StrEnum):
    """Asset type enumeration."""

    STOCKS = "stocks"
    OPTIONS = "options"
    FOREX = "forex"
    CRYPTO = "crypto"


ASSET_TYPES: list[AssetType] = [
    AssetType.STOCKS,
    AssetType.OPTIONS,
    AssetType.FOREX,
    AssetType.CRYPTO,
]

ASSET_TYPE_CONFIG: dict[AssetType, dict[str, str]] = {
    AssetType.STOCKS: {"prefix": ""},
    AssetType.OPTIONS: {"prefix": "O:"},
    AssetType.FOREX: {"prefix": "C:"},
    AssetType.CRYPTO: {"prefix": "X:"},
}

DateLike = str | date | datetime | pd.Timestamp | None


class ChecksConfig(TypedDict):
    """Parameters for price data checks."""

    abs_rel_diff_pct_p50: float
    abs_rel_diff_pct_p99: float
    show_plot: bool
