from datetime import date, datetime
from enum import StrEnum
from typing import Literal, NamedTuple, TypedDict

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
    """Per-asset comparison thresholds used by `compare_to_yf`."""

    abs_rel_diff_pct_p50: float
    abs_rel_diff_pct_p99: float


class OSIContract(NamedTuple):
    """Parsed OSI (Options Symbology Initiative) option ticker components.
    Underlying is the root symbol as emitted by Polygon (no yfinance normalization);
    strike is in dollars (raw 8-digit field divided by 1000).
    """

    underlying: str
    expiry: date
    option_type: Literal["C", "P"]
    strike: float
