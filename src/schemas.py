from datetime import date, datetime
from enum import Enum

import pandas as pd


class AssetType(str, Enum):
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
