from datetime import date, datetime
from typing import Literal

import pandas as pd

AssetType = Literal["stocks", "options", "forex", "crypto"]

ASSET_TYPES: list[AssetType] = ["stocks", "options", "forex", "crypto"]

ASSET_TYPE_CONFIG: dict[AssetType, dict[str, str]] = {
    "stocks": {"prefix": ""},
    "options": {"prefix": "O:"},
    "forex": {"prefix": "C:"},
    "crypto": {"prefix": "X:"},
}

DateLike = str | date | datetime | pd.Timestamp | None
