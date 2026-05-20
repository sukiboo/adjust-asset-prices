from pathlib import Path

import pytest

from src.constants import DEFAULT_DATA_DIR
from src.schemas import AssetType


@pytest.fixture(scope="session")
def data_dir() -> Path:
    p = Path(DEFAULT_DATA_DIR).expanduser().resolve()
    if not p.is_dir():
        pytest.skip(f"Data directory {p} does not exist.")
    return p


def require_asset_data(data_dir: Path, asset_type: AssetType) -> None:
    sub = data_dir / asset_type
    if not (sub.is_dir() and any(sub.glob("*.csv.gz"))):
        pytest.skip(f"No raw {asset_type} files at {sub}.")
