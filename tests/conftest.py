import io
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from src import Prices, check_options, check_prices
from src.constants import CHECKS_CONFIG, DEFAULT_DATA_DIR, OPTIONS_CHECKS_CONFIG
from src.schemas import AssetType
from src.utils import describe_adjusted_prices

__all__ = ["describe_adjusted_prices"]


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


@contextmanager
def quiet_output(keep_substrs: tuple[str, ...] = ()) -> Iterator[None]:
    # Capture stdout for the duration of the block. On exit:
    # - success: re-emit only lines containing one of keep_substrs.
    # - exception: dump the full capture so failures stay diagnosable inline.
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        sys.stdout = old_stdout
        captured = buf.getvalue()
        if failed:
            sys.stdout.write(captured)
            sys.stdout.flush()
        elif keep_substrs:
            for line in captured.splitlines():
                if any(s in line for s in keep_substrs):
                    print(line)


def quiet_get(
    prices: Prices, ticker: str, date_start: str, date_end: str, dividends: bool = False
) -> tuple[pd.DataFrame, AssetType]:
    with quiet_output():
        return prices.asset.get_prices(
            ticker=ticker, date_start=date_start, date_end=date_end, dividends=dividends
        )


def quiet_check(df: pd.DataFrame, asset_type: AssetType, dividends_adjusted: bool = False) -> bool:
    with quiet_output(("Price comparison", "violate the threshold")):
        return check_prices(
            df,
            config=CHECKS_CONFIG,
            asset_type=asset_type,
            show_plot=False,
            dividends_adjusted=dividends_adjusted,
        )


def quiet_get_options(
    prices: Prices, underlying: str, date_start: str, date_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Option contracts plus the split-only underlying reference the gate needs, retrieved via
    # the same paths Prices.process uses (get_prices for the underlying, dividends off; then
    # get_options for the contracts). Returns (calls, puts, underlying).
    with quiet_output():
        underlying_df, _ = prices.asset.get_prices(
            underlying, date_start, date_end, dividends=False
        )
        calls, puts = prices.options.get_options(underlying, date_start, date_end)
    return calls, puts, underlying_df


def quiet_check_options(
    calls: pd.DataFrame, puts: pd.DataFrame, underlying: str, underlying_df: pd.DataFrame
) -> bool:
    with quiet_output(("≤", "intrinsic floor", "bars positive", "within contract")):
        return check_options(calls, puts, underlying, underlying_df, OPTIONS_CHECKS_CONFIG)
