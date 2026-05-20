from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src import Prices, check_prices
from src.constants import CHECKS_CONFIG
from src.schemas import AssetType, ChecksConfig
from src.utils import build_target_index
from tests.conftest import require_asset_data

CHECKS_CONFIG_NO_PLOT: ChecksConfig = {**CHECKS_CONFIG, "show_plot": False}


@pytest.fixture(scope="module")
def stocks_prices(data_dir: Path) -> Prices:
    require_asset_data(data_dir, AssetType.STOCKS)
    return Prices(data_dir=str(data_dir))


def _close_at(df: pd.DataFrame, et_date: str) -> float:
    ticker = df.columns[0]
    et_index = df.index.tz_convert("America/New_York")  # type: ignore[attr-defined]
    target = pd.Timestamp(et_date).date()
    mask = (et_index.date == target) & (et_index.hour == 15) & (et_index.minute == 59)
    bars = cast(pd.Series, df.loc[mask, ticker])
    assert len(bars) == 1, f"expected one 15:59 ET bar on {et_date}, got {len(bars)}"
    return float(bars.iloc[0])


def _run_pipeline(
    prices: Prices, ticker: str, start: str, end: str
) -> tuple[pd.DataFrame, AssetType]:
    df, asset_type = prices.get_prices(ticker=ticker, date_start=start, date_end=end)
    expected = build_target_index(
        cast(pd.Timestamp, df.index[0]), cast(pd.Timestamp, df.index[-1]), asset_type
    )
    assert df.index.equals(expected), "pipeline output index does not match calendar"
    assert (df[ticker] > 0).all(), "pipeline output contains non-positive prices"
    return df, asset_type


@pytest.mark.integration
def test_aapl_2014_split_applied(stocks_prices: Prices) -> None:
    # AAPL 7:1 split ex-date 2014-06-09. Last pre-split session is 2014-06-06.
    # After back-adjustment, the close-to-close ratio across the boundary should
    # be near 1 (daily noise), not near 7 (the raw, unadjusted ratio).
    df, asset_type = _run_pipeline(stocks_prices, "AAPL", "2014-05-01", "2014-07-31")
    assert asset_type == AssetType.STOCKS

    pre = _close_at(df, "2014-06-06")
    post = _close_at(df, "2014-06-09")
    ratio = pre / post
    assert 0.9 < ratio < 1.1, f"split-adjusted close ratio {ratio:.3f} (expected ~1)"

    assert check_prices(df, config=CHECKS_CONFIG_NO_PLOT, asset_type=asset_type)
