from pathlib import Path

import pytest

from src import Prices
from src.schemas import AssetType
from tests.conftest import (
    describe_adjusted_prices,
    quiet_check,
    quiet_get,
    require_asset_data,
)


@pytest.fixture(scope="module")
def forex_prices(data_dir: Path) -> Prices:
    require_asset_data(data_dir, AssetType.FOREX)
    return Prices(data_dir=str(data_dir))


@pytest.mark.integration
def test_eur_usd_202503_202505(forex_prices: Prices) -> None:
    # EUR-USD 2-month window for two edge cases: the 2025-03-30 spring DST transition (exercises
    # the London-wall-clock date alignment in _our/_yf_daily_close — UTC indexing would merge
    # Sun-GMT/Mon-BST midnights) and the Apr 2 tariff spike (~0.6%/day, diluted out of p99 by the
    # 2-month sample). Also the EUR-USD→EURUSD=X translation (yfinance 404s on the hyphenated form).
    df, asset_type = quiet_get(forex_prices, "EUR-USD", "2025-03-01", "2025-04-30")
    assert asset_type == AssetType.FOREX, "❌ asset type misdetected (expected FOREX)"
    describe_adjusted_prices(df, "EUR-USD")
    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"


@pytest.mark.integration
def test_eur_usd_2022_2024(forex_prices: Prices) -> None:
    # EUR-USD 2022-2024 (~1.5M bars, ~150 weekend gaps): the forex path at scale — EURUSD=X
    # translation + daily-close tz alignment across multiple year-ends.
    df, asset_type = quiet_get(forex_prices, "EUR-USD", "2022-01-01", "2024-12-31")
    assert asset_type == AssetType.FOREX, "❌ asset type misdetected (expected FOREX)"
    describe_adjusted_prices(df, "EUR-USD")
    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"


@pytest.mark.integration
def test_usd_jpy_2022(forex_prices: Prices) -> None:
    # USD-JPY 2022 (BoJ intervention; ~115→150): high-nominal quote (~100-150 vs EUR-USD ~1.05)
    # confirms the relative-diff path isn't scale-sensitive + stresses the London-midnight
    # daily-close alignment (UTC-midnight blew abs_p99 past threshold on intervention days).
    # Ends 2022-12-30: the 12-31 file is a thin holiday Saturday with no USD-JPY → trips the
    # last-file asset-type probe.
    df, asset_type = quiet_get(forex_prices, "USD-JPY", "2022-01-01", "2022-12-30")
    assert asset_type == AssetType.FOREX, "❌ asset type misdetected (expected FOREX)"
    describe_adjusted_prices(df, "USD-JPY")
    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"


@pytest.mark.integration
def test_eur_gbp_2023(forex_prices: Prices) -> None:
    # EUR-GBP 2023: a cross-pair (no USD leg), exercising EURGBP=X translation independently.
    df, asset_type = quiet_get(forex_prices, "EUR-GBP", "2023-01-01", "2023-12-31")
    assert asset_type == AssetType.FOREX, "❌ asset type misdetected (expected FOREX)"
    describe_adjusted_prices(df, "EUR-GBP")
    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"
