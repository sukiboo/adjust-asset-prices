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
def crypto_prices(data_dir: Path) -> Prices:
    require_asset_data(data_dir, AssetType.CRYPTO)
    return Prices(data_dir=str(data_dir))


@pytest.mark.integration
def test_btc_usd_luna_crash(crypto_prices: Prices) -> None:
    # BTC-USD over the May-Jun 2022 Luna/Terra collapse: a high-volatility stress of the gate
    # against crypto's looser thresholds, which absorb yfinance's daily-vs-1-min Close noise.
    df, asset_type = quiet_get(crypto_prices, "BTC-USD", "2022-05-01", "2022-06-30")
    assert asset_type == AssetType.CRYPTO, "❌ asset type misdetected (expected CRYPTO)"
    describe_adjusted_prices(df, "BTC-USD")
    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"


@pytest.mark.integration
def test_btc_usd_2020_2022(crypto_prices: Prices) -> None:
    # BTC-USD 2020-2022 (~1.5M bars): backfill + comparison at scale across COVID/bull/FTX —
    # catches regressions that only surface on large inputs (memory, accumulated tz drift).
    df, asset_type = quiet_get(crypto_prices, "BTC-USD", "2020-01-01", "2022-12-31")
    assert asset_type == AssetType.CRYPTO, "❌ asset type misdetected (expected CRYPTO)"
    describe_adjusted_prices(df, "BTC-USD")
    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"


@pytest.mark.integration
def test_eth_usd_2021_2022(crypto_prices: Prices) -> None:
    # ETH-USD 2021-2022 (bull run, Merge, Luna, FTX): a second high-liquidity coin, surfacing
    # any per-ticker yfinance quirks in the comparison path.
    df, asset_type = quiet_get(crypto_prices, "ETH-USD", "2021-01-01", "2022-12-31")
    assert asset_type == AssetType.CRYPTO, "❌ asset type misdetected (expected CRYPTO)"
    describe_adjusted_prices(df, "ETH-USD")
    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"


@pytest.mark.integration
def test_sol_usd_2022_2023(crypto_prices: Prices) -> None:
    # SOL-USD 2022-2023: lower-liquidity coin with an extreme drawdown (~$170→$8 via FTX, then
    # ~15x recovery) — the harshest test of the crypto thresholds over a long range.
    df, asset_type = quiet_get(crypto_prices, "SOL-USD", "2022-01-01", "2023-12-31")
    assert asset_type == AssetType.CRYPTO, "❌ asset type misdetected (expected CRYPTO)"
    describe_adjusted_prices(df, "SOL-USD")
    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"
