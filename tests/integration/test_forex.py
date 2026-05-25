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
    # EUR-USD over a 2-month window (~43 sample days for the daily compare). Covers two
    # structural edge cases at once: the spring DST transition on 2025-03-30 (which
    # exercises the London-wall-clock date alignment in _our_daily_close / _yf_daily_close
    # — under UTC-date indexing this region would collapse Sun-GMT and Mon-BST midnights
    # onto the same UTC date) and the April 2 "Liberation Day" tariff announcement, where
    # EUR-USD ripped overnight and a single day's abs_diff hit ~0.6%. The 2-month sample
    # is large enough to dilute that tariff day out of the p99 tail; a 5-day window had
    # left it dominating the gate. compare_to_yf translates our `EUR-USD` to yfinance's
    # `EURUSD=X` internally; without that translation, yfinance returns a 404 and the
    # gate would fail.
    df, asset_type = quiet_get(forex_prices, "EUR-USD", "2025-03-01", "2025-04-30")
    assert asset_type == AssetType.FOREX, "❌ asset type misdetected (expected FOREX)"
    describe_adjusted_prices(df, "EUR-USD")
    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"


@pytest.mark.integration
def test_eur_usd_2022_2024(forex_prices: Prices) -> None:
    # EUR-USD across 2022-2024: ~1.5M-bar continuous 1-min grid with ~150 weekend
    # gaps backfilled by interpolation. Exercises the forex path at scale, including
    # the yfinance EURUSD=X translation and the daily-close tz alignment across
    # multiple year-end boundaries.
    df, asset_type = quiet_get(forex_prices, "EUR-USD", "2022-01-01", "2024-12-31")
    assert asset_type == AssetType.FOREX, "❌ asset type misdetected (expected FOREX)"
    describe_adjusted_prices(df, "EUR-USD")
    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"


@pytest.mark.integration
def test_usd_jpy_2022(forex_prices: Prices) -> None:
    # USD-JPY across 2022: yen weakened from ~115 to ~150, prompting the BoJ's first
    # FX intervention in 24 years (Sept-Oct). High-nominal-value quote (~100-150 vs
    # EUR-USD's ~1.05) confirms the relative-diff comparison path is not silently
    # scale-sensitive, and exercises the USDJPY=X translation. Stresses the London-
    # midnight daily-close alignment in _our_daily_close — under the prior UTC-midnight
    # alignment, intervention-day moves in the 23:00 → 23:59 UTC window blew abs_p99
    # well past the 1.0% forex threshold.
    # End at 2022-12-30 (Fri): Polygon's 2022-12-31 file is a thin holiday Saturday with
    # ~7k rows and no USD-JPY ticker, which trips determine_asset_type's last-file probe.
    df, asset_type = quiet_get(forex_prices, "USD-JPY", "2022-01-01", "2022-12-30")
    assert asset_type == AssetType.FOREX, "❌ asset type misdetected (expected FOREX)"
    describe_adjusted_prices(df, "USD-JPY")
    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"


@pytest.mark.integration
def test_eur_gbp_2023(forex_prices: Prices) -> None:
    # EUR-GBP across 2023: a cross-pair with no USD leg, exercising the EURGBP=X
    # translation path independently of any USD-quoted shortcut. Range chosen to
    # not overlap with test_usd_jpy_2022.
    df, asset_type = quiet_get(forex_prices, "EUR-GBP", "2023-01-01", "2023-12-31")
    assert asset_type == AssetType.FOREX, "❌ asset type misdetected (expected FOREX)"
    describe_adjusted_prices(df, "EUR-GBP")
    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"
