from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src import Prices
from src.schemas import AssetType
from tests.conftest import (
    describe_adjusted_prices,
    quiet_check,
    quiet_get,
    require_asset_data,
)

# Tolerance for the close-to-close ratio across a split boundary (~1 after back-adjustment).
# Tighter than options' 0.1: stocks are liquid (real boundary print, no ffill) and unleveraged.
CONTINUITY_EPS = 0.05


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
    assert len(bars) == 1, f"❌ expected one 15:59 ET bar on {et_date}, got {len(bars)}"
    return float(bars.iloc[0])


@pytest.mark.integration
def test_aapl_2014_split(stocks_prices: Prices) -> None:
    # AAPL 7:1 split (ex 2014-06-09): back-adjusted close ratio across the boundary ~1, not ~7.
    # Default split-only path → compared against yfinance's raw Close.
    df, asset_type = quiet_get(stocks_prices, "AAPL", "2014-05-01", "2014-07-31")
    assert asset_type == AssetType.STOCKS, "❌ asset type misdetected (expected STOCKS)"
    describe_adjusted_prices(df, "AAPL")

    pre = _close_at(df, "2014-06-06")
    post = _close_at(df, "2014-06-09")
    ratio = pre / post
    print(
        f"🪚  AAPL split 2014-06-09: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~7.0)"
    )
    assert (
        1 - CONTINUITY_EPS < ratio < 1 + CONTINUITY_EPS
    ), f"❌ split-adjusted close ratio {ratio:.3f} (expected ~1)"

    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"


@pytest.mark.integration
def test_aapl_2023_dividend_conventions(stocks_prices: Prices) -> None:
    # AAPL 2023 (4 dividends, no split) run BOTH ways: the convention switch in compare_to_yf —
    # split-only ↔ raw Close, dividend-adjusted ↔ Adj Close. The div-adj series sits below the
    # split-only one; a wrong convention pick would fail one side.
    split_only, asset_type = quiet_get(
        stocks_prices, "AAPL", "2023-01-01", "2023-12-31", dividends=False
    )
    div_adj, _ = quiet_get(stocks_prices, "AAPL", "2023-01-01", "2023-12-31", dividends=True)
    assert asset_type == AssetType.STOCKS, "❌ asset type misdetected (expected STOCKS)"
    describe_adjusted_prices(split_only, "AAPL")

    first_split = float(split_only["AAPL"].iloc[0])
    first_div = float(div_adj["AAPL"].iloc[0])
    print(f"🔩 AAPL 2023 first bar: split-only=${first_split:.4f}, div-adj=${first_div:.4f}")
    assert first_div < first_split, "❌ dividend back-adjustment should lower early-window prices"

    assert quiet_check(
        split_only, asset_type, dividends_adjusted=False
    ), "❌ split-only output failed yfinance comparison"
    assert quiet_check(
        div_adj, asset_type, dividends_adjusted=True
    ), "❌ dividend-adjusted output failed yfinance comparison"


@pytest.mark.integration
def test_spy_2020_2023(stocks_prices: Prices) -> None:
    # SPY 2020-2023 (no split, ~16 compounding dividends, half-days): the dividend path at
    # multi-year scale, compared against yfinance Adj Close.
    df, asset_type = quiet_get(stocks_prices, "SPY", "2020-01-01", "2023-12-31", dividends=True)
    assert asset_type == AssetType.STOCKS, "❌ asset type misdetected (expected STOCKS)"
    describe_adjusted_prices(df, "SPY")
    assert quiet_check(
        df, asset_type, dividends_adjusted=True
    ), "❌ dividend-adjusted price comparison to yfinance failed"


@pytest.mark.integration
def test_nvda_2020_2024_splits(stocks_prices: Prices) -> None:
    # NVDA 2020-2024: two splits (4:1, 10:1) + quarterly dividends — adjust_for_splits applied
    # twice and the split-then-dividend ordering. Adjusted close ratio ~1 across both boundaries.
    df, asset_type = quiet_get(stocks_prices, "NVDA", "2020-01-01", "2024-12-31", dividends=True)
    assert asset_type == AssetType.STOCKS, "❌ asset type misdetected (expected STOCKS)"
    describe_adjusted_prices(df, "NVDA")

    for split_date, prev_session, raw_ratio in [
        ("2021-07-20", "2021-07-19", "~4.0"),
        ("2024-06-10", "2024-06-07", "~10.0"),
    ]:
        pre = _close_at(df, prev_session)
        post = _close_at(df, split_date)
        ratio = pre / post
        print(
            f"🪚  NVDA split {split_date}: pre=${pre:.4f}, post=${post:.4f}, "
            f"ratio={ratio:.4f} (raw {raw_ratio})"
        )
        assert (
            1 - CONTINUITY_EPS < ratio < 1 + CONTINUITY_EPS
        ), f"❌ NVDA split-adjusted ratio at {split_date} = {ratio:.3f}"

    assert quiet_check(
        df, asset_type, dividends_adjusted=True
    ), "❌ dividend-adjusted price comparison to yfinance failed"


@pytest.mark.integration
def test_ge_2021_reverse_split(stocks_prices: Prices) -> None:
    # GE 1:8 reverse split (ex 2021-08-02, yfinance ratio 0.125 → ×8): adjusted close ratio ~1
    # (raw ~0.125). Window stays before GE's 2023 GEHC spinoff (not adjusted for).
    df, asset_type = quiet_get(stocks_prices, "GE", "2021-07-01", "2021-09-30")
    assert asset_type == AssetType.STOCKS, "❌ asset type misdetected (expected STOCKS)"
    describe_adjusted_prices(df, "GE")

    pre = _close_at(df, "2021-07-30")
    post = _close_at(df, "2021-08-02")
    ratio = pre / post
    print(
        f"🪚  GE split 2021-08-02: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~0.125)"
    )
    assert (
        1 - CONTINUITY_EPS < ratio < 1 + CONTINUITY_EPS
    ), f"❌ GE reverse-split-adjusted ratio = {ratio:.3f}"

    assert quiet_check(df, asset_type), "❌ price comparison to yfinance failed"


@pytest.mark.integration
def test_qyld_2023_distributions(stocks_prices: Prices) -> None:
    # QYLD monthly ROC distributions (yfinance lumps them under .dividends): the pipeline treats
    # them as ordinary dividends, matching yfinance's Adj Close, so the gate still passes.
    df, asset_type = quiet_get(stocks_prices, "QYLD", "2023-01-01", "2023-12-31", dividends=True)
    assert asset_type == AssetType.STOCKS, "❌ asset type misdetected (expected STOCKS)"
    describe_adjusted_prices(df, "QYLD")
    assert quiet_check(
        df, asset_type, dividends_adjusted=True
    ), "❌ dividend-adjusted price comparison to yfinance failed"


@pytest.mark.integration
def test_msft_2004_special_dividend(stocks_prices: Prices) -> None:
    # MSFT $3.08 special dividend (ex 2004-11-15, ~10% drop on a ~$30 stock): adjusted pre/post
    # ratio ~1, vs ~1.11 unadjusted — a drop big enough to bound cleanly (unlike a ~2% special).
    df, asset_type = quiet_get(stocks_prices, "MSFT", "2004-10-01", "2004-12-31", dividends=True)
    assert asset_type == AssetType.STOCKS, "❌ asset type misdetected (expected STOCKS)"
    describe_adjusted_prices(df, "MSFT")

    pre = _close_at(df, "2004-11-12")
    post = _close_at(df, "2004-11-15")
    ratio = pre / post
    print(
        f"🪏  MSFT special-div 2004-11-15: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~1.11)"
    )
    # Upper bound at 1.0: stocks rarely drop the full dividend intraday (observed 0.98), and
    # 1.0 cleanly excludes the unadjusted case (~1.11).
    assert 0.95 < ratio < 1.0, f"❌ MSFT special-div adjusted ratio = {ratio:.4f}"

    assert quiet_check(
        df, asset_type, dividends_adjusted=True
    ), "❌ dividend-adjusted price comparison to yfinance failed"


@pytest.mark.integration
def test_qqq_qqqq_rename_stitch(stocks_prices: Prices) -> None:
    # QQQ traded as QQQQ until 2011-03. Requesting QQQ with a --date-start before that surfaces a
    # leading gap that auto-stitches the QQQQ predecessor (matched by the stops + liquidity +
    # nearest-price filters). The stitched QQQ+QQQQ series matches yfinance's rename-aware
    # continuous QQQ. (Leading gap → no pre-gap segment, so the reused-ticker check is a no-op; the
    # interior genuine-rename path is validated manually — QQQ 2004→2011 — but too slow for CI.)
    df, asset_type = quiet_get(stocks_prices, "QQQ", "2010-06-01", "2011-12-31")
    assert asset_type == AssetType.STOCKS, "❌ asset type misdetected (expected STOCKS)"
    describe_adjusted_prices(df, "QQQ")

    # The QQQQ era (pre-2011-03) is stitched in, not left as a gap: bars exist in 2010.
    years = set(df.index.year)  # type: ignore[attr-defined]
    assert 2010 in years, "❌ no bars in the QQQQ era (2010) — predecessor not stitched"

    assert quiet_check(df, asset_type), "❌ stitched QQQ+QQQQ failed yfinance comparison"


@pytest.mark.integration
def test_meta_reused_ticker_drop(stocks_prices: Prices) -> None:
    # "META" was Meta Materials (~$12-15) until early 2022, then Facebook took the symbol on its
    # FB→META rename (2022-06-09) — two unrelated companies under one ticker. The window holds the
    # Meta-Materials Jan-2022 tail + Facebook-META from the rename. The loader picks up the foreign
    # head; the stitcher detects it (the predecessor FB connects only on the resume edge, not the
    # pre-gap edge) and drops it, then stitches FB back for the leading span. The result is
    # continuous Facebook history matching yfinance's rename-aware META.
    df, asset_type = quiet_get(stocks_prices, "META", "2022-01-01", "2022-09-30")
    assert asset_type == AssetType.STOCKS, "❌ asset type misdetected (expected STOCKS)"
    describe_adjusted_prices(df, "META")

    # The foreign Meta-Materials head (~$12-15) is gone: every price is Facebook-scale (well above
    # $100 across this window; Meta Materials never traded above ~$20).
    low = float(df["META"].min())
    print(f"🧹 META min price after drop: ${low:.2f} (Meta Materials traded ~$12-15)")
    assert low > 50, f"❌ foreign Meta-Materials head not dropped (min ${low:.2f})"

    assert quiet_check(df, asset_type), "❌ stitched FB→META failed yfinance comparison"


@pytest.mark.integration
def test_bbby_2023_delisting(stocks_prices: Prices) -> None:
    # BBBY pre-bankruptcy: yfinance serves a heavily-adjusted history (~$17-25) that no longer
    # matches the real Polygon traded prices (~$0.2-7) — ~92% divergence, so check_prices returns
    # False. The test INVERTS the assertion (`not quiet_check`): a documented known-divergence gap,
    # and a canary if yfinance ever changes its delisting handling.
    df, asset_type = quiet_get(stocks_prices, "BBBY", "2023-01-01", "2023-04-21")
    assert asset_type == AssetType.STOCKS, "❌ asset type misdetected (expected STOCKS)"
    describe_adjusted_prices(df, "BBBY")
    assert not quiet_check(
        df, asset_type
    ), "❌ BBBY/yfinance compare unexpectedly passes -- yfinance delisting handling may have changed"
