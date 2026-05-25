from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src import Prices
from src.schemas import AssetType
from src.utils import parse_osi_ticker
from tests.conftest import quiet_check_options, quiet_get_options, require_asset_data


@pytest.fixture(scope="module")
def options_prices(data_dir: Path) -> Prices:
    require_asset_data(data_dir, AssetType.OPTIONS)
    require_asset_data(data_dir, AssetType.STOCKS)  # underlying reference for the gate
    return Prices(data_dir=str(data_dir))


def _tickers(df: pd.DataFrame) -> set[str]:
    return set(df.index.get_level_values("ticker"))


def _contract_close_at(df: pd.DataFrame, ticker: str, et_date: str) -> float:
    series = df.xs(ticker, level="ticker")["close"]
    et_index = series.index.tz_convert("America/New_York")  # type: ignore[attr-defined]
    target = pd.Timestamp(et_date).date()
    mask = (et_index.date == target) & (et_index.hour == 15) & (et_index.minute == 59)
    bars = series[mask]
    assert (
        len(bars) == 1
    ), f"❌ expected one 15:59 ET bar for {ticker} on {et_date}, got {len(bars)}"
    return float(bars.iloc[0])


def _describe_options(calls: pd.DataFrame, puts: pd.DataFrame, underlying: str) -> None:
    n_calls = calls.index.get_level_values("ticker").nunique() if not calls.empty else 0
    n_puts = puts.index.get_level_values("ticker").nunique() if not puts.empty else 0
    print(
        f"\n🔎 {underlying} options: {len(calls):,} call bars / {n_calls:,} contracts, "
        f"{len(puts):,} put bars / {n_puts:,} contracts"
    )


def _busiest_spanning_call(
    calls: pd.DataFrame, underlying: str, pre_date: str, post_date: str
) -> str | None:
    # Busiest BASE-ROOT call (parsed underlying == `underlying`) with a 15:59 ET bar on both
    # dates, i.e. spanning the split (the pre-date bar is the relabeled/÷ratio'd pre-split
    # print, the post-date bar is native post-split). Derived generically so the test doesn't
    # hardcode a strike that may not have traded. OCC numeric-suffix roots (e.g. UVXY1) are
    # excluded: they carry distinct adjusted deliverables whose continuity isn't ~1.
    ts_level = cast(pd.DatetimeIndex, calls.index.get_level_values("timestamp_utc"))
    et_index = ts_level.tz_convert("America/New_York")
    tickers = calls.index.get_level_values("ticker")

    def day_tickers(d: str) -> set[str]:
        mask = (
            (et_index.date == pd.Timestamp(d).date())
            & (et_index.hour == 15)
            & (et_index.minute == 59)
        )
        return set(tickers[mask])

    common = {
        t
        for t in day_tickers(pre_date) & day_tickers(post_date)
        if parse_osi_ticker(t).underlying == underlying
    }
    if not common:
        return None
    return str(tickers[tickers.isin(common)].value_counts().index[0])


@pytest.mark.integration
def test_nvda_2024_split(options_prices: Prices) -> None:
    # NVDA 10:1 split ex-date 2024-06-10 (clean strikes). The pre-split $1150 call
    # O:NVDA240621C01150000 back-adjusts to the $115 successor O:NVDA240621C00115000
    # (premium ÷10). Across the split boundary the successor's close-to-close ratio is
    # ~1 (continuous), not ~10 (the raw, unadjusted ratio). The structural gate then
    # validates no-arb self-consistency over the full window. (NVDA also split 4:1 in
    # 2021, but that's before this window, so only the 10:1 applies here.)
    calls, puts, ref = quiet_get_options(options_prices, "NVDA", "2024-06-06", "2024-06-12")
    _describe_options(calls, puts, "NVDA")

    succ = "O:NVDA240621C00115000"
    assert succ in _tickers(calls), f"❌ {succ} (÷10 successor) missing from output"
    pre = _contract_close_at(calls, succ, "2024-06-07")
    post = _contract_close_at(calls, succ, "2024-06-10")
    ratio = pre / post
    print(f"🪚  NVDA {succ}: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~10)")
    assert 0.5 < ratio < 2.0, f"❌ split-adjusted call continuity ratio {ratio:.3f} (expected ~1)"

    assert quiet_check_options(calls, puts, "NVDA", ref), "❌ NVDA structural gate failed"


@pytest.mark.integration
def test_aapl_2014_split_suffixed_root(options_prices: Prices) -> None:
    # AAPL 7:1 split ex-date 2014-06-09 — the canonical non-clean / OCC-suffixed-root case.
    # The pre-split $440 put O:AAPL150117P00440000 has a non-clean ÷7 strike ($62.857), so
    # OCC re-struck it under the AAPL7 root. AAPL also split 4:1 on 2020-08-31, and the
    # pipeline back-adjusts for ALL known splits (yfinance Adj Close convention), so this
    # 2014 contract lands in ÷28 currency: strike $440/28 ≈ $15.71 → O:AAPL7150117P00015715,
    # premium $1.62/28 ≈ $0.058. The raw symbol is rewritten away entirely (no post-split
    # reuse in a 2014-only window). Continuity across the in-window 2014 split is ~1 (raw
    # would be ~28). Exercises the suffixed-root unification path AND the multi-split
    # back-adjustment of suffixed-root contracts (the fix for the pre-split-leak bug).
    calls, puts, ref = quiet_get_options(options_prices, "AAPL", "2014-06-06", "2014-06-10")
    _describe_options(calls, puts, "AAPL")

    assert "O:AAPL150117P00440000" not in _tickers(puts), "❌ raw pre-split symbol not rewritten"
    succ = "O:AAPL7150117P00015715"
    assert succ in _tickers(puts), f"❌ {succ} (÷28 suffixed-root successor) missing from output"
    pre = _contract_close_at(puts, succ, "2014-06-06")
    post = _contract_close_at(puts, succ, "2014-06-09")
    ratio = pre / post
    print(f"🪚  AAPL {succ}: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~28)")
    assert 0.5 < ratio < 2.0, f"❌ split-adjusted put continuity ratio {ratio:.3f} (expected ~1)"

    assert quiet_check_options(calls, puts, "AAPL", ref), "❌ AAPL structural gate failed"


@pytest.mark.integration
def test_tsla_2022_split(options_prices: Prices) -> None:
    # TSLA 3:1 split ex-date 2022-08-25 — a clean-strike integer ratio distinct from NVDA's
    # 10:1, on a different (very liquid) underlying. The window predates only this split
    # (TSLA's 2020 5:1 is before it, so fetch_splits returns just the 3:1). The busiest call
    # spanning the boundary should be continuous (ratio ~1, not ~3) after ÷3 back-adjustment,
    # and the structural gate should pass over the whole window.
    calls, puts, ref = quiet_get_options(options_prices, "TSLA", "2022-08-22", "2022-08-26")
    _describe_options(calls, puts, "TSLA")
    assert not calls.empty and not puts.empty, "❌ expected both TSLA calls and puts in range"

    succ = _busiest_spanning_call(calls, "TSLA", "2022-08-24", "2022-08-25")
    assert succ is not None, "❌ no call spans the TSLA split boundary"
    pre = _contract_close_at(calls, succ, "2022-08-24")
    post = _contract_close_at(calls, succ, "2022-08-25")
    ratio = pre / post
    print(f"🪚  TSLA {succ}: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~3)")
    assert 0.5 < ratio < 2.0, f"❌ split-adjusted call continuity ratio {ratio:.3f} (expected ~1)"

    assert quiet_check_options(calls, puts, "TSLA", ref), "❌ TSLA structural gate failed"


@pytest.mark.integration
def test_aapl_2023_no_split(options_prices: Prices) -> None:
    # A no-split window (AAPL's last split was 2020-08-31), so adjust_options_splits is a
    # no-op and the pipeline exercises the common path: load → backfill → structural gate
    # with zero symbol unification. Complements the two split-spanning tests, which never
    # exercise the clean-window branch. Both sides should be non-empty and pass the gate.
    calls, puts, ref = quiet_get_options(options_prices, "AAPL", "2023-06-01", "2023-06-09")
    _describe_options(calls, puts, "AAPL")
    assert not calls.empty and not puts.empty, "❌ expected both AAPL calls and puts in range"

    assert quiet_check_options(calls, puts, "AAPL", ref), "❌ AAPL structural gate failed"


@pytest.mark.integration
def test_nvda_2021_multi_split(options_prices: Prices) -> None:
    # Multi-split back-adjustment. The window sits before BOTH of NVDA's modern splits, so
    # fetch_splits returns the 2021-07-20 4:1 AND the 2024-06-10 10:1 — every contract is
    # back-adjusted by the cumulative ÷40 (yfinance Adj-Close convention), matching the ÷40
    # underlying the stocks pass produces. The in-window 4:1 boundary must still be continuous
    # (busiest spanning call ratio ~1, not ~4; the 2024 ÷10 hits both sides and cancels in the
    # ratio). The structural gate is the real multi-split check: it compares options against
    # the ÷40 underlying, so a contract that missed the 2024 ÷10 factor would sit ~10x off the
    # underlying and blow the no-arb bounds. (NVDA's 2006/2007 splits predate the window, so
    # fetch_splits excludes them — the cumulative factor is exactly ÷40.)
    calls, puts, ref = quiet_get_options(options_prices, "NVDA", "2021-07-16", "2021-07-22")
    _describe_options(calls, puts, "NVDA")
    assert not calls.empty and not puts.empty, "❌ expected both NVDA calls and puts in range"

    succ = _busiest_spanning_call(calls, "NVDA", "2021-07-19", "2021-07-20")
    assert succ is not None, "❌ no call spans the NVDA 2021 split boundary"
    pre = _contract_close_at(calls, succ, "2021-07-19")
    post = _contract_close_at(calls, succ, "2021-07-20")
    ratio = pre / post
    print(f"🪚  NVDA {succ}: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~4)")
    assert 0.5 < ratio < 2.0, f"❌ split-adjusted call continuity ratio {ratio:.3f} (expected ~1)"

    assert quiet_check_options(calls, puts, "NVDA", ref), "❌ NVDA structural gate failed"


@pytest.mark.integration
def test_tsla_2020_multi_split(options_prices: Prices) -> None:
    # Multi-split back-adjustment on a second underlying/ratio pair. The window predates both
    # TSLA splits, so fetch_splits returns the 2020-08-31 5:1 AND the 2022-08-25 3:1 → every
    # contract is back-adjusted by the cumulative ÷15, matching the ÷15 underlying. The
    # in-window 5:1 boundary stays continuous (busiest spanning call ratio ~1, not ~5; the 2022
    # ÷3 cancels in the ratio), and the gate ties the cumulative ÷15 options scaling to the ÷15
    # underlying — a missed 2022 ÷3 factor would push contracts ~3x off the underlying.
    calls, puts, ref = quiet_get_options(options_prices, "TSLA", "2020-08-27", "2020-09-02")
    _describe_options(calls, puts, "TSLA")
    assert not calls.empty and not puts.empty, "❌ expected both TSLA calls and puts in range"

    succ = _busiest_spanning_call(calls, "TSLA", "2020-08-28", "2020-08-31")
    assert succ is not None, "❌ no call spans the TSLA 2020 split boundary"
    pre = _contract_close_at(calls, succ, "2020-08-28")
    post = _contract_close_at(calls, succ, "2020-08-31")
    ratio = pre / post
    print(f"🪚  TSLA {succ}: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~5)")
    assert 0.5 < ratio < 2.0, f"❌ split-adjusted call continuity ratio {ratio:.3f} (expected ~1)"
    assert quiet_check_options(calls, puts, "TSLA", ref), "❌ TSLA structural gate failed"


# --- Reverse splits ---------------------------------------------------------------------------
# These verify the reverse-split branch (yfinance ratio < 1 → premium ×k, strike K→K×k) via
# split CONTINUITY and the cumulative back-adjustment factor. Whether they ALSO assert the
# structural gate now splits by *why* a reverse-split name is hard:
#   - penny stocks (the actual pre-split price is ~$1-3): the bound violations were tick
#     granularity + stale illiquid prints, ×k-amplified — but that noise lived in SYNTHETIC
#     backfilled bars and DEEP-ITM real prints. Now the gate scores REAL near-the-money bars
#     only, so GRPN (1:20) passes cleanly; it asserts the gate.
#   - vol/leveraged ETPs (VXX, UVXY, USO, ...): steep roll-decay carry drags the forward far
#     below spot, so ITM CALLS legitimately trade below S-K on REAL near-the-money bars too —
#     the spot-based floor is the wrong bound, and real-bar gating can't fix it. VXX still
#     SKIPS the gate (a carry-aware floor would be needed; see CLAUDE.md).
#   - high-yield mortgage REITs (CIM, ARR): dividend carry, same one-sided real call breach.
#   - confounded names (GE: GEHC/Vernova spinoffs divide the underlying but not the options;
#     AMC: simultaneous APE conversion breaks continuity).
# In every tested case the split ADJUSTMENT is correct (continuity = 1.0).


@pytest.mark.integration
def test_grpn_2020_reverse_split(options_prices: Prices) -> None:
    # GRPN 1:20 reverse split ex 2020-06-11 (yfinance ratio 0.05 → 1/0.05 = 20). A 1:20 reverse
    # raises the price ~20x, so pre-split premiums are multiplied by 20 and strikes rewritten
    # K→K×20; the busiest spanning call is continuous (ratio ~1, not ~0.05 = the raw 1/20 jump).
    # GRPN's only corporate action is this split (no spinoff/dividend confound). Gate ASSERTED:
    # GRPN's old ~17% floor breach was deep-ITM/synthetic noise (GRPN traded ~$1.30 pre-split,
    # so penny-tick stale prints ×20); on REAL near-the-money bars it passes (see section above).
    calls, puts, ref = quiet_get_options(options_prices, "GRPN", "2020-06-09", "2020-06-15")
    _describe_options(calls, puts, "GRPN")
    assert not calls.empty and not puts.empty, "❌ expected both GRPN calls and puts in range"

    succ = _busiest_spanning_call(calls, "GRPN", "2020-06-10", "2020-06-11")
    assert succ is not None, "❌ no call spans the GRPN reverse-split boundary"
    pre = _contract_close_at(calls, succ, "2020-06-10")
    post = _contract_close_at(calls, succ, "2020-06-11")
    ratio = pre / post
    print(f"🪚  GRPN {succ}: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~0.05)")
    assert 0.5 < ratio < 2.0, f"❌ reverse-split-adjusted call continuity {ratio:.3f} (expected ~1)"

    # Back-adjusted underlying reflects the ×20 (GRPN ~$1.30 real → ~$26), confirming the
    # reverse direction multiplied rather than divided.
    ref_level = float(ref["GRPN"].median())
    assert ref_level > 10, f"❌ underlying not back-adjusted by ×20 (median ${ref_level:.2f})"
    print(f"✔️  GRPN back-adjusted underlying median: ${ref_level:.2f} (×20 ~ $26, raw ~ $1.30)")

    assert quiet_check_options(calls, puts, "GRPN", ref), "❌ GRPN structural gate failed"


@pytest.mark.integration
def test_vxx_2023_multi_reverse_split(options_prices: Prices) -> None:
    # Multiple REVERSE splits. The window predates BOTH of VXX's in-range 1:4 reverse splits
    # (2023-03-07 and 2024-07-24, each yfinance ratio 0.25 → 1/0.25 = 4), so every contract is
    # back-adjusted by the cumulative ×16 (premium ×16, strike K→K×16), matching the ×16
    # underlying. Two checks pin this down, because continuity alone can't (the future ×4 hits
    # both sides of the in-window boundary and cancels in the pre/post ratio):
    #   (a) the in-window 2023-03-07 boundary is continuous (ratio ~1, not ~0.25), and
    #   (b) the CUMULATIVE factor is right — the back-adjusted underlying lands at the ×16 level
    #       (~$180, not ~$45 = ×4-only), and the spanning successor's strike is consistent with
    #       that ×16 underlying (a missed 2024 ×4 would re-strike the option ~4x too low → the
    #       busiest spanning successor would be a ~$38 strike against the ~$180 underlying).
    # Gate skipped — VXX is a steep-contango vol ETP whose roll-decay carry fails the CALL floor
    # ~37-47% with no split error (see the section comment above).
    calls, puts, ref = quiet_get_options(options_prices, "VXX", "2023-03-03", "2023-03-09")
    _describe_options(calls, puts, "VXX")
    assert not calls.empty and not puts.empty, "❌ expected both VXX calls and puts in range"

    succ = _busiest_spanning_call(calls, "VXX", "2023-03-06", "2023-03-07")
    assert succ is not None, "❌ no call spans the VXX 2023 reverse-split boundary"
    pre = _contract_close_at(calls, succ, "2023-03-06")
    post = _contract_close_at(calls, succ, "2023-03-07")
    ratio = pre / post
    print(f"🪚  VXX {succ}: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~0.25)")
    assert 0.5 < ratio < 2.0, f"❌ reverse-split-adjusted call continuity {ratio:.3f} (expected ~1)"

    # Cumulative ×16 check (distinguishes ×16 from the in-window-only ×4).
    ref_level = float(ref["VXX"].median())
    strike = parse_osi_ticker(succ).strike
    assert (
        ref_level > 100
    ), f"❌ underlying not back-adjusted by cumulative ×16 (median ${ref_level:.2f})"
    assert 0.4 < strike / ref_level < 4.0, (
        f"❌ successor strike ${strike:.2f} inconsistent with ×16 underlying ${ref_level:.2f} "
        f"(a missed 2024 ×4 would re-strike ~4x too low)"
    )
    print(
        f"✔️  VXX back-adjusted underlying median: ${ref_level:.2f} (×16 ~ $180, ×4-only ~ $45), "
        f"successor strike ${strike:.2f}"
    )
