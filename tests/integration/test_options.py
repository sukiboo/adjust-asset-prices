from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src import Prices
from src.schemas import AssetType
from src.utils import parse_osi_ticker
from tests.conftest import quiet_check_options, quiet_get_options, require_asset_data

# Tolerance for a spanning contract's close-to-close ratio across the split (~1 after back-
# adjustment). Probe uses last/first REAL trades (not ffilled synthetic bars) on a deep-ITM
# contract, where elasticity ≈ 1 keeps the ratio near 1 rather than swinging on gamma.
CONTINUITY_EPS = 0.1
# A continuity probe must be at least 20% in-the-money (strike ≤ this × underlying at the split).
ITM_MAX_MONEYNESS = 0.8


@pytest.fixture(scope="module")
def options_prices(data_dir: Path) -> Prices:
    require_asset_data(data_dir, AssetType.OPTIONS)
    require_asset_data(data_dir, AssetType.STOCKS)  # underlying reference for the gate
    return Prices(data_dir=str(data_dir))


def _tickers(df: pd.DataFrame) -> set[str]:
    return set(df.index.get_level_values("ticker"))


def _real_continuity(df: pd.DataFrame, ticker: str, split_date: str) -> tuple[float, float]:
    """(last REAL trade before `split_date`, first REAL trade on/after it) for `ticker`, using
    the `is_real` flag. Real prints only — a fixed-time bar would be a backfilled (ffilled)
    synthetic value for illiquid contracts, making the continuity ratio vacuously 1.0."""
    series = df.xs(ticker, level="ticker")
    real = series[series["is_real"]]["close"]
    et_dates = real.index.tz_convert("America/New_York").date  # type: ignore[attr-defined]
    split = pd.Timestamp(split_date).date()
    before = real[et_dates < split]
    after = real[et_dates >= split]
    assert len(before) and len(
        after
    ), f"❌ {ticker} lacks real trades on both sides of {split_date}"
    return float(before.iloc[-1]), float(after.iloc[0])


def _describe_options(calls: pd.DataFrame, puts: pd.DataFrame, underlying: str) -> None:
    n_calls = calls.index.get_level_values("ticker").nunique() if not calls.empty else 0
    n_puts = puts.index.get_level_values("ticker").nunique() if not puts.empty else 0
    print(
        f"\n🔎 {underlying} options: {len(calls):,} call bars / {n_calls:,} contracts, "
        f"{len(puts):,} put bars / {n_puts:,} contracts"
    )


def _spot_at(ref: pd.DataFrame, underlying: str, split_date: str) -> float:
    s = ref[underlying]
    et = cast(pd.DatetimeIndex, s.index).tz_convert("America/New_York")
    on_split = s[(et.date == pd.Timestamp(split_date).date()) & (et.hour == 15) & (et.minute == 59)]
    return float(on_split.iloc[0]) if len(on_split) else float(s.median())


def _busiest_spanning_call(
    calls: pd.DataFrame, underlying: str, split_date: str, ref: pd.DataFrame | None = None
) -> str | None:
    # Busiest BASE-ROOT call with REAL (traded) bars on BOTH sides of `split_date` — so its
    # back-adjusted pre-split prints and native post-split prints can be compared. Requiring real
    # trades both sides (via `is_real`) excludes synthetic-strike successors that never traded
    # natively post-split, and OCC suffix roots (e.g. UVXY1) whose deliverable isn't continuous.
    # When `ref` is given, also require DEEP-ITM (strike ≤ ITM_MAX_MONEYNESS × underlying at the
    # split) — low elasticity, so a continuity ratio reflects the split adjustment, not ATM/OTM
    # gamma; without `ref` (e.g. a strike-consistency check), any moneyness qualifies. Returns
    # None if none qualifies (thin / reverse-split names have no deep-ITM two-sided call).
    spot = _spot_at(ref, underlying, split_date) if ref is not None else None
    real = calls[calls["is_real"]]
    tickers = real.index.get_level_values("ticker")
    ts = cast(pd.DatetimeIndex, real.index.get_level_values("timestamp_utc"))
    et_dates = ts.tz_convert("America/New_York").date
    split = pd.Timestamp(split_date).date()
    before = set(tickers[et_dates < split])
    after = set(tickers[et_dates >= split])
    cands = []
    for t in before & after:
        parsed = parse_osi_ticker(cast(str, t))
        if parsed.underlying != underlying:
            continue
        if spot is not None and parsed.strike > ITM_MAX_MONEYNESS * spot:
            continue
        cands.append(t)
    if not cands:
        return None
    return str(tickers[tickers.isin(cands)].value_counts().index[0])


@pytest.mark.integration
def test_nvda_2024_split(options_prices: Prices) -> None:
    # NVDA 10:1 split (ex 2024-06-10, clean strikes): the pre-split $1150 call back-adjusts to the
    # $115 successor O:NVDA240621C00115000 (÷10). Asserts the exact successor symbol (strike ÷10),
    # deep-ITM continuity ~1, and the structural gate. (Only the 10:1 applies in this window.)
    calls, puts, ref = quiet_get_options(options_prices, "NVDA", "2024-06-06", "2024-06-12")
    _describe_options(calls, puts, "NVDA")

    succ = "O:NVDA240621C00115000"
    assert succ in _tickers(calls), f"❌ {succ} (÷10 successor) missing from output"
    pre, post = _real_continuity(calls, succ, "2024-06-10")
    ratio = pre / post
    print(f"🪚  NVDA {succ}: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~10)")
    assert (
        1 - CONTINUITY_EPS < ratio < 1 + CONTINUITY_EPS
    ), f"❌ split-adjusted call continuity ratio {ratio:.3f} (expected ~1)"

    assert quiet_check_options(calls, puts, "NVDA", ref), "❌ NVDA structural gate failed"


@pytest.mark.integration
def test_aapl_2014_split_suffixed_root(options_prices: Prices) -> None:
    # AAPL 7:1 split (ex 2014-06-09) — the non-clean / OCC-suffixed-root case. The $440 put has a
    # non-clean ÷7 strike, so OCC re-struck it under the AAPL7 root; with the 2020 4:1 also
    # back-adjusted (÷28 total, Adj-Close convention) it lands at O:AAPL7150117P00015715 ($440/28).
    # Asserts the raw symbol is gone, the ÷28 suffixed successor exists, continuity ~1, and the gate
    # — exercising suffixed-root unification + multi-split back-adjustment (the pre-split-leak fix).
    calls, puts, ref = quiet_get_options(options_prices, "AAPL", "2014-06-06", "2014-06-10")
    _describe_options(calls, puts, "AAPL")

    assert "O:AAPL150117P00440000" not in _tickers(puts), "❌ raw pre-split symbol not rewritten"
    succ = "O:AAPL7150117P00015715"
    assert succ in _tickers(puts), f"❌ {succ} (÷28 suffixed-root successor) missing from output"
    pre, post = _real_continuity(puts, succ, "2014-06-09")
    ratio = pre / post
    print(f"🪚  AAPL {succ}: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~28)")
    assert (
        1 - CONTINUITY_EPS < ratio < 1 + CONTINUITY_EPS
    ), f"❌ split-adjusted put continuity ratio {ratio:.3f} (expected ~1)"

    assert quiet_check_options(calls, puts, "AAPL", ref), "❌ AAPL structural gate failed"


@pytest.mark.integration
def test_tsla_2022_split(options_prices: Prices) -> None:
    # TSLA 3:1 split (ex 2022-08-25) — a different ratio/underlying from NVDA's 10:1. Window
    # predates only this split (2020 5:1 is earlier). Deep-ITM continuity ~1 (not ~3) + gate.
    calls, puts, ref = quiet_get_options(options_prices, "TSLA", "2022-08-22", "2022-08-26")
    _describe_options(calls, puts, "TSLA")
    assert not calls.empty and not puts.empty, "❌ expected both TSLA calls and puts in range"

    succ = _busiest_spanning_call(calls, "TSLA", "2022-08-25", ref=ref)
    assert succ is not None, "❌ no deep-ITM call spans the TSLA split boundary"
    pre, post = _real_continuity(calls, succ, "2022-08-25")
    ratio = pre / post
    print(f"🪚  TSLA {succ}: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~3)")
    assert (
        1 - CONTINUITY_EPS < ratio < 1 + CONTINUITY_EPS
    ), f"❌ split-adjusted call continuity ratio {ratio:.3f} (expected ~1)"

    assert quiet_check_options(calls, puts, "TSLA", ref), "❌ TSLA structural gate failed"


@pytest.mark.integration
def test_aapl_2023_no_split(options_prices: Prices) -> None:
    # No-split window (AAPL's last split was 2020-08-31): adjust_options_splits is a no-op, so this
    # exercises the clean common path (load → backfill → gate, zero unification).
    calls, puts, ref = quiet_get_options(options_prices, "AAPL", "2023-06-01", "2023-06-09")
    _describe_options(calls, puts, "AAPL")
    assert not calls.empty and not puts.empty, "❌ expected both AAPL calls and puts in range"

    assert quiet_check_options(calls, puts, "AAPL", ref), "❌ AAPL structural gate failed"


@pytest.mark.integration
def test_nvda_2021_multi_split(options_prices: Prices) -> None:
    # Multi-split: window predates BOTH NVDA splits (2021 4:1 + 2024 10:1), so every contract is
    # back-adjusted by the cumulative ÷40, matching the ÷40 underlying. In-window 4:1 continuity ~1
    # (the 2024 ÷10 cancels in the pre/post ratio); the gate is the real multi-split check — a
    # missed 2024 ÷10 would sit ~10x off the underlying and blow the bounds.
    calls, puts, ref = quiet_get_options(options_prices, "NVDA", "2021-07-16", "2021-07-22")
    _describe_options(calls, puts, "NVDA")
    assert not calls.empty and not puts.empty, "❌ expected both NVDA calls and puts in range"

    succ = _busiest_spanning_call(calls, "NVDA", "2021-07-20", ref=ref)
    assert succ is not None, "❌ no deep-ITM call spans the NVDA 2021 split boundary"
    pre, post = _real_continuity(calls, succ, "2021-07-20")
    ratio = pre / post
    print(f"🪚  NVDA {succ}: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~4)")
    assert (
        1 - CONTINUITY_EPS < ratio < 1 + CONTINUITY_EPS
    ), f"❌ split-adjusted call continuity ratio {ratio:.3f} (expected ~1)"

    assert quiet_check_options(calls, puts, "NVDA", ref), "❌ NVDA structural gate failed"


@pytest.mark.integration
def test_tsla_2020_multi_split(options_prices: Prices) -> None:
    # Multi-split on a second underlying/ratio: window predates both TSLA splits (2020 5:1 + 2022
    # 3:1) → cumulative ÷15, matching the ÷15 underlying. In-window 5:1 continuity ~1 (the 2022 ÷3
    # cancels); the gate ties the ÷15 scaling to the underlying (a missed ÷3 sits ~3x off).
    calls, puts, ref = quiet_get_options(options_prices, "TSLA", "2020-08-27", "2020-09-02")
    _describe_options(calls, puts, "TSLA")
    assert not calls.empty and not puts.empty, "❌ expected both TSLA calls and puts in range"

    succ = _busiest_spanning_call(calls, "TSLA", "2020-08-31", ref=ref)
    assert succ is not None, "❌ no deep-ITM call spans the TSLA 2020 split boundary"
    pre, post = _real_continuity(calls, succ, "2020-08-31")
    ratio = pre / post
    print(f"🪚  TSLA {succ}: pre=${pre:.4f}, post=${post:.4f}, ratio={ratio:.4f} (raw ~5)")
    assert (
        1 - CONTINUITY_EPS < ratio < 1 + CONTINUITY_EPS
    ), f"❌ split-adjusted call continuity ratio {ratio:.3f} (expected ~1)"
    assert quiet_check_options(calls, puts, "TSLA", ref), "❌ TSLA structural gate failed"


# --- Reverse splits ---------------------------------------------------------------------------
# These verify the reverse-split branch (yfinance ratio < 1 → premium ×k, strike K→K×k) via the
# cumulative back-adjustment FACTOR (×20 / ×16 on the underlying + successor strike). They do NOT
# use a continuity probe: a reverse split only happens on thin / volatile names (penny stocks,
# vol ETPs), which have no deep-ITM call that traded both sides of the split — the only
# two-sided real contracts are ATM/OTM, whose prices swing 1.7-4.2x on gamma (not adjustment
# error). Whether they ALSO assert the structural gate splits by *why* the name is hard:
#   - penny stocks (actual pre-split price ~$1-3): the old bound violations were tick granularity
#     + stale illiquid prints, ×k-amplified — but that noise lived in SYNTHETIC backfilled bars
#     and DEEP-ITM real prints. The gate now scores REAL near-the-money bars only, so GRPN (1:20)
#     passes cleanly; it asserts the gate.
#   - vol/leveraged ETPs (VXX, UVXY, USO, ...): steep roll-decay carry drags the forward far
#     below spot, so ITM CALLS legitimately trade below S-K on REAL near-the-money bars too —
#     the spot-based floor is the wrong bound, and real-bar gating can't fix it. VXX still
#     SKIPS the gate (a carry-aware floor would be needed; see CLAUDE.md).
#   - high-yield mortgage REITs (CIM, ARR): dividend carry, same one-sided real call breach.
#   - confounded names (GE: GEHC/Vernova spinoffs divide the underlying but not the options;
#     AMC: simultaneous APE conversion breaks continuity).


@pytest.mark.integration
def test_grpn_2020_reverse_split(options_prices: Prices) -> None:
    # GRPN 1:20 reverse split (ex 2020-06-11, ×20). NO continuity probe — GRPN's option market is
    # too thin (only ATM/OTM strikes traded both sides, swinging 1.7-4.2x on gamma); the ×20
    # underlying factor + gate cover it. Gate ASSERTED: its old ~17% floor breach was synthetic/
    # deep-ITM noise, gone now that the gate scores real near-the-money bars.
    calls, puts, ref = quiet_get_options(options_prices, "GRPN", "2020-06-09", "2020-06-15")
    _describe_options(calls, puts, "GRPN")
    assert not calls.empty and not puts.empty, "❌ expected both GRPN calls and puts in range"

    # Back-adjusted underlying reflects the ×20 (GRPN ~$1.30 real → ~$26), confirming the
    # reverse direction multiplied rather than divided.
    ref_level = float(ref["GRPN"].median())
    assert (
        18 < ref_level < 34
    ), f"❌ underlying not back-adjusted by ×20 (median ${ref_level:.2f}, expect ~$26)"
    print(f"✔️  GRPN back-adjusted underlying median: ${ref_level:.2f} (×20 ~ $26, raw ~ $1.30)")

    assert quiet_check_options(calls, puts, "GRPN", ref), "❌ GRPN structural gate failed"


@pytest.mark.integration
def test_vxx_2023_multi_reverse_split(options_prices: Prices) -> None:
    # VXX: two 1:4 reverse splits (2023-03-07 + 2024-07-24), window predates both → cumulative ×16
    # (matching the ×16 underlying). The cumulative FACTOR pins this (continuity can't — no deep-ITM
    # two-sided probe, and the future ×4 cancels anyway): underlying lands at ×16 (~$180, not ~$45),
    # and the successor strike is consistent with it (a missed 2024 ×4 → ~4x-too-low strike).
    # Gate skipped — steep-contango carry fails the spot-based call floor with no split error.
    calls, puts, ref = quiet_get_options(options_prices, "VXX", "2023-03-03", "2023-03-09")
    _describe_options(calls, puts, "VXX")
    assert not calls.empty and not puts.empty, "❌ expected both VXX calls and puts in range"

    succ = _busiest_spanning_call(calls, "VXX", "2023-03-07")
    assert succ is not None, "❌ no real-both-sides call spans the VXX 2023 reverse-split boundary"

    # Cumulative ×16 check (underlying median + successor strike; distinguishes ×16 from ×4-only).
    ref_level = float(ref["VXX"].median())
    strike = parse_osi_ticker(succ).strike
    assert (
        150 < ref_level < 210
    ), f"❌ underlying not back-adjusted by cumulative ×16 (median ${ref_level:.2f}, expect ~$180)"
    assert 0.5 < strike / ref_level < 2.0, (
        f"❌ successor strike ${strike:.2f} inconsistent with ×16 underlying ${ref_level:.2f} "
        f"(a missed 2024 ×4 would re-strike ~4x too low → ratio ~0.22)"
    )
    print(
        f"✔️  VXX back-adjusted underlying median: ${ref_level:.2f} (×16 ~ $180, ×4-only ~ $45), "
        f"successor strike ${strike:.2f}"
    )
