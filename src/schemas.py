from datetime import date, datetime
from enum import StrEnum
from typing import Literal, NamedTuple, TypedDict

import pandas as pd

PriceFileFormat = Literal["parquet", "csv"]


class AssetType(StrEnum):
    """Asset type enumeration."""

    STOCKS = "stocks"
    OPTIONS = "options"
    FOREX = "forex"
    CRYPTO = "crypto"


ASSET_TYPES: list[AssetType] = [
    AssetType.STOCKS,
    AssetType.OPTIONS,
    AssetType.FOREX,
    AssetType.CRYPTO,
]

ASSET_TYPE_CONFIG: dict[AssetType, dict[str, str]] = {
    AssetType.STOCKS: {"prefix": ""},
    AssetType.OPTIONS: {"prefix": "O:"},
    AssetType.FOREX: {"prefix": "C:"},
    AssetType.CRYPTO: {"prefix": "X:"},
}

DateLike = str | date | datetime | pd.Timestamp | None


class ChecksConfig(TypedDict):
    """Per-asset thresholds for the single-series `compare_to_yf` gate."""

    abs_rel_diff_pct_p50: float
    abs_rel_diff_pct_p99: float
    yf_max_missing_run_sessions: int  # max contiguous sessions yfinance may be missing (coverage)


class OptionsChecksConfig(TypedDict):
    """Thresholds for the options structural gate.

    The strict gates (positivity, no bars past expiry) have no threshold — any
    violation fails, and they apply to every bar regardless of moneyness.

    The no-arb bound checks — upper bounds (C ≤ S, P ≤ K) and intrinsic floors
    (C ≥ max(S-K, 0), P ≥ max(K-S, 0)) — are percentile-gated on the violation
    *relative to the underlying price*: illiquid ITM prints lag spot and routinely
    breach either bound by a few % without a real arb, and that noise scales with
    the price level, so a relative gate ports across assets where an absolute dollar
    gate wouldn't. A systematic mis-scaling (a whole contract class off by the split
    ratio) breaches a large fraction of bars and pushes p99 well past the band.

    Both values are percentages (like `ChecksConfig`'s `abs_rel_diff_pct_*`), expressed as
    the number itself: `1.0` means 1%, `50.0` means 50%.

    - `noarb_violation_pct_p99`: gate fires when the p99 of `100 × violation / underlying`
      (the breach/shortfall as a % of the underlying price) exceeds this. Default 1.0 (1%).
    - `deep_itm_intrinsic_pct`: a bar is "deep-ITM" when its intrinsic value is more than this
      % of the underlying price (`intrinsic > deep_itm_intrinsic_pct/100 × underlying`); e.g.
      50.0 → intrinsic > half the spot price. Deep-ITM bars are excluded from BOTH no-arb bound
      checks (not from positivity / expiry). Such contracts are stock proxies whose illiquid
      last-trade prints breach the bounds as a matter of course (sub-intrinsic bid-side fills,
      stale prints during a move); the violation rate climbs monotonically with moneyness
      (empirically ~1% near the money → ~34% at >80% ITM on NVDA's 2024 split window). The
      bounds are only informative for the liquid near-the-money / OTM contracts, and a
      systematic split error is still caught there because it hits all moneyness uniformly.
      Default 50.0.
    """

    noarb_violation_pct_p99: float
    deep_itm_intrinsic_pct: float


class OSIContract(NamedTuple):
    """Parsed OSI (Options Symbology Initiative) option ticker components.
    Underlying is the root symbol as emitted by Polygon (no yfinance normalization);
    strike is in dollars (raw 8-digit field divided by 1000).
    """

    underlying: str
    expiry: date
    option_type: Literal["C", "P"]
    strike: float


class Predecessor(NamedTuple):
    """A former ticker discovered by the rename auto-stitch (e.g. FB before FB→META). `symbol`
    is also the predecessor's OSI option root (`FB` ⇄ `O:FB…`), and `[start, end]` is the span
    over which we hold its bars. The options pass loads `O:<symbol>…` contracts bounded to this
    span (so a reused ticker can't leak in) and rewrites them to the live root for continuity.
    """

    symbol: str
    start: date
    end: date
