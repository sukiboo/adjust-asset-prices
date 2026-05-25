from .schemas import AssetType, ChecksConfig, OptionsChecksConfig, PriceFileFormat

DEFAULT_FORMAT: PriceFileFormat = "parquet"
DEFAULT_DATE_START: str | None = None
DEFAULT_DATE_END: str | None = None
DEFAULT_DATA_DIR = "./data/files"
DEFAULT_SAVE_DIR = "./data/prices"

CHECKS_CONFIG: dict[AssetType, ChecksConfig] = {
    AssetType.STOCKS: {"abs_rel_diff_pct_p50": 0.05, "abs_rel_diff_pct_p99": 0.5},
    AssetType.CRYPTO: {"abs_rel_diff_pct_p50": 0.1, "abs_rel_diff_pct_p99": 1.5},
    AssetType.FOREX: {"abs_rel_diff_pct_p50": 0.05, "abs_rel_diff_pct_p99": 0.5},
}
OPTIONS_CHECKS_CONFIG: OptionsChecksConfig = {
    "noarb_violation_pct_p99": 1.0,  # gate fails when p99 no-arb breach (% of spot) exceeds this
    "deep_itm_intrinsic_pct": 50.0,  # exclude bars with intrinsic > this% of spot from the bounds
}

# OSI option-symbol encoding + split-unification tuning (used by the OSI parse/format helpers
# in utils.py and the options split-unifier in prices/options.py).
# OSI encodes strikes as an integer count of milli-dollars (1/1000 $).
OSI_STRIKE_SCALE = 1000
# Float slack for "is this a whole number": clean milli-strike and integer split ratio.
INTEGER_TOLERANCE = 1e-6
# Only x:1 / 1:x splits with integer x >= 2 are handled (a 1:1 isn't a split).
MIN_SPLIT_FACTOR = 2
# 1¢: max strike gap to match a spanning non-clean contract to its OCC suffixed successor.
OPTIONS_SUCCESSOR_STRIKE_TOL = 0.01

SHOW_PLOT = True
