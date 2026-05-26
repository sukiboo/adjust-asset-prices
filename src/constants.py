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

# Options-internal machinery (OSI symbology + split-unification), used by the OSI parse/format
# helpers in utils.py and the split-unifier in prices/options.py. Not user-facing knobs — these
# encode the OSI/OCC standard and empirical matching tolerances; change only if you know the spec.
OPTIONS_INTERNALS = {
    "strike_scale": 1000,  # OSI encodes strikes as an integer count of milli-dollars (1/1000 $)
    "integer_tol": 1e-6,  # float slack for "is this a whole number" (clean strike / split ratio)
    "min_split_factor": 2,  # only x:1 and 1:x with integer x >= 2 splits are handled
    "successor_strike_tol": 0.01,  # 1¢ max strike gap to match a non-clean contract's successor
}

SHOW_PLOT = True
