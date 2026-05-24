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
    "noarb_violation_p99_rel": 0.05,
    "deep_itm_moneyness_cap": 0.5,
}
SHOW_PLOT = True
