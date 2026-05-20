from .schemas import ChecksConfig, PriceFileFormat

DEFAULT_FORMAT: PriceFileFormat = "parquet"
DEFAULT_DATE_START: str | None = None
DEFAULT_DATE_END: str | None = None
DEFAULT_DATA_DIR = "./data/files"
DEFAULT_SAVE_DIR = "./data/prices"

CHECKS_CONFIG: ChecksConfig = {
    "abs_rel_diff_pct_p50": 0.1,
    "abs_rel_diff_pct_p99": 1.0,
    "show_plot": True,
}
