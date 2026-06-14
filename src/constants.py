from .schemas import AssetType, ChecksConfig, OptionsChecksConfig, PriceFileFormat

DEFAULT_FORMAT: PriceFileFormat = "parquet"
DEFAULT_DATE_START: str | None = None
DEFAULT_DATE_END: str | None = None
DEFAULT_DATA_DIR = "./data/files"
DEFAULT_SAVE_DIR = "./data/prices"
DEFAULT_SHOW_PLOT = False

# Where users can get the raw daily price files when they have none locally (pre-built daily
# files for all asset types through 2026). Surfaced by `check_data_dir` in the no-data error.
DATA_SOURCE_URL = "https://www.dropbox.com/scl/fo/xd5a5s5cwa0imf6gvplzv/AL1ffzRw3_AEfeEwRoKLQms?rlkey=ah6c8ps5zvco29npoeoro831k&dl=0"

CHECKS_CONFIG: dict[AssetType, ChecksConfig] = {
    AssetType.STOCKS: {"abs_rel_diff_pct_p50": 0.05, "abs_rel_diff_pct_p99": 0.5},
    AssetType.CRYPTO: {"abs_rel_diff_pct_p50": 0.1, "abs_rel_diff_pct_p99": 1.5},
    AssetType.FOREX: {"abs_rel_diff_pct_p50": 0.05, "abs_rel_diff_pct_p99": 0.5},
}
OPTIONS_CHECKS_CONFIG: OptionsChecksConfig = {
    "noarb_violation_pct_p99": 1.0,  # gate fails when p99 no-arb breach (% of spot) exceeds this
    "deep_itm_intrinsic_pct": 50.0,  # exclude bars with intrinsic > this% of spot from the bounds
}

# compare_to_yf coverage guard: fail if our series has a contiguous run longer than this of trading
# sessions that yfinance has no data for — that span is uncorroborated (a wrong stitch or ticker).
# Small scattered gaps (yfinance hiccups, weekend forex) are well under this and pass.
YF_MAX_MISSING_RUN_SESSIONS = 10

# Ticker-rename auto-stitch tuning (stocks only). A long gap in a live ticker's raw history
# (interior, or leading vs date_start) means it likely traded under a former symbol — e.g. QQQ
# was QQQQ 2004-2011 — which is recovered from our own files. Not user-facing; see aliases.py.
ALIAS_INTERNALS = {
    "min_gap_trading_days": 30,  # raw gap (NYSE sessions) that triggers a predecessor search
    "boundary_slop_sessions": 3,  # post-resume sessions to skip (rename overlap) before stops check
    "survivor_window_sessions": 126,  # post-resume sessions scanned for "still trading" (~6 months)
    "liquidity_window_sessions": 5,  # sessions each side of the rename for the median bar-count check
    "liquidity_frac": 0.5,  # min predecessor median bars/day vs the requested series' (filters thin)
    "splice_sanity_pct": 10.0,  # max plausible % price jump across a rename splice; else no match
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
