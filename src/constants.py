from .schemas import AssetType, ChecksConfig, OptionsChecksConfig, PriceFileFormat

DEFAULT_FORMAT: PriceFileFormat = "parquet"
DEFAULT_DATE_START: str | None = None
DEFAULT_DATE_END: str | None = None
DEFAULT_DATA_DIR = "./data/files"
DEFAULT_SAVE_DIR = "./data/prices"
DEFAULT_SHOW_PLOT = False

# Thread pool size for the per-file raw-data reads (load_symbol_rows / load_options_data).
# pyarrow's CSV reader releases the GIL during decompress+parse, so threads give real
# parallelism here (~3.5x on the read-bound load); empirically saturates around 8 workers.
READ_MAX_WORKERS = 8

# Drop bars before this on load (all assets) — corruption guard: some raw files store window_start
# ~1e6 too small (a ~1970 unit bug) which backfill would otherwise fan out as phantom rows.
MIN_PLAUSIBLE_DATE = "2000-01-01"

# Where users can get the raw daily price files when they have none locally (pre-built daily
# files for all asset types through 2026). Surfaced by `check_data_dir` in the no-data error.
DATA_SOURCE_URL = "https://www.dropbox.com/scl/fo/xd5a5s5cwa0imf6gvplzv/AL1ffzRw3_AEfeEwRoKLQms?rlkey=ah6c8ps5zvco29npoeoro831k&dl=0"

# Per-asset thresholds for the single-series compare-to-yf gate (stocks/crypto/forex), tuned per
# asset to absorb upstream yfinance noise (stocks tight, crypto/forex looser). `abs_rel_diff_pct_*`
# gate the |our − yf| daily-close % (median / p99). `yf_max_missing_run_sessions` is the coverage
# guard: fail if our series has a contiguous run longer than this of sessions yfinance has no data
# for — an uncorroborated span (e.g. a wrong stitch). Small scattered gaps stay well under it.
CHECKS_CONFIG: dict[AssetType, ChecksConfig] = {
    AssetType.STOCKS: {
        "abs_rel_diff_pct_p50": 0.05,
        "abs_rel_diff_pct_p99": 0.5,
        "yf_max_missing_run_sessions": 10,
    },
    AssetType.CRYPTO: {
        "abs_rel_diff_pct_p50": 0.1,
        "abs_rel_diff_pct_p99": 1.5,
        "yf_max_missing_run_sessions": 10,
    },
    AssetType.FOREX: {
        "abs_rel_diff_pct_p50": 0.05,
        "abs_rel_diff_pct_p99": 0.5,
        "yf_max_missing_run_sessions": 10,
    },
}

# Thresholds for the options structural (no-arb) gate. Both are percentages.
OPTIONS_CHECKS_CONFIG: OptionsChecksConfig = {
    "noarb_violation_pct_p99": 1.0,  # gate fails when p99 no-arb breach (% of spot) exceeds this
    "deep_itm_intrinsic_pct": 50.0,  # exclude bars with intrinsic > this% of spot from the bounds
}

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
    "save_batch_rows": 10_000_000,  # max rows/batch when streaming backfill to disk to avoid OOM
}

# Retry policy for yfinance fetches (transient Yahoo failures: an exception, or — for `.history()`
# — a logged error plus empty result). Linear backoff: sleep YF_RETRY_BACKOFF * attempt seconds.
YF_MAX_RETRIES = 4
YF_RETRY_BACKOFF = 2.0
