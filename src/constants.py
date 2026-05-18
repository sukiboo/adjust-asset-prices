from .schemas import ChecksConfig

CHECKS_CONFIG: ChecksConfig = {
    "gap_threshold_mins": 1,
    "num_gaps_display": 10,
    "abs_rel_diff_pct_p50": 0.1,
    "abs_rel_diff_pct_p99": 1.0,
    "show_plot": True,
}
