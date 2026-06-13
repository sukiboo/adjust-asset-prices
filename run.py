import argparse
import sys
from typing import get_args

from src import Prices
from src.constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_DATE_END,
    DEFAULT_DATE_START,
    DEFAULT_FORMAT,
    DEFAULT_SAVE_DIR,
    DEFAULT_SHOW_PLOT,
)
from src.schemas import PriceFileFormat
from src.utils import parsable_date, save_on_fail_decider


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adjust raw asset prices for a single ticker.")
    p.add_argument("ticker", type=str.upper, help="Ticker symbol, e.g. BTC-USD")
    p.add_argument("--format", choices=list(get_args(PriceFileFormat)), default=DEFAULT_FORMAT)
    p.add_argument("--date-start", type=parsable_date, default=DEFAULT_DATE_START)
    p.add_argument("--date-end", type=parsable_date, default=DEFAULT_DATE_END)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    p.add_argument(
        "--plot",
        action="store_true",
        default=DEFAULT_SHOW_PLOT,
        help="Display the price-comparison plot against yfinance. Off by default.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Save the data even if the checks fail, skipping the interactive prompt. In an "
        "interactive terminal you'd otherwise be asked [y/N]; --force is also the only way to "
        "override in a non-interactive run (script/CI).",
    )
    # Mutually exclusive: a dividend-adjusted underlying can't align with the (never
    # dividend-adjusted) options, so passing both errors at parse time, before any retrieval.
    adjust = p.add_mutually_exclusive_group()
    adjust.add_argument(
        "--dividends",
        action="store_true",
        help="Back-adjust the stock series for cash dividends (Adj-Close-style total return). "
        "Off by default — the output is the actual split-adjusted traded price. Mutually "
        "exclusive with --options.",
    )
    adjust.add_argument(
        "--options",
        action="store_true",
        help="Also load + backfill + split-unify + structural-gate + save all option "
        "contracts on the underlying (the underlying is saved split-only, aligned with the "
        "options), after the underlying's stocks pass succeeds. Aborts if either pass fails. "
        "Mutually exclusive with --dividends.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ok = Prices(data_dir=args.data_dir).process(
        ticker=args.ticker,
        date_start=args.date_start,
        date_end=args.date_end,
        dividends=args.dividends,
        options=args.options,
        save_dir=args.save_dir,
        format=args.format,
        show_plot=args.plot,
        confirm_on_fail=save_on_fail_decider(args.force),
    )
    sys.exit(0 if ok else 1)
