import argparse
from typing import get_args

from src import Prices, save_if_valid
from src.constants import (
    CHECKS_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_DATE_END,
    DEFAULT_DATE_START,
    DEFAULT_FORMAT,
    DEFAULT_SAVE_DIR,
    SHOW_PLOT,
)
from src.schemas import PriceFileFormat
from src.utils import parsable_date


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adjust raw asset prices for a single ticker.")
    p.add_argument("ticker", type=str.upper, help="Ticker symbol, e.g. BTC-USD")
    p.add_argument("--format", choices=list(get_args(PriceFileFormat)), default=DEFAULT_FORMAT)
    p.add_argument("--date-start", type=parsable_date, default=DEFAULT_DATE_START)
    p.add_argument("--date-end", type=parsable_date, default=DEFAULT_DATE_END)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    p.add_argument(
        "--options",
        action="store_true",
        help="Load all option contracts on the underlying instead of the underlying itself "
        "(load-only for now — backfill / structural gate / save not yet implemented)",
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _print_options_summary(label: str, df) -> None:
    if df.empty:
        print(f"  {label}: 0 bars / 0 contracts")
        return
    ts = df.index.get_level_values("timestamp_utc")
    n_contracts = df.index.get_level_values("ticker").nunique()
    print(f"  {label}: {len(df):,} bars / {n_contracts:,} contracts " f"({ts.min()} → {ts.max()})")


if __name__ == "__main__":
    args = parse_args()

    prices = Prices(data_dir=args.data_dir, debug=args.debug)

    if args.options:
        calls, puts = prices.load_options(
            underlying=args.ticker, date_start=args.date_start, date_end=args.date_end
        )
        calls, puts = prices.backfill_options(calls, puts)
        calls, puts = prices.adjust_options_splits(calls, puts, args.ticker)
        print(f"📊 Options for {args.ticker}:")
        _print_options_summary("calls", calls)
        _print_options_summary("puts", puts)
        print(
            "(load + backfill + split-unification only — "
            "structural gate / save not yet implemented)"
        )
    else:
        df, asset_type = prices.get_prices(
            ticker=args.ticker, date_start=args.date_start, date_end=args.date_end
        )
        save_if_valid(
            df,
            save_dir=args.save_dir,
            format=args.format,
            config=CHECKS_CONFIG,
            asset_type=asset_type,
            show_plot=SHOW_PLOT,
        )
