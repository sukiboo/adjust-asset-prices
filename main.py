import argparse
import sys
from typing import get_args

from src import Prices, save_if_valid, save_options_if_valid
from src.constants import (
    CHECKS_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_DATE_END,
    DEFAULT_DATE_START,
    DEFAULT_FORMAT,
    DEFAULT_SAVE_DIR,
    OPTIONS_CHECKS_CONFIG,
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
        help="Also load + backfill + split-unify + structural-gate + save all option "
        "contracts on the underlying, after the underlying's stocks pass succeeds. "
        "Aborts if either pass's checks fail.",
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _print_options_summary(label: str, df) -> None:
    if df.empty:
        print(f"   {label}: 0 bars / 0 contracts")
        return
    ts = df.index.get_level_values("timestamp_utc")
    n_contracts = df.index.get_level_values("ticker").nunique()
    print(f"   {label}: {len(df):,} bars / {n_contracts:,} contracts " f"({ts.min()} → {ts.max()})")


if __name__ == "__main__":
    args = parse_args()

    prices = Prices(data_dir=args.data_dir, debug=args.debug)
    intrinsic_ref = None  # set in the args.options branch below if --options

    if args.options:
        # Options need the split-adjusted-only underlying as the intrinsic-floor
        # reference (cash dividends are priced into option premiums, not back-adjusted
        # out — using the dividend-adjusted series would overstate intrinsic on
        # pre-dividend bars and trigger false alarms). Decompose `get_prices` so we
        # can capture the split-only intermediate before dividends are applied.
        raw_df, asset_type = prices.load_prices(
            ticker=args.ticker, date_start=args.date_start, date_end=args.date_end
        )
        backfilled = prices.backfill_prices(
            raw_df, asset_type, date_start=args.date_start, date_end=args.date_end
        )
        intrinsic_ref = prices.adjust_splits(backfilled, asset_type)
        df = prices.adjust_dividends(intrinsic_ref, asset_type)
    else:
        df, asset_type = prices.get_prices(
            ticker=args.ticker, date_start=args.date_start, date_end=args.date_end
        )

    stocks_ok = save_if_valid(
        df,
        save_dir=args.save_dir,
        format=args.format,
        config=CHECKS_CONFIG,
        asset_type=asset_type,
        show_plot=SHOW_PLOT,
    )
    if args.options and not stocks_ok:
        print(f"❌ {args.ticker} stock price failed verification -- aborting options pass!")
        sys.exit(1)

    if args.options:
        assert intrinsic_ref is not None  # set above when args.options is True
        calls, puts = prices.load_options(
            underlying=args.ticker, date_start=args.date_start, date_end=args.date_end
        )
        calls, puts = prices.adjust_options_splits(calls, puts, args.ticker)
        calls, puts = prices.backfill_options(calls, puts)
        print(f"\n🗃️  Option contracts for {args.ticker}:")
        _print_options_summary("calls", calls)
        _print_options_summary("puts", puts)
        options_ok = save_options_if_valid(
            calls,
            puts,
            underlying=args.ticker,
            underlying_df=intrinsic_ref,
            save_dir=args.save_dir,
            format=args.format,
            config=OPTIONS_CHECKS_CONFIG,
        )
        if not options_ok:
            sys.exit(1)
