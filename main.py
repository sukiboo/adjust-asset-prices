import argparse

from src import Prices, save_if_valid
from src.constants import CHECKS_CONFIG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adjust raw asset prices for a single ticker.")
    p.add_argument("ticker", help="Ticker symbol, e.g. BTC-USD")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    p.add_argument("--date-start", default=None)
    p.add_argument("--date-end", default=None)
    p.add_argument("--data-dir", default="./data/files")
    p.add_argument("--save-dir", default="./data/prices")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prices = Prices(data_dir=args.data_dir, debug=args.debug)
    df = prices.get_prices(ticker=args.ticker, date_start=args.date_start, date_end=args.date_end)
    save_if_valid(df, save_dir=args.save_dir, format=args.format, config=CHECKS_CONFIG)
