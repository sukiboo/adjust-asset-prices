from src import Prices

if __name__ == "__main__":
    prices = Prices(data_dir="./data/files")
    df = prices.get_prices(ticker="BTC-USD", date_start="2025-01-01", date_end=None)
