import pandas as pd

from .checks import check_prices, save_if_valid  # noqa: F401
from .prices import Prices  # noqa: F401
from .utils import load_prices, save_prices, verify_saved_prices  # noqa: F401

pd.set_option("display.float_format", "{:.2f}".format)
