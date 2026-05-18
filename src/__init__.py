import pandas as pd

from .checks import check_prices  # noqa: F401
from .prices import Prices  # noqa: F401
from .utils import (  # noqa: F401
    load_prices,
    save_if_valid,
    save_prices,
    verify_saved_prices,
)

pd.set_option("display.float_format", "{:.2f}".format)
