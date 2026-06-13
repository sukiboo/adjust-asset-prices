from collections.abc import Callable

from ..checks import save_if_valid, save_options_if_valid
from ..constants import (
    CHECKS_CONFIG,
    DEFAULT_FORMAT,
    DEFAULT_SAVE_DIR,
    DEFAULT_SHOW_PLOT,
    OPTIONS_CHECKS_CONFIG,
)
from ..schemas import AssetType, PriceFileFormat
from .assets import AssetPrices
from .options import OptionsPrices


class Prices:
    """Facade composing `self.asset` (stocks/crypto/forex) and `self.options`, both sharing
    one `data_dir`, plus `process` — the end-to-end retrieve → check → save pipeline."""

    def __init__(self, data_dir: str) -> None:
        self.asset = AssetPrices(data_dir)
        self.options = OptionsPrices(self.asset)

    def process(
        self,
        ticker: str,
        date_start: str | None = None,
        date_end: str | None = None,
        dividends: bool = False,
        options: bool = False,
        save_dir: str = DEFAULT_SAVE_DIR,
        format: PriceFileFormat = DEFAULT_FORMAT,
        show_plot: bool = DEFAULT_SHOW_PLOT,
        confirm_on_fail: Callable[[], bool] | None = None,
    ) -> bool:
        """Retrieve → check → save `ticker`, and (when `options`) its option contracts.
        The asset series is always retrieved via the same `get_prices` path — split-only when
        `options` (which is mutually exclusive with `dividends`), so it aligns with the
        contracts and doubles as the structural gate's reference. `confirm_on_fail` is the CLI's
        interactive override: when the asset checks fail it's invoked and the data saved anyway
        iff it returns True (default `None` → strict, never save on fail). Returns True iff
        everything checked (or was overridden) and saved/verified.
        """
        df, asset_type = self.asset.get_prices(ticker, date_start, date_end, dividends=dividends)
        if options:
            assert (
                asset_type == AssetType.STOCKS
            ), f"❌ options require a stock underlying, got {asset_type}"

        saved = save_if_valid(
            df,
            save_dir=save_dir,
            format=format,
            config=CHECKS_CONFIG,
            asset_type=asset_type,
            show_plot=show_plot,
            dividends_adjusted=dividends,
            confirm_on_fail=confirm_on_fail,
        )
        if options and saved:
            calls, puts = self.options.get_options(ticker, date_start, date_end)
            saved = save_options_if_valid(
                calls,
                puts,
                underlying=ticker,
                underlying_df=df,
                save_dir=save_dir,
                format=format,
                config=OPTIONS_CHECKS_CONFIG,
            )
        return saved
