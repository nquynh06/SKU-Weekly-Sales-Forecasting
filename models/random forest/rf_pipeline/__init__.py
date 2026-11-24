"""Quality-gated, one-week-ahead SKU sales quantity forecasting."""

__version__ = "1.0.0"
CONTRACT_VERSION = "quantity_rf_v1"

from .config import Config
from .data import DataContractError, load_weekly_sales
from .features import build_features

__all__ = ["Config", "DataContractError", "load_weekly_sales", "build_features"]


def main():
    from .cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
