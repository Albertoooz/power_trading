from typing import cast

import pandas as pd
import yfinance as yf
from pandas import DataFrame

from .base_loader import BaseDataLoader


class YahooFinanceLoader(BaseDataLoader):
    """Loader for Yahoo Finance data."""

    def load_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> DataFrame:
        """Load data from Yahoo Finance.

        Args:
            symbol: The ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (default: "1d")

        Returns:
            DataFrame with OHLCV data
        """
        data = cast(
            DataFrame,
            yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
            ),
        )

        if len(data) == 0:
            return pd.DataFrame()

        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-level columns
            data.columns = [f"{symbol}_{col[0]}" for col in data.columns]

            # Create mapping for renaming
            rename_map = {
                f"{symbol}_Open": "Open",
                f"{symbol}_High": "High",
                f"{symbol}_Low": "Low",
                f"{symbol}_Close": "Close",
                f"{symbol}_Volume": "Volume",
            }

            # Print debug info
            print("\n=== ORIGINAL DATA ===")
            print(f"Columns: {data.columns}")
            print("Sample:\n", data.head(2))

            # Rename columns
            data = data.rename(columns=rename_map)

            # Print debug info
            print("\nDiscovered mapping:", rename_map)

            print("\n=== FINAL DATA ===")
            print(f"Columns: {data.columns}")
            print("Sample:\n", data.head(2))

        return data
