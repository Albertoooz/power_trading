import yfinance as yf
import pandas as pd
from .base_loader import BaseDataLoader


class YahooFinanceLoader(BaseDataLoader):
    def load_data(
        self, ticker: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """Load and properly format data for Backtrader with uppercase columns."""
        # Download data with column organization
        data = yf.download(ticker, start=start, end=end, interval=interval, group_by='column')
        
        if data.empty:
            raise ValueError(f"No data found for {ticker}")

        # Debug original structure
        print("\n=== ORIGINAL DATA ===")
        print("Columns:", data.columns)
        print("Sample:\n", data.head(2))

        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten to 'Close_AAPL' format
            data.columns = [f"{col[1]}_{col[0]}" for col in data.columns]
            print("\nFlattened columns:", data.columns)

        # Find and map columns
        col_mapping = {}
        for standard_col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            # Find matching columns (case insensitive)
            matches = [col for col in data.columns if standard_col.lower() in col.lower()]
            if matches:
                col_mapping[matches[0]] = standard_col

        print("\nDiscovered mapping:", col_mapping)

        # Verify we found all required columns
        required = ['Close', 'High', 'Low', 'Open', 'Volume']
        if len(col_mapping) < len(required):
            missing = set(required) - set(col_mapping.values())
            raise ValueError(
                f"Missing columns: {missing}\n"
                f"Available columns: {list(data.columns)}\n"
                f"Mapping: {col_mapping}"
            )

        # Apply mapping and select columns
        data = data.rename(columns=col_mapping)
        data = data[required]  # Enforce column order

        # Final processing
        data.index = pd.to_datetime(data.index)
        data.dropna(inplace=True)

        print("\n=== FINAL DATA ===")
        print("Columns:", data.columns)
        print("Sample:\n", data.head(2))

        return data