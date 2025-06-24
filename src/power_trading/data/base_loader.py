from abc import ABC, abstractmethod
import pandas as pd


class BaseDataLoader(ABC):
    @abstractmethod
    def load_data(
        self, ticker: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """Load historical market data"""
        pass
