from pandas import DataFrame

from power_trading.data.yahoo_loader import YahooFinanceLoader


def test_load_data_success() -> None:
    """Test successful data loading."""
    loader = YahooFinanceLoader()
    data = loader.load_data("AAPL", "2023-01-01", "2023-01-31")
    assert isinstance(data, DataFrame)
    assert not data.empty


def test_load_data_invalid_ticker() -> None:
    """Test loading data for invalid ticker."""
    loader = YahooFinanceLoader()
    data = loader.load_data("INVALID_TICKER", "2023-01-01", "2023-01-31")
    assert isinstance(data, DataFrame)
    assert data.empty
