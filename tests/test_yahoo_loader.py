from power_trading.data.yahoo_loader import YahooFinanceLoader
import pytest


def test_yahoo_loader_success():
    loader = YahooFinanceLoader()
    df = loader.load_data("AAPL", "2023-01-01", "2023-01-15")
    assert not df.empty


def test_yahoo_loader_invalid():
    loader = YahooFinanceLoader()
    with pytest.raises(ValueError):
        loader.load_data("INVALID123", "2023-01-01", "2023-01-15")
