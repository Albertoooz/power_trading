# type: ignore
import argparse
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
import yfinance as yf
from pandas import DataFrame, Series
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands

# Lista instrumentów do analizy
INSTRUMENTS = [
    "AAPL",  # Apple
]


class TechnicalIndicators:
    @staticmethod
    def calculate_macd(
        close: Series, params: dict[str, Any]
    ) -> tuple[Series, Series, Series]:
        """Calculate MACD indicator."""
        try:
            macd_indicator = MACD(
                close=close,
                window_slow=params["macd_slow"],
                window_fast=params["macd_fast"],
                window_sign=params["macd_signal"],
            )
            macd = cast(Series, macd_indicator.macd())
            signal = cast(Series, macd_indicator.macd_signal())
            hist = cast(Series, macd_indicator.macd_diff())
            return macd, signal, hist
        except Exception as e:
            print(f"Error in calculate_macd: {str(e)}")
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    @staticmethod
    def calculate_rsi(close: Series, period: int) -> Series:
        """Calculate RSI indicator."""
        try:
            rsi_indicator = RSIIndicator(close=close, window=period)
            return cast(Series, rsi_indicator.rsi())
        except Exception as e:
            print(f"Error in calculate_rsi: {str(e)}")
            return pd.Series(dtype=float)

    @staticmethod
    def calculate_bollinger_bands(
        close: Series, period: int, std: float
    ) -> tuple[Series, Series, Series]:
        """Calculate Bollinger Bands."""
        try:
            bb_indicator = BollingerBands(close=close, window=period, window_dev=std)
            upper = cast(Series, bb_indicator.bollinger_hband())
            middle = cast(Series, bb_indicator.bollinger_mavg())
            lower = cast(Series, bb_indicator.bollinger_lband())
            return upper, middle, lower
        except Exception as e:
            print(f"Error in calculate_bollinger_bands: {str(e)}")
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    @staticmethod
    def calculate_stochastic(
        high: Series, low: Series, close: Series, period: int, smooth: int
    ) -> tuple[Series, Series]:
        """Calculate Stochastic Oscillator."""
        try:
            stoch_indicator = StochasticOscillator(
                high=high, low=low, close=close, window=period, smooth_window=smooth
            )
            k = cast(Series, stoch_indicator.stoch())
            d = cast(Series, stoch_indicator.stoch_signal())
            return k, d
        except Exception as e:
            print(f"Error in calculate_stochastic: {str(e)}")
            return pd.Series(dtype=float), pd.Series(dtype=float)

    @staticmethod
    def calculate_all(df: DataFrame, params: dict[str, Any]) -> DataFrame:
        """Calculate all technical indicators."""
        try:
            # Create a copy of the dataframe to avoid modifying the original
            df = df.copy()

            # Extract price data - we need to handle multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                # If we have multi-level columns, select the appropriate level
                close = (
                    df["Close"]["EURUSD=X"]
                    if ("Close", "EURUSD=X") in df.columns
                    else df["Close"].iloc[:, 0]
                )
                high = (
                    df["High"]["EURUSD=X"]
                    if ("High", "EURUSD=X") in df.columns
                    else df["High"].iloc[:, 0]
                )
                low = (
                    df["Low"]["EURUSD=X"]
                    if ("Low", "EURUSD=X") in df.columns
                    else df["Low"].iloc[:, 0]
                )
            else:
                # Single level columns
                close = df["Close"]
                high = df["High"]
                low = df["Low"]

            # Convert to float and handle NaN values
            close = pd.to_numeric(close, errors="coerce")
            high = pd.to_numeric(high, errors="coerce")
            low = pd.to_numeric(low, errors="coerce")

            # Forward fill then backward fill NaN values
            close = close.fillna(method="ffill").fillna(method="bfill")
            high = high.fillna(method="ffill").fillna(method="bfill")
            low = low.fillna(method="ffill").fillna(method="bfill")

            # Create result DataFrame with the same index
            result = pd.DataFrame(index=df.index)

            # Ensure we have enough data
            if len(close) < params["macd_slow"] + params["macd_signal"]:
                raise ValueError("Not enough data for indicator calculation")

            # MACD calculation
            macd_indicator = MACD(
                close=close,
                window_slow=params["macd_slow"],
                window_fast=params["macd_fast"],
                window_sign=params["macd_signal"],
            )
            result["macd"] = cast(Series, macd_indicator.macd())
            result["macd_signal"] = cast(Series, macd_indicator.macd_signal())
            result["macd_hist"] = cast(Series, macd_indicator.macd_diff())

            # RSI calculation
            rsi_indicator = RSIIndicator(close=close, window=params["rsi_period"])
            result["rsi"] = cast(Series, rsi_indicator.rsi())

            # Bollinger Bands calculation
            bb_indicator = BollingerBands(
                close=close, window=params["bb_period"], window_dev=params["bb_std"]
            )
            result["bb_high"] = cast(Series, bb_indicator.bollinger_hband())
            result["bb_mid"] = cast(Series, bb_indicator.bollinger_mavg())
            result["bb_low"] = cast(Series, bb_indicator.bollinger_lband())

            # Stochastic Oscillator calculation
            stoch_indicator = StochasticOscillator(
                high=high,
                low=low,
                close=close,
                window=params["stoch_period"],
                smooth_window=params["stoch_smooth"],
            )
            result["stoch_k"] = cast(Series, stoch_indicator.stoch())
            result["stoch_d"] = cast(Series, stoch_indicator.stoch_signal())

            # Add price data
            result["Close"] = close
            result["High"] = high
            result["Low"] = low

            # Handle any remaining NaN values
            result = result.fillna(method="ffill").fillna(method="bfill")

            return result

        except Exception as e:
            print(f"Error in calculate_all: {str(e)}")
            print(f"Data types: {df.dtypes}")
            print(f"Data head:\n{df.head()}")
            return pd.DataFrame()


class Strategy:
    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.indicators = TechnicalIndicators()

    def generate_signals(self, indicators: DataFrame) -> Series:
        """Generate trading signals based on technical indicators."""
        try:
            if indicators.empty:
                return pd.Series(dtype=float)

            # Initialize signals
            signals = pd.Series(0.0, index=indicators.index, dtype=float)

            # MACD signals (weight: 1.0)
            macd_hist = indicators["macd_hist"].astype(float)
            macd_signals = pd.Series(0.0, index=indicators.index, dtype=float)
            for i in range(len(macd_signals)):
                if i >= 20:  # Need at least 20 periods for normalization
                    max_hist = max(abs(macd_hist[i - 20 : i + 1]))  # Use last 20 periods
                    if max_hist > 0:
                        macd_signals[i] = macd_hist[i] / max_hist

            # RSI signals (weight: 1.0)
            rsi = indicators["rsi"].astype(float)
            rsi_signals = pd.Series(0.0, index=indicators.index, dtype=float)
            rsi_signals[rsi < self.params["rsi_oversold"]] = 1.0  # Oversold -> Buy
            rsi_signals[rsi > self.params["rsi_overbought"]] = -1.0  # Overbought -> Sell

            # Combine signals
            signals = macd_signals + rsi_signals

            # Normalize combined signals
            max_signal = max(abs(signals))
            if max_signal > 0:
                signals = signals / max_signal

            return signals

        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return pd.Series(dtype=float)

    def backtest(self, data: DataFrame) -> float:
        """Run backtest and calculate Sharpe ratio."""
        try:
            indicators = self.indicators.calculate_all(data, self.params)
            signals = self.generate_signals(indicators)

            # Calculate returns
            returns = data["Close"].pct_change()
            strategy_returns = signals.shift(1) * returns

            # Calculate Sharpe Ratio
            if len(strategy_returns) > 0:
                sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
                return float(sharpe)
            return 0.0

        except Exception as e:
            print(f"Error in backtest: {str(e)}")
            return 0.0


def optimize_strategy(
    symbol: str, start_date: str, end_date: str, n_trials: int = 100
) -> dict[str, Any]:
    """Optimize strategy parameters using Optuna."""
    print(f"\nOptimizing strategy for {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Number of trials: {n_trials}")

    # Download data
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        print(f"No data available for {symbol}")
        return {
            "symbol": symbol,
            "best_params": {},
            "best_value": 0,
            "error": "No data available",
        }

    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "macd_fast": trial.suggest_int("macd_fast", 8, 20),
            "macd_slow": trial.suggest_int("macd_slow", 21, 40),
            "macd_signal": trial.suggest_int("macd_signal", 5, 15),
            "rsi_period": trial.suggest_int("rsi_period", 5, 25),
            "rsi_overbought": trial.suggest_float("rsi_overbought", 65, 85),
            "rsi_oversold": trial.suggest_float("rsi_oversold", 15, 35),
            "bb_period": trial.suggest_int("bb_period", 10, 50),
            "bb_std": trial.suggest_float("bb_std", 1.5, 3.0),
            "stoch_period": trial.suggest_int("stoch_period", 5, 25),
            "stoch_smooth": trial.suggest_int("stoch_smooth", 2, 10),
            "stoch_overbought": trial.suggest_float("stoch_overbought", 75, 85),
            "stoch_oversold": trial.suggest_float("stoch_oversold", 15, 25),
        }

        strategy = Strategy(params)
        return strategy.backtest(data)

    try:
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_value = study.best_value

        print("\n=== Best Parameters ===")
        print(f"Best Sharpe Ratio: {best_value:.4f}")
        print("\nOptimal Parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")

        best_params["symbol"] = symbol
        best_params["start_date"] = start_date
        best_params["end_date"] = end_date
        best_params["best_value"] = best_value

        return best_params
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        return {"symbol": symbol, "best_params": {}, "best_value": 0, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize trading strategy parameters")
    parser.add_argument("--symbol", default="AAPL", help="Trading symbol")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default="2023-01-01", help="End date")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")

    args = parser.parse_args()

    best_params = optimize_strategy(args.symbol, args.start, args.end, args.trials)
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")


if __name__ == "__main__":
    main()


def calculate_indicators(data: DataFrame, params: dict) -> DataFrame:
    """Calculate technical indicators for the strategy.

    Args:
        data: DataFrame with OHLCV data
        params: Dictionary with indicator parameters

    Returns:
        DataFrame with calculated indicators
    """
    # Calculate MACD
    macd = MACD(
        close=data["Close"],
        window_slow=params["macd_slow"],
        window_fast=params["macd_fast"],
        window_sign=params["macd_signal"],
    )

    # Calculate RSI
    rsi = RSIIndicator(
        close=data["Close"],
        window=params["rsi_period"],
    )

    # Calculate Bollinger Bands
    bb = BollingerBands(
        close=data["Close"],
        window=params["bb_period"],
        window_dev=params["bb_std"],
    )

    # Calculate Stochastic
    stoch = StochasticOscillator(
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        window=params["stoch_period"],
        smooth_window=params["stoch_smooth"],
    )

    # Combine all indicators
    data = data.copy()
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_hist"] = macd.macd_diff()
    data["rsi"] = rsi.rsi()
    data["bb_upper"] = bb.bollinger_hband()
    data["bb_lower"] = bb.bollinger_lband()
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()

    return data
