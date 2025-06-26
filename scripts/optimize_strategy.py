import argparse

import numpy as np
import optuna
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator  # type: ignore
from ta.trend import MACD  # type: ignore
from ta.volatility import BollingerBands  # type: ignore

# Lista instrumentów do analizy
INSTRUMENTS = [
    "AAPL",  # Apple
]


class TechnicalIndicators:
    @staticmethod
    def calculate_macd(
        close: pd.Series, params: dict
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        try:
            macd_indicator = MACD(
                close=close,
                window_slow=params["macd_slow"],
                window_fast=params["macd_fast"],
                window_sign=params["macd_signal"],
            )
            macd = macd_indicator.macd()
            signal = macd_indicator.macd_signal()
            hist = macd_indicator.macd_diff()
            return macd, signal, hist
        except Exception as e:
            print(f"Error in calculate_macd: {str(e)}")
            return pd.Series(), pd.Series(), pd.Series()

    @staticmethod
    def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            rsi_indicator = RSIIndicator(close=close, window=period)
            return rsi_indicator.rsi()
        except Exception as e:
            print(f"Error in calculate_rsi: {str(e)}")
            return pd.Series()

    @staticmethod
    def calculate_bollinger_bands(
        close: pd.Series, period: int, std: float
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            bb_indicator = BollingerBands(close=close, window=period, window_dev=std)
            upper = bb_indicator.bollinger_hband()
            middle = bb_indicator.bollinger_mavg()
            lower = bb_indicator.bollinger_lband()
            return upper, middle, lower
        except Exception as e:
            print(f"Error in calculate_bollinger_bands: {str(e)}")
            return pd.Series(), pd.Series(), pd.Series()

    @staticmethod
    def calculate_stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int, smooth: int
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        try:
            stoch_indicator = StochasticOscillator(
                high=high, low=low, close=close, window=period, smooth_window=smooth
            )
            k = stoch_indicator.stoch()
            d = stoch_indicator.stoch_signal()
            return k, d
        except Exception as e:
            print(f"Error in calculate_stochastic: {str(e)}")
            return pd.Series(), pd.Series()

    @staticmethod
    def calculate_all(df: pd.DataFrame, params: dict) -> pd.DataFrame:
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
            result["macd"] = macd_indicator.macd()
            result["macd_signal"] = macd_indicator.macd_signal()
            result["macd_hist"] = macd_indicator.macd_diff()

            # RSI calculation
            rsi_indicator = RSIIndicator(close=close, window=params["rsi_period"])
            result["rsi"] = rsi_indicator.rsi()

            # Bollinger Bands calculation
            bb_indicator = BollingerBands(
                close=close, window=params["bb_period"], window_dev=params["bb_std"]
            )
            result["bb_high"] = bb_indicator.bollinger_hband()
            result["bb_mid"] = bb_indicator.bollinger_mavg()
            result["bb_low"] = bb_indicator.bollinger_lband()

            # Stochastic Oscillator calculation
            stoch_indicator = StochasticOscillator(
                high=high,
                low=low,
                close=close,
                window=params["stoch_period"],
                smooth_window=params["stoch_smooth"],
            )
            result["stoch_k"] = stoch_indicator.stoch()
            result["stoch_d"] = stoch_indicator.stoch_signal()

            # Add price data
            result["Close"] = close
            result["High"] = high
            result["Low"] = low

            # Handle any remaining NaN values
            result = result.fillna(method="ffill").fillna(method="bfill")

            # Print debug information
            print("\nIndicator Statistics:")
            print(f"Original data length: {len(df)}")
            print(f"Valid data length: {len(result)}")
            print(f"MACD range: {result['macd'].min():.2f} to {result['macd'].max():.2f}")
            print(f"RSI range: {result['rsi'].min():.2f} to {result['rsi'].max():.2f}")
            print(
                f"Stochastic range: "
                f"{result['stoch_k'].min():.2f} to {result['stoch_k'].max():.2f}"
            )

            return result

        except Exception as e:
            print(f"Error in calculate_all: {str(e)}")
            print(f"Data types: {df.dtypes}")
            print(f"Data head:\n{df.head()}")
            return pd.DataFrame()


class Strategy:
    def __init__(self, params: dict):
        self.params = params
        self.indicators = TechnicalIndicators()

    def generate_signals(self, indicators: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on technical indicators."""
        try:
            if indicators.empty:
                return pd.Series()

            # Initialize signals
            signals = pd.Series(0, index=indicators.index)

            # MACD signals (weight: 1.0)
            macd_hist = indicators["macd_hist"]
            macd_signals = pd.Series(0.0, index=indicators.index)
            for i in range(len(macd_signals)):
                if i >= 20:  # Need at least 20 periods for normalization
                    max_hist = max(abs(macd_hist[i - 20 : i + 1]))  # Use last 20 periods
                    if max_hist > 0:
                        macd_signals[i] = macd_hist[i] / max_hist

            # RSI signals (weight: 1.0)
            rsi = indicators["rsi"]
            rsi_signals = pd.Series(0.0, index=indicators.index)
            rsi_signals[rsi < self.params["rsi_oversold"]] = 1.0  # Oversold -> Buy
            rsi_signals[rsi > self.params["rsi_overbought"]] = -1.0  # Overbought -> Sell

            # Bollinger Bands signals (weight: 0.5)
            bb_signals = pd.Series(0.0, index=indicators.index)
            price = indicators["Close"]
            bb_signals[price < indicators["bb_low"]] = 1.0  # Below lower band -> Buy
            bb_signals[price > indicators["bb_high"]] = -1.0  # Above upper band -> Sell
            bb_signals = bb_signals * 0.5

            # Stochastic signals (weight: 0.5)
            stoch_signals = pd.Series(0.0, index=indicators.index)
            stoch_k = indicators["stoch_k"]
            stoch_d = indicators["stoch_d"]
            stoch_signals[
                (stoch_k < self.params["stoch_oversold"])
                & (stoch_d < self.params["stoch_oversold"])
            ] = 1.0
            stoch_signals[
                (stoch_k > self.params["stoch_overbought"])
                & (stoch_d > self.params["stoch_overbought"])
            ] = -1.0
            stoch_signals = stoch_signals * 0.5

            # Combine signals
            signals = macd_signals + rsi_signals + bb_signals + stoch_signals

            # Normalize combined signals
            max_signal = max(abs(signals))
            if max_signal > 0:
                signals = signals / max_signal

            # Apply thresholds for stronger signals
            signals[abs(signals) < 0.3] = 0  # Clear weak signals
            signals[signals > 0] = 1  # Strong buy signals
            signals[signals < 0] = -1  # Strong sell signals

            # Print debug information
            print("\nSignal Statistics:")
            print(f"Buy signals: {(signals == 1).sum()}")
            print(f"Sell signals: {(signals == -1).sum()}")
            print(f"Neutral signals: {(signals == 0).sum()}")

            return signals

        except Exception as e:
            print(f"Error in generate_signals: {str(e)}")
            return pd.Series()

    def backtest(self, data: pd.DataFrame) -> float:
        """Run backtest and calculate Sharpe ratio."""
        try:
            # Calculate indicators
            indicators = self.indicators.calculate_all(data, self.params)
            if indicators.empty:
                return float("-inf")

            # Generate signals
            signals = self.generate_signals(indicators)
            if signals.empty:
                return float("-inf")

            # Calculate returns
            price_series = indicators["Close"]
            returns = price_series.pct_change()

            # Calculate strategy returns
            strategy_returns = (
                signals.shift(1) * returns
            )  # Shift signals to avoid look-ahead bias

            # Calculate Sharpe ratio
            if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                return float("-inf")

            sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()

            # Print debug information
            print("\nBacktest Statistics:")
            print(f"Total trades: {(signals != 0).sum()}")
            print(f"Average return: {strategy_returns.mean():.4%}")
            print(f"Return std: {strategy_returns.std():.4%}")
            print(f"Sharpe ratio: {sharpe_ratio:.2f}")

            return float(sharpe_ratio)

        except Exception as e:
            print(f"Error in backtest: {str(e)}")
            print(f"Data shape: {data.shape}")
            print(
                f"Indicators: {indicators.shape if 'indicators' in locals() else 'N/A'}"
            )
            print(f"Signals shape: {getattr(signals, 'shape', 'Not calculated')}")
            return float("-inf")


def optimize_strategy(
    symbol: str, start_date: str, end_date: str, n_trials: int = 100
) -> dict:
    """Optimize strategy parameters for a given symbol."""
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

    def objective(trial):
        params = {
            "macd_fast": trial.suggest_int("macd_fast", 8, 20),
            "macd_slow": trial.suggest_int("macd_slow", 21, 40),
            "macd_signal": trial.suggest_int("macd_signal", 5, 15),
            "rsi_period": trial.suggest_int("rsi_period", 10, 30),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 40),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 60, 80),
            "bb_period": trial.suggest_int("bb_period", 10, 30),
            "bb_std": trial.suggest_float("bb_std", 1.5, 3.0),
            "stoch_period": trial.suggest_int("stoch_period", 5, 20),
            "stoch_smooth": trial.suggest_int("stoch_smooth", 2, 10),
            "stoch_oversold": trial.suggest_int("stoch_oversold", 10, 30),
            "stoch_overbought": trial.suggest_int("stoch_overbought", 70, 90),
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

        return {"symbol": symbol, "best_params": best_params, "best_value": best_value}
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        return {"symbol": symbol, "best_params": {}, "best_value": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Optimize trading strategy parameters")
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of optimization trials",
    )

    args = parser.parse_args()

    for symbol in INSTRUMENTS:
        optimize_strategy(
            symbol=symbol,
            start_date=args.start,
            end_date=args.end,
            n_trials=args.trials,
        )


if __name__ == "__main__":
    main()


def calculate_indicators(data: pd.DataFrame, params: dict) -> pd.DataFrame:
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
