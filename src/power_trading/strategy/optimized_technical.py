from typing import Any

import backtrader as bt


class OptimizedTechnicalStrategy(bt.Strategy):
    """Strategy that combines multiple technical indicators with optimized
    parameters."""

    params = dict(
        rsi_period=14,
        rsi_upper=70,
        rsi_lower=30,
        rsi_weight=0.3,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        macd_weight=0.3,
        bb_period=20,
        bb_dev=2,
        bb_weight=0.4,
        atr_period=14,
        risk_pct=0.02,  # Risk 2% per trade
    )

    def __init__(self) -> None:
        """Initialize strategy parameters."""
        # RSI
        self.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.p.rsi_period,
        )

        # MACD
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
        )

        # Bollinger Bands
        self.bb = bt.indicators.BollingerBands(
            self.data.close,
            period=self.p.bb_period,
            devfactor=self.p.bb_dev,
        )

        # ATR for position sizing
        self.atr = bt.indicators.ATR(
            self.data,
            period=self.p.atr_period,
        )

        self.trades: list[dict[str, Any]] = []
        self.equity_curve: list[float] = []

    def get_rsi_signal(self) -> float:
        """Get RSI signal (-1 to 1)."""
        rsi_value = float(self.rsi[0])  # Access RSI value directly
        if rsi_value < self.p.rsi_lower:
            return 1.0  # Oversold - buy signal
        elif rsi_value > self.p.rsi_upper:
            return -1.0  # Overbought - sell signal
        return 0.0

    def get_macd_signal(self) -> float:
        """Get MACD signal (-1 to 1)."""
        macd = float(self.macd.lines.macd[0])  # Access MACD line
        signal = float(self.macd.lines.signal[0])  # Access signal line
        result: float = (
            1.0 if macd > signal else -1.0
        )  # Buy when MACD crosses above signal
        return result

    def get_bb_signal(self) -> float:
        """Get Bollinger Bands signal (-1 to 1)."""
        price = float(self.data.close[0])
        bottom = float(self.bb.lines.bot[0])  # Access bottom band
        top = float(self.bb.lines.top[0])  # Access top band
        if price < bottom:
            return 1.0  # Price below lower band - buy signal
        elif price > top:
            return -1.0  # Price above upper band - sell signal
        return 0.0

    def get_weighted_signal(self) -> float:
        """Calculate weighted signal from all indicators."""
        rsi_signal = self.get_rsi_signal() * self.p.rsi_weight
        macd_signal = self.get_macd_signal() * self.p.macd_weight
        bb_signal = self.get_bb_signal() * self.p.bb_weight

        return rsi_signal + macd_signal + bb_signal

    def calculate_position_size(self) -> int:
        """Calculate position size based on risk percentage."""
        price = float(self.data.close[0])
        atr = float(self.atr[0])  # Access ATR value directly

        if atr == 0:  # Avoid division by zero
            return 0

        risk_amount = float(self.broker.getvalue()) * self.p.risk_pct
        stop_distance = atr * 2  # Use 2 ATR for stop loss

        return max(
            1, int(risk_amount / (price * stop_distance))
        )  # Ensure at least 1 share

    def next(self) -> None:
        """Execute trading logic."""
        signal = self.get_weighted_signal()
        position_size = self.calculate_position_size()

        if not self.position:
            if signal > 0.2:  # Lower threshold for more trades
                self.buy(size=position_size)
                self.trades.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": float(self.data.close[0]),
                        "size": position_size,
                    }
                )
            elif signal < -0.2:  # Lower threshold for more trades
                self.sell(size=position_size)
                self.trades.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": float(self.data.close[0]),
                        "size": -position_size,
                    }
                )
        else:
            if (self.position.size > 0 and signal < -0.2) or (
                self.position.size < 0 and signal > 0.2
            ):
                self.close()
                self.trades.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": float(self.data.close[0]),
                        "size": 0,
                    }
                )

        self.equity_curve.append(
            float(self.broker.getvalue())
        )  # Convert to float explicitly

    def log(self, txt: str, dt: Any | None = None) -> None:
        """Log message with optional datetime."""
        dt = dt or self.data.datetime.date(0)
        print(f"{dt}: {txt}")
