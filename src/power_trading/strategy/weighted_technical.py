from typing import Any

import backtrader as bt


class WeightedTechnicalStrategy(bt.Strategy):
    """Strategy that combines multiple technical indicators with weights."""

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

        self.trades: list[dict[str, Any]] = []
        self.equity_curve: list[float] = []

    def get_rsi_signal(self) -> float:
        """Get RSI signal (-1 to 1)."""
        rsi_value = float(self.rsi[0])
        if rsi_value < self.p.rsi_lower:
            return 1.0
        elif rsi_value > self.p.rsi_upper:
            return -1.0
        return 0.0

    def get_macd_signal(self) -> float:
        """Get MACD signal (-1 to 1)."""
        macd = float(self.macd.macd[0])
        signal = float(self.macd.signal[0])
        result: float = 1.0 if macd > signal else -1.0
        return result

    def get_bb_signal(self) -> float:
        """Get Bollinger Bands signal (-1 to 1)."""
        price = float(self.data.close[0])
        bottom = float(self.bb.bot[0])
        top = float(self.bb.top[0])
        if price < bottom:
            return 1.0
        elif price > top:
            return -1.0
        return 0.0

    def get_weighted_signal(self) -> float:
        """Calculate weighted signal from all indicators."""
        rsi_signal = self.get_rsi_signal() * self.p.rsi_weight
        macd_signal = self.get_macd_signal() * self.p.macd_weight
        bb_signal = self.get_bb_signal() * self.p.bb_weight

        return rsi_signal + macd_signal + bb_signal

    def next(self) -> None:
        """Execute trading logic."""
        signal = self.get_weighted_signal()

        if not self.position:
            if signal > 0.3:  # Strong buy signal
                self.buy()
                self.trades.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": float(self.data.close[0]),  # Convert to float explicitly
                        "size": 1,
                    }
                )
            elif signal < -0.3:  # Strong sell signal
                self.sell()
                self.trades.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": float(self.data.close[0]),  # Convert to float explicitly
                        "size": -1,
                    }
                )
        else:
            if (self.position.size > 0 and signal < -0.3) or (
                self.position.size < 0 and signal > 0.3
            ):
                self.close()
                self.trades.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": float(self.data.close[0]),  # Convert to float explicitly
                        "size": 0,
                    }
                )

        self.equity_curve.append(
            float(self.broker.getvalue())
        )  # Convert to float explicitly
