from typing import Any

import backtrader as bt


class RSIReversionStrategy(bt.Strategy):
    """RSI mean reversion strategy."""

    params = dict(
        period=14,
        overbought=70,
        oversold=30,
    )

    def __init__(self) -> None:
        """Initialize strategy parameters."""
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)
        self.trades: list[dict[str, Any]] = []
        self.equity_curve: list[float] = []

    def next(self) -> None:
        """Execute trading logic."""
        self.equity_curve.append(self.broker.getvalue())

        if not self.position:
            if self.rsi < self.p.oversold:
                self.buy()
                self.trades.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": self.data.close[0],
                        "size": 1,
                    }
                )
            elif self.rsi > self.p.overbought:
                self.sell()
                self.trades.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": self.data.close[0],
                        "size": -1,
                    }
                )
        else:
            if self.position.size > 0 and self.rsi > 50:
                self.sell()
                self.trades.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": self.data.close[0],
                        "size": -1,
                    }
                )
            elif self.position.size < 0 and self.rsi < 50:
                self.buy()
