from typing import Any

import backtrader as bt


class MovingAverageStrategy(bt.Strategy):
    """Simple moving average crossover strategy."""

    params = dict(
        fast_period=10,
        slow_period=30,
    )

    def __init__(self) -> None:
        """Initialize strategy parameters."""
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.trades: list[dict[str, Any]] = []
        self.equity_curve: list[float] = []

    def next(self) -> None:
        """Execute trading logic."""
        self.equity_curve.append(self.broker.getvalue())
        if not self.position:
            if self.crossover > 0:  # Fast MA crosses above slow MA
                self.buy()
                self.trades.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": self.data.close[0],
                        "size": 1,
                    }
                )
        elif self.crossover < 0:  # Fast MA crosses below slow MA
            self.close()
            self.trades.append(
                {
                    "date": self.data.datetime.date(0),
                    "price": self.data.close[0],
                    "size": -1,
                }
            )
