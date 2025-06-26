import backtrader as bt


class BreakoutStrategy(bt.Strategy):
    """Simple breakout strategy."""

    params = dict(
        period=20,
    )

    def __init__(self) -> None:
        """Initialize strategy parameters."""
        self.period = self.p.period
        self.high_band = bt.indicators.Highest(self.data.high, period=self.period)
        self.low_band = bt.indicators.Lowest(self.data.low, period=self.period)

    def next(self) -> None:
        """Execute trading logic."""
        if not self.position:
            if self.data.close[0] > self.high_band[-1]:
                self.buy()
        else:
            if self.data.close[0] < self.low_band[-1]:
                self.sell()
