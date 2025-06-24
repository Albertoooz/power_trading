import backtrader as bt


class MovingAverageStrategy(bt.Strategy):
    params = (
        ('fast_period', 20),
        ('slow_period', 50),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.equity_curve = []
        self.trades = []

    def next(self):
        self.equity_curve.append(self.broker.getvalue())
        if not self.position:
            if self.crossover > 0:
                self.buy()
                self.trades.append({'date': self.data.datetime.date(0), 'price': self.data.close[0], 'size': 1})
        elif self.crossover < 0:
            self.close()
            self.trades.append({'date': self.data.datetime.date(0), 'price': self.data.close[0], 'size': -1})

