import backtrader as bt

class RsiReversionStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_lower', 30),
        ('rsi_upper', 70),
        ('size', 1),  # wielkość pozycji
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.trades = []
        self.equity_curve = []

    def next(self):
        self.equity_curve.append(self.broker.getvalue())

        if not self.position:
            if self.rsi < self.p.rsi_lower:
                self.buy(size=self.p.size)
                self.trades.append({
                    'date': self.data.datetime.date(0),
                    'price': self.data.close[0],
                    'size': 1
                })
        else:
            if self.rsi > self.p.rsi_upper:
                self.close()
                self.trades.append({
                    'date': self.data.datetime.date(0),
                    'price': self.data.close[0],
                    'size': -1
                })
