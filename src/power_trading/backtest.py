# src/my_trading_bot/backtest.py

from datetime import datetime

import backtrader as bt

from .strategy import MyStrategy


def run_backtest():
    cerebro = bt.Cerebro()
    data = bt.feeds.YahooFinanceData(
        dataname='AAPL',
        fromdate=datetime(2023, 1, 1),
        todate=datetime(2024, 1, 1)
    )
    cerebro.adddata(data)
    cerebro.addstrategy(MyStrategy)
    cerebro.run()
    cerebro.plot()

if __name__ == "__main__":
    run_backtest()
