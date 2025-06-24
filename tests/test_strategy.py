# import backtrader as bt
# import pandas as pd

# from power_trading.strategy import MyStrategy  # dostosuj jeśli masz inną nazwę folderu


# def test_strategy_instantiation() -> None:
#     cerebro = bt.Cerebro()
#     cerebro.addstrategy(MyStrategy)

#     # Przygotuj proste dane do backtestu
#     data_dict = {
#         "open": [1, 2, 3],
#         "high": [1, 2, 3],
#         "low": [1, 2, 3],
#         "close": [1, 2, 3],
#         "volume": [100, 200, 300],
#         "openinterest": [0, 0, 0],
#     }
#     df = pd.DataFrame(data_dict, index=pd.date_range("2023-01-01", periods=3))

#     data = bt.feeds.PandasData(dataname=df)
#     cerebro.adddata(data)

#     # Uruchom backtest (inicjuje obiekty i stan strategii)
#     strategies = cerebro.run()
#     strat = strategies[0]

#     assert isinstance(strat, MyStrategy)
