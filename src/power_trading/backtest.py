import importlib
from datetime import datetime
import backtrader as bt
import pandas as pd
import numpy as np
from power_trading.data.yahoo_loader import YahooFinanceLoader


from plotly.subplots import make_subplots
import plotly.graph_objects as go


class PlotlyVisualizer:
    @classmethod
    def plot_backtest(cls, strategy, strategy_params, df, equity_curve):
        buys = []
        sells = []
        for trade in strategy.trades:
            if trade['size'] > 0:
                buys.append((trade['date'], trade['price']))
            else:
                sells.append((trade['date'], trade['price']))

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.5, 0.3, 0.2],
            vertical_spacing=0.02
        )

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ), row=1, col=1)

        # Dodaj wskaźniki jeśli są dostępne
        if strategy_params.get('strategy') == 'moving_average':
            fast_ma = np.where(np.isnan(strategy.fast_ma.array), None, strategy.fast_ma.array)
            slow_ma = np.where(np.isnan(strategy.slow_ma.array), None, strategy.slow_ma.array)

            fig.add_trace(go.Scatter(
                x=df.index,
                y=fast_ma,
                line=dict(color='blue', width=1),
                name=f'Fast MA ({strategy_params.get("fast_period")})'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index,
                y=slow_ma,
                line=dict(color='red', width=1),
                name=f'Slow MA ({strategy_params.get("slow_period")})'
            ), row=1, col=1)

        if buys:
            buy_dates, buy_prices = zip(*buys)
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                marker=dict(color='lime', symbol='triangle-up', size=12, line=dict(color='black', width=1)),
                name='BUY'
            ), row=1, col=1)

        if sells:
            sell_dates, sell_prices = zip(*sells)
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                marker=dict(color='orangered', symbol='triangle-down', size=12, line=dict(color='black', width=1)),
                name='SELL'
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            name='Equity Curve',
            line=dict(color='green', width=2)
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='rgba(200, 100, 50, 0.5)'
        ), row=3, col=1)

        fig.update_layout(
            title=f'{strategy_params.get("ticker")} Backtest Results',
            height=900,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Equity", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)

        return fig


def load_strategy(strategy_name: str):
    module_name = f"power_trading.strategy.{strategy_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Strategy module '{strategy_name}' not found.")

    class_name = ''.join(word.capitalize() for word in strategy_name.split('_')) + 'Strategy'

    try:
        strategy_class = getattr(module, class_name)
    except AttributeError:
        raise ValueError(f"Strategy class '{class_name}' not found in module '{module_name}'.")

    return strategy_class


def compute_stats(equity_curve, initial_cash, trade_analysis):
    equity = pd.Series(equity_curve)
    returns = equity.pct_change().dropna()
    cagr = ((equity.iloc[-1] / initial_cash) ** (252 / len(equity)) - 1) * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    sortino = (returns.mean() / returns[returns < 0].std()) * np.sqrt(252) if returns[returns < 0].std() != 0 else 0
    drawdown = (equity / equity.cummax() - 1).min() * 100
    std = returns.std() * np.sqrt(252)
    var = std ** 2

    total_trades = trade_analysis.total.closed if hasattr(trade_analysis.total, 'closed') else 0
    won = trade_analysis.won.total if hasattr(trade_analysis.won, 'total') else 0
    lost = trade_analysis.lost.total if hasattr(trade_analysis.lost, 'total') else 0
    win_rate = (won / total_trades) * 100 if total_trades > 0 else 0
    loss_rate = (lost / total_trades) * 100 if total_trades > 0 else 0
    avg_win = trade_analysis.won.pnl.average if won > 0 else 0
    avg_loss = trade_analysis.lost.pnl.average if lost > 0 else 0
    expectancy = (win_rate * avg_win + loss_rate * avg_loss) / 100 if total_trades > 0 else 0
    profit_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0

    alpha = 0
    beta = 0
    info_ratio = sharpe
    tracking_error = std
    treynor = sharpe
    fees = 0
    turnover = (total_trades / len(equity)) * 100 if len(equity) > 0 else 0

    return {
        "Start Equity": initial_cash,
        "End Equity": equity.iloc[-1],
        "Net Profit %": ((equity.iloc[-1] - initial_cash) / initial_cash) * 100,
        "CAGR %": cagr,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown %": drawdown,
        "Annual Std Dev": std * 100,
        "Annual Variance": var * 100,
        "Alpha": alpha,
        "Beta": beta,
        "Information Ratio": info_ratio,
        "Tracking Error": tracking_error * 100,
        "Treynor Ratio": treynor,
        "Total Orders": total_trades,
        "Win Rate %": win_rate,
        "Loss Rate %": loss_rate,
        "Average Win": avg_win,
        "Average Loss": avg_loss,
        "Expectancy": expectancy,
        "Profit-Loss Ratio": profit_loss_ratio,
        "Total Fees $": fees,
        "Portfolio Turnover %": turnover
    }


def run_backtest(params):
    df = YahooFinanceLoader().load_data(params['ticker'], params['start'], params['end'])
    df.index = pd.to_datetime(df.index)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(params['cash'])

    data_feed = bt.feeds.PandasData(
        dataname=df,
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1
    )
    cerebro.adddata(data_feed)

    strategy_class = load_strategy(params['strategy'])

    # Przygotuj parametry strategii do Cerebro, filtrując te, które są nie None i nie 'strategy' itp.
    strat_params = {k: v for k, v in params.items() if k not in ['strategy', 'ticker', 'start', 'end', 'cash'] and v is not None}

    cerebro.addstrategy(strategy_class, **strat_params)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    print("Running backtest...")
    results = cerebro.run()
    strategy = results[0]
    trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()

    equity_curve = pd.Series(strategy.equity_curve, index=df.index[-len(strategy.equity_curve):])

    stats = compute_stats(strategy.equity_curve, params['cash'], trade_analysis)

    print("\n===== Backtest Report =====")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    fig = PlotlyVisualizer.plot_backtest(strategy, params, df, equity_curve)
    fig.show()

    return stats, fig
