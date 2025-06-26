import importlib
from typing import Any

try:
    import backtrader as bt  # type: ignore
except ImportError:
    raise ImportError("Please install backtrader: pip install backtrader")

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
except ImportError:
    raise ImportError("Please install plotly: pip install plotly")

from power_trading.data.yahoo_loader import YahooFinanceLoader


class PlotlyVisualizer:
    @classmethod
    def plot_backtest(
        cls,
        strategy: Any,
        strategy_params: dict,
        df: pd.DataFrame,
        equity_curve: pd.Series,
    ) -> go.Figure:
        """Plot backtest results using Plotly.

        Args:
            strategy: The backtest strategy instance
            strategy_params: Dictionary of strategy parameters
            df: DataFrame with OHLCV data
            equity_curve: Series with equity curve data

        Returns:
            Plotly Figure object
        """
        # Prepare trade points
        buys = []
        sells = []
        for trade in strategy.trades:
            if trade["size"] > 0:
                buys.append((trade["date"], trade["price"]))
            else:
                sells.append((trade["date"], trade["price"]))

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.3, 0.2],
            vertical_spacing=0.02,
        )

        # Plot candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # Plot buy points
        if buys:
            buy_dates, buy_prices = zip(*buys)
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode="markers",
                    marker=dict(
                        color="lime",
                        symbol="triangle-up",
                        size=12,
                        line=dict(color="black", width=1),
                    ),
                    name="BUY",
                ),
                row=1,
                col=1,
            )

        # Plot sell points
        if sells:
            sell_dates, sell_prices = zip(*sells)
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode="markers",
                    marker=dict(
                        color="orangered",
                        symbol="triangle-down",
                        size=12,
                        line=dict(color="black", width=1),
                    ),
                    name="SELL",
                ),
                row=1,
                col=1,
            )

        # Plot equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                name="Equity Curve",
                line=dict(color="green", width=2),
            ),
            row=2,
            col=1,
        )

        # Plot volume
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color="rgba(200, 100, 50, 0.5)",
            ),
            row=3,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=f'{strategy_params.get("ticker")} Backtest Results',
            height=900,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
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

    class_name = (
        "".join(word.capitalize() for word in strategy_name.split("_")) + "Strategy"
    )

    try:
        strategy_class = getattr(module, class_name)
    except AttributeError:
        raise ValueError(
            f"Strategy class '{class_name}' not found in module '{module_name}'."
        )

    return strategy_class


def compute_stats(equity_curve, initial_cash, trade_analysis, benchmark_returns=None):
    # Convert equity curve to pandas Series if it isn't already
    equity = pd.Series(equity_curve)

    # Calculate returns and handle edge cases
    returns = equity.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    # Debug prints
    print("\nDebug information for returns calculation:")
    print(f"Number of returns: {len(returns)}")
    print(f"Mean return: {returns.mean():.6f}")
    print(f"Std return: {returns.std():.6f}")
    print(f"Min return: {returns.min():.6f}")
    print(f"Max return: {returns.max():.6f}")

    # Calculate alpha and beta if benchmark data is provided
    alpha = 0
    beta = 0
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # Align the benchmark returns with strategy returns
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) > 0:
            aligned_returns = returns[common_index]
            aligned_benchmark = benchmark_returns[common_index]

            # Calculate beta using covariance method with error handling
            try:
                covariance = np.cov(aligned_returns, aligned_benchmark)[0][1]
                benchmark_variance = np.var(aligned_benchmark)
                if benchmark_variance != 0:
                    beta = covariance / benchmark_variance

                    # Calculate alpha using CAPM formula
                    risk_free_rate = 0.02  # Assuming 2% risk-free rate
                    expected_return = risk_free_rate + beta * (
                        aligned_benchmark.mean() * 252 - risk_free_rate
                    )
                    actual_return = aligned_returns.mean() * 252
                    alpha = actual_return - expected_return
            except Exception as e:
                print(f"Error calculating alpha/beta: {str(e)}")
                alpha = 0
                beta = 0

    # Calculate CAGR with error handling
    try:
        cagr = ((equity.iloc[-1] / initial_cash) ** (252 / len(equity)) - 1) * 100
    except Exception:
        cagr = 0

    # Calculate risk metrics with proper error handling
    try:
        annual_return = returns.mean() * 252
        annual_std = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # Sharpe Ratio with minimum variance threshold
        min_std_threshold = 1e-8
        if annual_std > min_std_threshold:
            sharpe = annual_return / annual_std
        else:
            sharpe = 0

        # Sortino Ratio with error handling
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > min_std_threshold:
            sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino = 0

    except Exception as e:
        print(f"Error calculating risk metrics: {str(e)}")
        annual_return = 0
        annual_std = 0
        sharpe = 0
        sortino = 0

    # Calculate drawdown with error handling
    try:
        drawdown = (equity / equity.cummax() - 1).min() * 100
    except Exception:
        drawdown = 0

    # Calculate trading metrics
    total_trades = (
        trade_analysis.total.closed if hasattr(trade_analysis.total, "closed") else 0
    )
    won = trade_analysis.won.total if hasattr(trade_analysis.won, "total") else 0
    lost = trade_analysis.lost.total if hasattr(trade_analysis.lost, "total") else 0

    # Calculate win rate and related metrics with error handling
    if total_trades > 0:
        win_rate = (won / total_trades) * 100
        loss_rate = (lost / total_trades) * 100
    else:
        win_rate = 0
        loss_rate = 0

    avg_win = trade_analysis.won.pnl.average if won > 0 else 0
    avg_loss = trade_analysis.lost.pnl.average if lost > 0 else 0

    # Calculate expectancy and profit-loss ratio with error handling
    if total_trades > 0:
        expectancy = (win_rate * avg_win + loss_rate * avg_loss) / 100
    else:
        expectancy = 0

    if avg_loss != 0:
        profit_loss_ratio = avg_win / abs(avg_loss)
    else:
        profit_loss_ratio = 0

    # Calculate portfolio metrics
    info_ratio = sharpe  # Using Sharpe as Information Ratio for simplicity
    tracking_error = annual_std
    treynor = sharpe if beta != 0 else 0
    turnover = (total_trades / len(equity)) * 100 if len(equity) > 0 else 0

    return {
        "Start Equity": initial_cash,
        "End Equity": equity.iloc[-1],
        "Net Profit %": ((equity.iloc[-1] - initial_cash) / initial_cash) * 100,
        "CAGR %": cagr,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown %": drawdown,
        "Annual Std Dev": annual_std * 100,
        "Annual Variance": (annual_std**2) * 100,
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
        "Total Fees $": 0,
        "Portfolio Turnover %": turnover,
    }


def run_backtest(params):
    df = YahooFinanceLoader().load_data(params["ticker"], params["start"], params["end"])
    df.index = pd.to_datetime(df.index)

    # Load benchmark data (S&P 500)
    benchmark_df = YahooFinanceLoader().load_data("^GSPC", params["start"], params["end"])
    benchmark_returns = benchmark_df["Close"].pct_change().dropna()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(params["cash"])

    data_feed = bt.feeds.PandasData(
        dataname=df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=-1,
    )
    cerebro.adddata(data_feed)

    strategy_class = load_strategy(params["strategy"])

    strat_params = {
        k: v
        for k, v in params.items()
        if k not in ["strategy", "ticker", "start", "end", "cash"] and v is not None
    }

    cerebro.addstrategy(strategy_class, **strat_params)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")

    print("Running backtest...")
    results = cerebro.run()
    strategy = results[0]
    trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()

    # Create equity curve with proper dates
    dates = df.index[-(len(strategy.equity_curve)) :]  # Get actual dates from data
    equity_curve = pd.Series(strategy.equity_curve, index=dates)

    # Ensure benchmark returns align with our trading dates
    benchmark_returns = benchmark_returns[benchmark_returns.index.isin(dates)]

    stats = compute_stats(equity_curve, params["cash"], trade_analysis, benchmark_returns)

    print("\n===== Backtest Report =====")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    fig = PlotlyVisualizer.plot_backtest(strategy, params, df, equity_curve)
    fig.show()

    return stats, fig
