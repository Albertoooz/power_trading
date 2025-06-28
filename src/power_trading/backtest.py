import importlib
from typing import Any, cast

try:
    import backtrader as bt
except ImportError:
    raise ImportError("Please install backtrader: pip install backtrader")

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("Please install plotly: pip install plotly")

from power_trading.data.yahoo_loader import YahooFinanceLoader


class PlotlyVisualizer:
    @classmethod
    def plot_backtest(
        cls,
        strategy: Any,
        strategy_params: dict[str, Any],
        df: DataFrame,
        equity_curve: Series,
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
        # Calculate monthly returns for heatmap
        daily_returns = equity_curve.pct_change()
        monthly_returns = pd.DataFrame(index=equity_curve.index)
        monthly_returns["Year"] = monthly_returns.index.year
        monthly_returns["Month"] = monthly_returns.index.month
        monthly_returns["Returns"] = daily_returns

        # Calculate cumulative returns for each month
        monthly_returns = (
            monthly_returns.groupby(["Year", "Month"])["Returns"]
            .apply(lambda x: (1 + x).prod() - 1)
            .reset_index()
        )

        # Pivot the data for heatmap
        heatmap_data = monthly_returns.pivot(
            index="Year", columns="Month", values="Returns"
        )

        # Create month labels
        month_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        # Prepare trade points
        buys = []
        sells = []
        for trade in strategy.trades:
            if trade["size"] > 0:
                buys.append((trade["date"], trade["price"]))
            else:
                sells.append((trade["date"], trade["price"]))

        # Create figure with subplots including heatmap
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.4, 0.2, 0.15, 0.25],  # Adjusted heights to include heatmap
            vertical_spacing=0.02,
            subplot_titles=("Price Action", "Equity Curve", "Volume", "Monthly Returns"),
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

        # Add monthly returns heatmap
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values * 100,  # Convert to percentage
                x=month_labels,
                y=heatmap_data.index,
                colorscale=[
                    [0, "red"],  # Negative returns
                    [0.5, "white"],  # Zero
                    [1, "green"],  # Positive returns
                ],
                zmid=0,  # Center the color scale at zero
                text=np.round(heatmap_data.values * 100, 1),
                texttemplate="%{text}%",
                textfont={"size": 10},
                hoverongaps=False,
                showscale=True,
                name="Monthly Returns",
                colorbar=dict(
                    len=0.25,  # Length of colorbar relative to the plot height
                    yanchor="bottom",  # Anchor point for y position
                    y=0,  # Position at the bottom of the plot
                    title="Returns %",
                    titleside="right",
                    thickness=20,  # Width of the colorbar
                ),
            ),
            row=4,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=f'{strategy_params.get("ticker")} Backtest Results',
            height=1200,  # Increased height to accommodate heatmap
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )

        # Update axes labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Equity", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        fig.update_yaxes(title_text="Year", row=4, col=1)
        fig.update_xaxes(title_text="Month", row=4, col=1)

        # Update heatmap specific layout
        fig.update_layout(
            {
                f"xaxis{4}": {
                    "tickmode": "array",
                    "ticktext": month_labels,
                    "tickvals": list(range(12)),
                },
                f"yaxis{4}": {"autorange": "reversed"},  # Most recent year at top
            }
        )

        return fig


def load_strategy(strategy_name: str) -> type[bt.Strategy]:
    """Load strategy class by name.

    Args:
        strategy_name: Name of the strategy module

    Returns:
        Strategy class

    Raises:
        ValueError: If strategy module or class not found
    """
    module_name = f"power_trading.strategy.{strategy_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Strategy module '{strategy_name}' not found.")

    class_name = (
        "".join(word.capitalize() for word in strategy_name.split("_")) + "Strategy"
    )

    try:
        strategy_class = cast(type[bt.Strategy], getattr(module, class_name))
    except AttributeError:
        raise ValueError(
            f"Strategy class '{class_name}' not found in module '{module_name}'."
        )

    return strategy_class


def compute_stats(
    equity_curve: Series,
    initial_cash: float,
    trade_analysis: Any,
    benchmark_returns: Series | None = None,
) -> dict[str, float | int]:
    """Compute backtest statistics.

    Args:
        equity_curve: Series with equity curve data
        initial_cash: Initial cash amount
        trade_analysis: Trade analyzer results
        benchmark_returns: Optional benchmark returns series

    Returns:
        Dictionary with computed statistics
    """
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

    # Calculate trading metrics with safe access to trade analysis data
    total_trades = 0
    won = 0
    lost = 0
    avg_win = 0
    avg_loss = 0

    try:
        if hasattr(trade_analysis, "total") and hasattr(trade_analysis.total, "total"):
            total_trades = trade_analysis.total.total
            if hasattr(trade_analysis, "won"):
                won = (
                    trade_analysis.won.total
                    if hasattr(trade_analysis.won, "total")
                    else 0
                )
                avg_win = (
                    trade_analysis.won.pnl.average
                    if hasattr(trade_analysis.won, "pnl")
                    else 0
                )
            if hasattr(trade_analysis, "lost"):
                lost = (
                    trade_analysis.lost.total
                    if hasattr(trade_analysis.lost, "total")
                    else 0
                )
                avg_loss = (
                    trade_analysis.lost.pnl.average
                    if hasattr(trade_analysis.lost, "pnl")
                    else 0
                )
    except Exception as e:
        print(f"Error accessing trade analysis data: {str(e)}")

    # Calculate win rate and related metrics
    win_rate = (won / total_trades * 100) if total_trades > 0 else 0
    loss_rate = (lost / total_trades * 100) if total_trades > 0 else 0

    # Calculate expectancy and profit-loss ratio
    expectancy = (
        (win_rate * avg_win + loss_rate * avg_loss) / 100 if total_trades > 0 else 0
    )
    profit_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0

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


def run_backtest(params: dict[str, Any]) -> tuple[dict[str, float | int], go.Figure]:
    """Run backtest with given parameters.

    Args:
        params: Dictionary with backtest parameters

    Returns:
        Tuple of (statistics dictionary, plotly figure)
    """
    df = YahooFinanceLoader().load_data(params["ticker"], params["start"], params["end"])
    df.index = pd.to_datetime(df.index)

    # Load benchmark data (S&P 500)
    benchmark_df = YahooFinanceLoader().load_data("^GSPC", params["start"], params["end"])
    benchmark_returns = pd.Series(
        benchmark_df["Close"].pct_change().dropna().values,
        index=benchmark_df["Close"].pct_change().dropna().index,
    )

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

    main_fig = PlotlyVisualizer.plot_backtest(strategy, params, df, equity_curve)
    main_fig.show()

    return stats, main_fig
