import argparse

from power_trading.backtest import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=[
            "moving_average",
            "rsi_reversion",
            "optimized_technical",
            "momentum_gtaa",
            "residual_momentum",
            "optimized_residual_momentum",
        ],
        required=True,
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--cash", type=float, default=10000)

    # Moving Average params
    parser.add_argument("--fast_period", type=int)
    parser.add_argument("--slow_period", type=int)

    # RSI Reversion params
    parser.add_argument("--rsi_period", type=int)
    parser.add_argument("--rsi_lower", type=float)
    parser.add_argument("--rsi_upper", type=float)

    # Optimized Technical Strategy params
    parser.add_argument(
        "--risk_pct", type=float, help="Risk percentage per trade (default: 0.02)"
    )
    parser.add_argument("--macd_fast", type=int, help="MACD fast period (default: 11)")
    parser.add_argument("--macd_slow", type=int, help="MACD slow period (default: 23)")
    parser.add_argument(
        "--macd_signal", type=int, help="MACD signal period (default: 15)"
    )
    parser.add_argument(
        "--bb_period", type=int, help="Bollinger Bands period (default: 27)"
    )
    parser.add_argument(
        "--bb_dev", type=float, help="Bollinger Bands deviation (default: 2.19)"
    )
    parser.add_argument("--stoch_period", type=int, help="Stochastic period (default: 8)")
    parser.add_argument(
        "--stoch_smooth", type=int, help="Stochastic smoothing (default: 2)"
    )

    # Momentum GTAA Strategy params
    parser.add_argument(
        "--sma_period", type=int, help="SMA period for trend following (default: 200)"
    )
    parser.add_argument(
        "--momentum_period", type=int, help="Lookback period for momentum (default: 90)"
    )
    parser.add_argument(
        "--pyramid_pct", type=float, help="Percent move for pyramid entry (default: 0.02)"
    )
    parser.add_argument(
        "--profit_target_pct", type=float, help="Profit target percentage (default: 0.30)"
    )
    parser.add_argument(
        "--stop_loss_pct", type=float, help="Stop loss percentage (default: 0.25)"
    )
    parser.add_argument(
        "--position_size", type=float, help="Position size as fraction (default: 0.95)"
    )

    # Residual Momentum Strategy params
    parser.add_argument(
        "--atr_period", type=int, help="ATR period for volatility stops (default: 90)"
    )
    parser.add_argument(
        "--atr_multiplier", type=float, help="ATR multiplier for stop loss (default: 2.0)"
    )
    parser.add_argument(
        "--max_pyramids", type=int, help="Maximum number of pyramid entries (default: 3)"
    )

    args = parser.parse_args()

    params = vars(args)
    run_backtest(params)


if __name__ == "__main__":
    main()
