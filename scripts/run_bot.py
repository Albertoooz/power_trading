import argparse
from datetime import datetime
from power_trading.backtest import run_backtest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["moving_average", "rsi_reversion"], required=True)
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


    args = parser.parse_args()

    params = vars(args)
    run_backtest(params)

if __name__ == "__main__":
    main()
