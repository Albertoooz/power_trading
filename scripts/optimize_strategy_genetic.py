#!/usr/bin/env python3
"""Genetic algorithm-based strategy optimizer.

Uses genetic algorithms to find optimal parameters for trading strategies.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import backtrader as bt
import pandas as pd

from power_trading.data.yahoo_loader import YahooFinanceLoader
from power_trading.genetic_optimizer import GeneticOptimizer, GeneticOptimizerConfig
from power_trading.strategy.residual_momentum import ResidualMomentumStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_cerebro(df: pd.DataFrame) -> bt.Cerebro:
    """Set up Backtrader cerebro instance with analyzers."""
    cerebro = bt.Cerebro()

    # Convert DataFrame to Backtrader data feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # Use index as datetime
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=-1,  # No open interest data
    )

    # Add data feed
    cerebro.adddata(data)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Set initial cash
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

    return cerebro


def get_parameter_ranges() -> dict[str, tuple[float, float]]:
    """Define parameter ranges for optimization."""
    return {
        "sma_period": (50, 400),
        "momentum_period": (20, 180),
        "atr_period": (20, 180),
        "atr_multiplier": (1.0, 4.0),
        "bb_period": (10, 40),
        "bb_dev": (1.0, 3.0),
        "pyramid_pct": (0.01, 0.05),
        "profit_target_pct": (0.1, 0.5),
        "position_size": (0.5, 1.0),
        "max_pyramids": (1, 5),
    }


def save_results(results: dict, output_dir: Path) -> None:
    """Save optimization results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best parameters
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(
            {
                "best_params": results["best_params"],
                "best_fitness": results["best_fitness"],
            },
            f,
            indent=2,
        )

    # Save generation history
    history_df = pd.DataFrame(results["generation_history"])
    history_df.to_csv(output_dir / "generation_history.csv", index=False)

    # Save summary
    with open(output_dir / "optimization_summary.txt", "w") as f:
        f.write(f"Optimization completed at: {datetime.now()}\n\n")
        f.write(f"Best fitness achieved: {results['best_fitness']}\n\n")
        f.write("Best parameters:\n")
        for param, value in results["best_params"].items():
            f.write(f"{param}: {value}\n")
        f.write("\nGeneration summary:\n")
        f.write(f"Total generations: {len(results['generation_history'])}\n")
        f.write(
            f"Initial best fitness: {results['generation_history'][0]['best_fitness']}\n"
        )
        f.write(
            f"Final best fitness: {results['generation_history'][-1]['best_fitness']}\n"
        )


def main() -> None:
    """Main optimization function."""
    parser = argparse.ArgumentParser(
        description="Optimize trading strategy parameters using genetic algorithms"
    )
    parser.add_argument(
        "--symbol", type=str, default="SPY", help="Stock symbol to optimize for"
    )
    parser.add_argument(
        "--start", type=str, default="2010-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--generations", type=int, default=10, help="Number of generations"
    )
    parser.add_argument("--population", type=int, default=20, help="Population size")
    parser.add_argument(
        "--selected",
        type=int,
        default=4,
        help="Number of best individuals to select as parents",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.7,
        help="Probability of crossover between parents",
    )
    parser.add_argument(
        "--mutation-std", type=float, default=0.1, help="Standard deviation for mutation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="optimization_results/genetic",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data for {args.symbol} from {args.start} to {args.end}")
    loader = YahooFinanceLoader()
    data = loader.load_data(args.symbol, args.start, args.end)

    # Set up cerebro
    cerebro = setup_cerebro(data)

    # Configure genetic optimizer
    config = GeneticOptimizerConfig(
        generation_count=args.generations,
        population_size=args.population,
        selected_count=args.selected,
        base_std=args.mutation_std,
        relative_std=True,
        crossover_rate=args.crossover_rate,
        seed=args.seed,
    )

    # Create and run optimizer
    logger.info("Starting genetic optimization...")
    optimizer = GeneticOptimizer(
        strategy_class=ResidualMomentumStrategy,
        param_ranges=get_parameter_ranges(),
        config=config,
    )

    results = optimizer.optimize(cerebro)

    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir)
    logger.info(f"Results saved to {output_dir}")

    # Print best results
    logger.info("Optimization completed!")
    logger.info(f"Best fitness: {results['best_fitness']}")
    logger.info("Best parameters:")
    for param, value in results["best_params"].items():
        logger.info(f"  {param}: {value}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nOptimization interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)
        sys.exit(1)
