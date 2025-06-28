import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, TypeVar, runtime_checkable

import backtrader as bt
import numpy as np

# Define types for backtrader
T = TypeVar("T")


@runtime_checkable
class Analyzer(Protocol):
    """Protocol for backtrader Analyzer class."""

    def get_analysis(self) -> dict[str, float]: ...


@runtime_checkable
class Strategy(Protocol):
    """Protocol for backtrader Strategy class."""

    analyzers: dict[str, Analyzer]


@runtime_checkable
class Cerebro(Protocol):
    """Protocol for backtrader Cerebro class."""

    broker: Any
    datas: list[Any]

    def addstrategy(self, strategy_class: type[Strategy], **kwargs: Any) -> None: ...
    def run(self) -> list[Strategy]: ...
    def adddata(self, data: Any) -> None: ...
    def addanalyzer(self, analyzer: type[Analyzer], _name: str) -> None: ...


# Lazy imports for runtime
def get_backtrader() -> bt:
    """Import backtrader at runtime."""
    import backtrader as bt  # noqa

    return bt


@dataclass
class GeneticOptimizerConfig:
    """Configuration for genetic optimization."""

    generation_count: int = 5
    population_size: int = 20
    selected_count: int = 4
    base_std: float = 0.1
    relative_std: bool = True
    crossover_rate: float = 0.7
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.selected_count > self.population_size:
            raise ValueError("Selected count must be less than population size")
        if self.selected_count < 2:
            raise ValueError("Selected count must be at least 2")
        if self.crossover_rate < 0 or self.crossover_rate > 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        if self.base_std <= 0:
            raise ValueError("Base standard deviation must be positive")


class GeneticOptimizer:
    """Genetic algorithm optimizer for trading strategies."""

    def __init__(
        self,
        strategy_class: type[bt.Strategy],
        param_ranges: dict[str, tuple[float, float]],
        config: GeneticOptimizerConfig | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            strategy_class: The strategy class to optimize
            param_ranges: Dictionary mapping parameter names to (min, max) ranges
            config: Optional configuration for the genetic algorithm
        """
        self.strategy_class = strategy_class
        self.param_ranges = param_ranges
        self.config = config or GeneticOptimizerConfig()

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        self.best_params: dict[str, float | int] = {}
        self.best_fitness: float = float("-inf")
        self.population: list[dict[str, float | int]] = []
        self.generation_history: list[dict[str, Any]] = []

    def _init_population(self) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(self.config.population_size):
            params: dict[str, float | int] = {}
            for name, (min_val, max_val) in self.param_ranges.items():
                value = random.uniform(min_val, max_val)  # nosec B311
                if name.endswith("_period") or name == "max_pyramids":
                    params[name] = int(value)
                else:
                    params[name] = value
            self.population.append(params)

    def _evaluate_individual(
        self, params: dict[str, float | int], cerebro: bt.Cerebro
    ) -> float:
        """Evaluate a single parameter set using backtrader."""
        cerebro.addstrategy(self.strategy_class, **params)

        # Run backtest
        results = cerebro.run()

        # Get the first strategy instance
        strat = results[0]

        # Calculate fitness using Sharpe Ratio and Returns
        analysis = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = analysis.get("sharperatio", 0.0)
        if sharpe_ratio is None:
            sharpe_ratio = 0.0

        analysis = strat.analyzers.returns.get_analysis()
        returns = analysis.get("rtot", 0.0)
        if returns is None:
            returns = 0.0

        # Combine metrics into fitness score
        fitness = max(0.0, sharpe_ratio) * 0.6 + max(0.0, returns) * 0.4

        return float(fitness)

    def _select_parents(self, fitnesses: list[float]) -> list[dict[str, float | int]]:
        """Select best individuals as parents."""
        population_with_fitness = list(zip(fitnesses, self.population))
        sorted_pop = sorted(population_with_fitness, key=lambda x: x[0], reverse=True)
        return [params for _, params in sorted_pop[: self.config.selected_count]]

    def _crossover(
        self,
        parent1: dict[str, float | int],
        parent2: dict[str, float | int],
    ) -> tuple[dict[str, float | int], dict[str, float | int]]:
        """Perform crossover between two parents."""
        if random.random() > self.config.crossover_rate:  # nosec B311
            return parent1.copy(), parent2.copy()

        child1 = {}
        child2 = {}
        for param in self.param_ranges.keys():
            if random.random() < 0.5:  # nosec B311
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]

        return child1, child2

    def _mutate(self, individual: dict[str, float | int]) -> dict[str, float | int]:
        """Mutate an individual."""
        mutated = individual.copy()
        for param, (min_val, max_val) in self.param_ranges.items():
            # Calculate standard deviation for mutation
            std = self.config.base_std
            if self.config.relative_std:
                param_range = max_val - min_val
                std = self.config.base_std * param_range

            # Apply mutation with probability
            if random.random() < 0.1:  # nosec B311
                value = float(mutated[param]) + np.random.normal(0, std)
                # Convert to int for period parameters
                if param.endswith("_period") or param == "max_pyramids":
                    mutated[param] = int(max(min_val, min(max_val, value)))
                else:
                    mutated[param] = max(min_val, min(max_val, value))

        return mutated

    def optimize(self, cerebro: bt.Cerebro) -> dict[str, Any]:
        """Run genetic optimization.

        Args:
            cerebro: Backtrader cerebro instance with data feeds and analyzers

        Returns:
            Dictionary containing optimization results
        """
        # Initialize population
        self._init_population()

        # Run generations
        for gen in range(self.config.generation_count):
            # Evaluate current population
            fitnesses = []
            for params in self.population:
                # Create fresh cerebro instance for each evaluation
                cerebro_clone = bt.Cerebro()
                cerebro_clone.broker = cerebro.broker
                for data in cerebro.datas:
                    cerebro_clone.adddata(data)
                # Add analyzers
                cerebro_clone.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
                cerebro_clone.addanalyzer(bt.analyzers.Returns, _name="returns")

                fitness = self._evaluate_individual(params, cerebro_clone)
                fitnesses.append(fitness)

                # Update best if needed
                if fitness > self.best_fitness:
                    self.best_fitness = float(fitness)  # Explicit conversion to float
                    self.best_params = params.copy()

            # Record generation stats
            gen_stats = {
                "generation": gen + 1,
                "best_fitness": max(fitnesses),
                "avg_fitness": sum(fitnesses) / len(fitnesses),
                "best_params": self.population[fitnesses.index(max(fitnesses))].copy(),
                "timestamp": datetime.now().isoformat(),
            }
            self.generation_history.append(gen_stats)

            # Select parents for next generation
            parents = self._select_parents(fitnesses)

            # Create next generation with type annotation
            next_gen: list[dict[str, float | int]] = []
            next_gen.extend(parents)  # Elitism

            # Create next generation through crossover and mutation
            while len(next_gen) < self.config.population_size:
                parent1 = random.choice(parents)  # nosec B311
                parent2 = random.choice(parents)  # nosec B311
                child1, child2 = self._crossover(parent1, parent2)
                next_gen.extend([self._mutate(child1), self._mutate(child2)])

            # Truncate to population size
            self.population = next_gen[: self.config.population_size]

        return {
            "best_params": self.best_params,
            "best_fitness": self.best_fitness,
            "generation_history": self.generation_history,
        }
