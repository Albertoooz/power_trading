<p align="center">
  <a href="https://pre-commit.com/" target="_blank"><img src="https://pre-commit.com/logo.svg" height="100" alt="pre-commit logo" /></a>
  <a href="https://github.com/astral-sh/ruff" target="_blank"><img src="https://raw.githubusercontent.com/astral-sh/ruff/8c20f14e62ddaf7b6d62674f300f5d19cbdc5acb/docs/assets/bolt.svg" height="100" alt="ruff logo" style="background-color: #ef5552" /></a>
  <a href="https://bandit.readthedocs.io/" target="_blank"><img src="https://raw.githubusercontent.com/pycqa/bandit/main/logo/logo.svg" height="100" alt="bandit logo" /></a>
  <a href="https://docs.pytest.org/" target="_blank"><img src="https://raw.githubusercontent.com/pytest-dev/pytest/main/doc/en/img/pytest_logo_curves.svg" height="100" alt="pytest logo" /></a>
</p>

# Power Trading

A trading strategy optimization framework that uses technical indicators and Optuna for hyperparameter optimization.

## Features

- Multiple technical indicators (MACD, RSI, Bollinger Bands, Stochastic)
- Strategy parameter optimization using Optuna
- Backtesting with proper signal generation and return calculation
- Support for multiple instruments (stocks, indices, forex)
- Performance metrics calculation (Sharpe ratio, win ratio, etc.)

## Installation

1. Make sure you have Python 3.8+ and Poetry installed
2. Clone this repository
3. Install dependencies:
```bash
poetry install
```

## Usage

### Running the Optimization Script

```bash
poetry run python scripts/optimize_strategy.py --start 2020-01-01 --end 2023-12-31 --trials 100
```

This will:
1. Download historical data for all configured instruments
2. Run Optuna optimization to find the best strategy parameters
3. Print optimization results and performance metrics

### Running Individual Backtests

```bash
poetry run python scripts/run_bot.py \
    --strategy moving_average \
    --ticker AAPL \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --cash 10000 \
    --fast_period 20 \
    --slow_period 50
```

## Strategy Parameters

### Technical Indicators

- MACD (Moving Average Convergence Divergence)
  - Fast period: 8-20
  - Slow period: 21-40
  - Signal period: 5-15

- RSI (Relative Strength Index)
  - Period: 10-30
  - Oversold threshold: 20-40
  - Overbought threshold: 60-80

- Bollinger Bands
  - Period: 10-30
  - Standard deviation: 1.5-3.0

- Stochastic Oscillator
  - Period: 5-20
  - Smoothing: 2-10
  - Oversold threshold: 10-30
  - Overbought threshold: 70-90

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License

---

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Testing](#testing)  
- [Docker](#docker)  

---

## Prerequisites

- [Python](https://www.python.org/downloads/) **>=3.12, <4.0**  
- [Poetry](https://python-poetry.org/docs/#installation) **for dependency management**  
- [pre-commit](https://pre-commit.com/#install) **for git hooks**  
- [docker](https://docs.docker.com/get-docker/) _(optional)_

---

## Testing

1. Run tests:

   ```bash
   poetry run pytest
   ```

---

## Docker

1. Build the Docker image:

   ```bash
   docker build -t power_trading .
   ```

2. Run the Docker container:

   ```bash
   docker run -p 8000:8000 power_trading
   ```
