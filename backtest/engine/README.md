# Backtest Engine Module

A production-quality Python module for backtesting trading strategies using technical indicator data.

## Overview

This module provides a clean API and CLI for running backtests on trading strategies defined in YAML files. It evaluates rules against indicator data, simulates a long-only portfolio, and compares performance against standard benchmarks.

### Features

- **YAML Strategy Ingestion**: Load weighted multi-strategy configurations
- **Rule Evaluation Engine**: Evaluate rules with "all"/"any" combination logic
- **Long-Only Portfolio Simulation**: Fractional shares, daily Close price execution
- **Full Trade Logging**: Detailed transaction records with position tracking
- **Benchmark Comparisons**: Cash-Only (5% risk-free), Lump-Sum, Monthly DCA
- **Dual Interface**: Both Python API and command-line interface
- **Production Quality**: Type hints, comprehensive tests (>95% coverage), PEP8 compliant

## Input Requirements

### Indicator CSV File

The indicator CSV must contain:

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Trading date |
| Close | float | Closing price |
| [indicators] | float | Any indicator columns referenced by strategies |

Example:
```csv
Date,Close,rsi_14,macd_hist_12_26_9,bb_lower_20_2,bb_upper_20_2
2023-01-03,380.82,45.2,0.5,375.0,390.0
2023-01-04,383.76,52.1,0.8,376.0,391.0
```

### Strategy YAML File

```yaml
strategies:
  - name: "Strategy_Name"
    weight: 0.5          # Weight for multi-strategy aggregation
    combine: "all"       # "all" or "any" - rule combination mode
    rules:
      - indicator: "rsi_14"
        operator: "<"    # <, <=, >, >=, ==, !=
        value: 30        # Static threshold
        action: "buy"    # "buy" or "sell"
        strength: 1.0    # 0.0 to 1.0

      - indicator: "close"
        operator: ">"
        value_indicator: "bb_upper_20_2"  # Compare to another indicator
        action: "sell"
        strength: 0.8
```

## Signal Generation Logic

1. **Rule Evaluation**: Each rule is evaluated against the data
   - `value`: Compare indicator to static threshold
   - `value_indicator`: Compare indicator to another indicator column

2. **Rule Combination**:
   - `combine: "all"`: All rules of same type (buy/sell) must trigger
   - `combine: "any"`: Average of triggered rules

3. **Strategy Aggregation**: Weighted average of all strategy scores

4. **Signal Generation**:
   - `BUY`: Aggregate score >= +0.3
   - `SELL`: Aggregate score <= -0.3
   - `HOLD`: Otherwise

## Portfolio Simulation

- **Long-only**: Can only hold long positions
- **Execution**: All trades at daily Close price
- **Fractional shares**: Supported
- **BUY signal**: Invest all available cash
- **SELL signal**: Sell entire position

## Benchmarks

| Benchmark | Description |
|-----------|-------------|
| Cash-Only | 5% annual risk-free rate, daily compounded |
| Lump-Sum Buy-and-Hold | Invest all capital on day 1 |
| Monthly DCA | Equal investments every 20 trading days |

## Usage

### Python API

```python
from backtest.engine import run_backtest

# Basic usage
result = run_backtest(
    data_file="SPY_train.csv",
    strategy_files="strategy.yaml",
)

# Print summary
print(result.summary())

# Access detailed results
print(f"Total Return: {result.portfolio.total_return_pct:.2f}%")
print(f"Number of Trades: {result.portfolio.num_trades}")

# Get daily DataFrame
daily_df = result.portfolio.to_dataframe()

# Get trade log
trades_df = result.portfolio.trades_to_dataframe()

# Compare vs benchmarks
print(result.comparison)

# Multiple strategies with custom capital
result = run_backtest(
    data_file="data.csv",
    strategy_files=["strat1.yaml", "strat2.yaml"],
    initial_capital=50000,
    output_file="results.csv",
    trades_file="trades.csv",
)
```

### Command Line Interface

```bash
# Basic usage
python -m backtest.runner --data SPY_train.csv --strategies strategy.yaml

# Short form with multiple strategies
python -m backtest.runner -d data.csv -s strat1.yaml strat2.yaml

# Custom capital and output files
python -m backtest.runner -d data.csv -s strategy.yaml -c 50000 -o results.csv -t trades.csv
```

### CLI Output Example

```
============================================================
BACKTEST RESULTS
============================================================

Data: SPY_train.csv
Date Range: 2023-01-03 to 2023-12-29
Trading Days: 252
Initial Capital: $100,000.00

Strategies Used:
  - RSI_Strategy (weight: 50.00%)
  - MACD_Strategy (weight: 50.00%)

------------------------------------------------------------
PERFORMANCE COMPARISON
------------------------------------------------------------

Trading Strategy               |  $112,345.67 |  +12.35%
Cash-Only (5% Risk-Free)       |  $105,000.00 |   +5.00%
Lump-Sum Buy-and-Hold          |  $115,234.56 |  +15.23%
Monthly DCA                    |  $110,567.89 |  +10.57%

------------------------------------------------------------
TRADING SUMMARY
------------------------------------------------------------

Total Trades: 24
  - Buys: 12
  - Sells: 12

============================================================
```

## Code Structure

```
backtest/engine/
├── __init__.py           # Package exports
├── exceptions.py         # Custom exception classes
├── strategy_loader.py    # YAML parsing and validation
├── rule_engine.py        # Rule evaluation and signal generation
├── portfolio.py          # Portfolio simulation
├── benchmarks.py         # Benchmark calculations
├── runner.py             # Main API and CLI entry point
├── conftest.py           # Pytest fixtures
├── test_*.py             # Test modules
└── README.md             # This file
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `exceptions.py` | Custom exceptions for clear error handling |
| `strategy_loader.py` | Load and validate YAML strategies |
| `rule_engine.py` | Evaluate rules, generate signals |
| `portfolio.py` | Simulate portfolio, log trades |
| `benchmarks.py` | Calculate benchmark comparisons |
| `runner.py` | Main `run_backtest()` API and CLI |

## Running Tests

```bash
# Run all tests
pytest backtest/engine/

# Run with coverage report
pytest backtest/engine/ --cov=backtest/engine --cov-report=term-missing

# Run with branch coverage
pytest backtest/engine/ --cov=backtest/engine --cov-branch --cov-report=term-missing
```

### Coverage Requirements

- Line coverage: >95%
- Branch coverage: >95%

## Error Handling

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | Data or strategy file not found |
| `InvalidStrategyError` | YAML structure invalid |
| `InvalidRuleError` | Rule definition invalid |
| `MissingIndicatorError` | Required indicator not in data |
| `InsufficientDataError` | Not enough data rows |
| `InvalidParameterError` | Invalid function parameter |

```python
from backtest.engine import run_backtest
from backtest.engine.exceptions import (
    InvalidStrategyError,
    MissingIndicatorError,
    BacktestError,
)

try:
    result = run_backtest("data.csv", "strategy.yaml")
except InvalidStrategyError as e:
    print(f"Strategy error: {e}")
except MissingIndicatorError as e:
    print(f"Missing indicators: {e.missing_indicators}")
except BacktestError as e:
    print(f"Backtest failed: {e}")
```

## Design Principles

1. **Pure Functions**: Calculations are stateless and side-effect free
2. **Vectorized Operations**: Uses pandas/numpy for efficiency
3. **No Global State**: All functions receive explicit inputs
4. **Comprehensive Validation**: Fail fast with clear error messages
5. **Type Safety**: Full type hints for public functions
6. **Testability**: Modular design enables unit testing

## License

MIT License
