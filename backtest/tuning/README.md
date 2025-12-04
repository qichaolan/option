# Backtest Parameter Tuning Module

A grid search parameter optimization module for backtesting trading strategies.

## Overview

This module provides tools to systematically search through parameter combinations to find optimal strategy configurations. It wraps the existing backtest engine and automatically evaluates different parameter values to identify the best-performing settings.

## Features

- **Grid Search Optimization**: Exhaustive search over all parameter combinations
- **YAML Path-Based Modification**: Modify any numeric value in strategy YAML files
- **Performance Ranking**: Results sorted by `strategy_final_value`
- **Flexible Configuration**: Define parameter ranges in YAML config files
- **CLI and Python API**: Use from command line or import in Python scripts

## Installation

The module is part of the backtest package. Ensure the backtest package is installed:

```bash
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from backtest.tuning import run_parameter_search

result = run_parameter_search(
    data_path="data/SPY_indicators.csv",
    strategy_template_path="strategies/RSI_strategy.yaml",
    param_config_path="config/rsi_params.yaml",
    initial_capital=100000.0,
    top_n=5,
    verbose=True,
)

print(result.summary())
print(f"Best parameters: {result.best_parameters}")
print(f"Best final value: ${result.best_final_value:,.2f}")
```

### Command Line Interface

```bash
# Basic usage
python -m backtest.tuning.optimizer \
    -d data/SPY_indicators.csv \
    -s strategies/RSI_strategy.yaml \
    -p config/rsi_params.yaml

# With options
python -m backtest.tuning.optimizer \
    -d data/SPY_indicators.csv \
    -s strategies/RSI_strategy.yaml \
    -p config/rsi_params.yaml \
    --capital 50000 \
    --top-n 10 \
    --verbose \
    --output results.json
```

## Configuration

### Parameter Configuration File

Create a YAML file defining which parameters to tune:

```yaml
parameters:
  - name: rsi_oversold        # Human-readable name
    path: strategies[0].rules[0].value  # YAML path to the value
    start: 20                 # Minimum value
    end: 40                   # Maximum value
    step: 5                   # Step size

  - name: rsi_overbought
    path: strategies[0].rules[1].value
    start: 60
    end: 80
    step: 5
```

### Strategy Template

Your strategy YAML file serves as the template. The optimizer will modify the values at the specified paths:

```yaml
strategies:
  - name: "RSI_Strategy"
    weight: 1.0
    combine: "all"
    rules:
      - indicator: "rsi_14"
        operator: "<"
        value: 30              # <- This will be tuned (path: strategies[0].rules[0].value)
        action: "buy"
        strength: 1.0
      - indicator: "rsi_14"
        operator: ">"
        value: 70              # <- This will be tuned (path: strategies[0].rules[1].value)
        action: "sell"
        strength: 1.0
```

## YAML Path Syntax

The path syntax supports:

- **Simple keys**: `strategies`, `name`, `value`
- **Array indices**: `rules[0]`, `strategies[1]`
- **Nested paths**: `strategies[0].rules[1].value`

### Examples

| Path | Target |
|------|--------|
| `strategies[0].weight` | Weight of first strategy |
| `strategies[0].rules[0].value` | Value in first rule of first strategy |
| `strategies[1].rules[2].strength` | Strength of third rule in second strategy |

## Output

### Summary Report

```
======================================================================
PARAMETER SEARCH RESULTS
======================================================================

Total combinations evaluated: 25/25

SEARCH SPACE:
  rsi_oversold: 20 to 40 (5 values)
  rsi_overbought: 60 to 80 (5 values)

======================================================================
TOP PERFORMING PARAMETER SETS
======================================================================

Rank #1
----------------------------------------
Parameters:
  rsi_oversold: 25
  rsi_overbought: 75
Final Value: $112,500.00
Total Return: 12.50%
Excess vs Lump Sum: 3.25%
Excess vs DCA: 2.10%
Trades: 15 (Buys: 8, Sells: 7)

...
```

### JSON Output

When using `--output`, results are saved as JSON:

```json
{
  "top_results": [
    {
      "parameters": {"rsi_oversold": 25, "rsi_overbought": 75},
      "strategy_final_value": 112500.0,
      "total_return_pct": 12.5,
      "excess_vs_lumpsum": 3.25,
      "excess_vs_dca": 2.1,
      "num_trades": 15,
      "num_buys": 8,
      "num_sells": 7
    }
  ],
  "total_combinations": 25,
  "evaluated_combinations": 25,
  "best_parameters": {"rsi_oversold": 25, "rsi_overbought": 75},
  "best_final_value": 112500.0,
  "search_space": {
    "rsi_oversold": [20, 25, 30, 35, 40],
    "rsi_overbought": [60, 65, 70, 75, 80]
  }
}
```

## API Reference

### `run_parameter_search()`

Main function to run parameter optimization.

```python
def run_parameter_search(
    data_path: str,
    strategy_template_path: str,
    param_config_path: str,
    initial_capital: float = 100000.0,
    top_n: int = 5,
    verbose: bool = False,
) -> SearchResult:
```

**Parameters:**
- `data_path`: Path to indicator data CSV file
- `strategy_template_path`: Path to base strategy YAML template
- `param_config_path`: Path to parameter configuration YAML
- `initial_capital`: Initial capital for backtest (default: 100000)
- `top_n`: Number of top results to return (default: 5)
- `verbose`: Print progress during search (default: False)

**Returns:**
- `SearchResult` object with optimization results

### `SearchResult`

Result object containing optimization outcomes.

**Attributes:**
- `top_results`: List of top performing `ParameterResult` objects
- `total_combinations`: Total number of parameter combinations
- `evaluated_combinations`: Number successfully evaluated
- `best_parameters`: Dict of best parameter values
- `best_final_value`: Final portfolio value of best combination
- `search_space`: Dict mapping parameter names to their value ranges

**Methods:**
- `summary()`: Returns human-readable summary string
- `to_dict()`: Returns dictionary for JSON serialization

### `ParameterResult`

Result for a single parameter combination.

**Attributes:**
- `parameters`: Dict of parameter name to value
- `strategy_final_value`: Final portfolio value
- `total_return_pct`: Total return percentage
- `excess_vs_lumpsum`: Excess return vs lump sum benchmark
- `excess_vs_dca`: Excess return vs DCA benchmark
- `num_trades`: Total number of trades
- `num_buys`: Number of buy trades
- `num_sells`: Number of sell trades

## Example Use Cases

### Tuning RSI Thresholds

```yaml
# rsi_params.yaml
parameters:
  - name: oversold_threshold
    path: strategies[0].rules[0].value
    start: 20
    end: 35
    step: 5
  - name: overbought_threshold
    path: strategies[0].rules[1].value
    start: 65
    end: 80
    step: 5
```

### Tuning Bollinger Band Parameters

```yaml
# bb_params.yaml
parameters:
  - name: lower_band_multiplier
    path: strategies[0].rules[0].value
    start: 1.5
    end: 2.5
    step: 0.25
  - name: upper_band_multiplier
    path: strategies[0].rules[1].value
    start: 1.5
    end: 2.5
    step: 0.25
```

### Tuning Multiple Strategies

```yaml
# multi_strategy_params.yaml
parameters:
  - name: strategy1_rsi_buy
    path: strategies[0].rules[0].value
    start: 25
    end: 35
    step: 5
  - name: strategy2_mfi_buy
    path: strategies[1].rules[0].value
    start: 20
    end: 30
    step: 5
```

## Best Practices

1. **Start with coarse grids**: Use larger step sizes initially to identify promising regions
2. **Refine promising areas**: Once you find good regions, run finer searches
3. **Watch for overfitting**: Validate best parameters on out-of-sample data
4. **Consider computational cost**: Large parameter spaces can take significant time
5. **Use verbose mode**: Track progress for long-running optimizations

## Limitations

- **Grid search only**: Does not support random search, Bayesian optimization, etc.
- **Numeric values only**: Cannot tune string parameters or strategy structure
- **No parallelization**: Runs backtests sequentially (can be extended)
- **Memory usage**: Stores all results in memory during search

## Testing

Run the test suite:

```bash
# Run all tuning tests
python -m pytest backtest/tuning/ -v

# Run with coverage
python -m pytest backtest/tuning/ --cov=backtest/tuning --cov-report=term-missing
```

## Module Structure

```
backtest/tuning/
├── __init__.py           # Package exports
├── exceptions.py         # Custom exceptions
├── param_config.py       # Parameter configuration parsing
├── yaml_utils.py         # YAML path manipulation utilities
├── optimizer.py          # Main optimization logic and CLI
├── test_param_config.py  # Tests for param_config
├── test_yaml_utils.py    # Tests for yaml_utils
├── test_optimizer.py     # Tests for optimizer
└── README.md            # This file
```
