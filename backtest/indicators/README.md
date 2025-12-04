# Stock Indicator Generation Module

A production-quality Python module for calculating technical indicators from OHLCV (Open, High, Low, Close, Volume) stock data.

## Overview

This module provides a clean API and CLI for loading stock price data from CSV files and calculating a curated set of modern, high-value technical indicators proven effective in quantitative trading.

### Features

- **Reliable CSV Loading**: Handles date parsing, column name cleaning, and chronological sorting
- **Comprehensive Validation**: Detects missing columns, invalid data types, duplicate dates, and data quality issues
- **Modern Indicators**: Focus on high-value signals used in contemporary quant strategies
- **Train/Test Split**: Walk-forward style data splitting for backtesting and ML workflows
- **Dual Interface**: Both Python API and command-line interface
- **Production Quality**: Type hints, comprehensive tests (>95% coverage), PEP8 compliant

## Input Schema

The input CSV file must contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Trading date (various formats supported) |
| Open | float | Opening price |
| High | float | High price |
| Low | float | Low price |
| Close | float | Closing price |
| Volume | int/float | Trading volume |

Additional columns (e.g., `Adj Close`) are preserved but not required.

### Example Input (SPY.csv)

```csv
Date,Open,High,Low,Close,Adj Close,Volume
2023-01-03,384.37,386.43,377.83,380.82,380.82,75089200
2023-01-04,383.18,385.88,380.00,383.76,383.76,70783900
2023-01-05,381.72,381.84,378.76,379.38,379.38,71935600
```

## Indicators Calculated

### Trend / Momentum

| Indicator | Description |
|-----------|-------------|
| sma_5, sma_9, sma_20, sma_50, sma_200 | Simple Moving Averages |
| ema_9, ema_21, ema_50 | Exponential Moving Averages |
| macd_12_26_9 | MACD Line (12-day EMA - 26-day EMA) |
| macd_signal_12_26_9 | MACD Signal Line (9-day EMA of MACD) |
| macd_hist_12_26_9 | MACD Histogram (MACD - Signal) |
| rsi_14 | Relative Strength Index (14-period, Wilder's smoothing) |
| mfi_14 | Money Flow Index (volume-weighted RSI) |

### Volatility

| Indicator | Description |
|-----------|-------------|
| atr_14 | Average True Range (14-period, Wilder's smoothing) |
| hv_20 | Historical Volatility (20-day annualized realized volatility) |

### Volume-Based

| Indicator | Description |
|-----------|-------------|
| obv | On-Balance Volume |
| vol_sma_20 | 20-day Volume Simple Moving Average |

### Market Structure

| Indicator | Description |
|-----------|-------------|
| pivot_high_3 | Pivot High (3 bars left/right confirmation) |
| pivot_low_3 | Pivot Low (3 bars left/right confirmation) |
| bb_mid_20_2 | Bollinger Band Middle (20-day SMA) |
| bb_upper_20_2 | Bollinger Band Upper (middle + 2 std dev) |
| bb_lower_20_2 | Bollinger Band Lower (middle - 2 std dev) |

## Usage

### Python API

```python
from backtest import build_indicators

# Load and calculate indicators
df = build_indicators("SPY.csv")

# Access indicator values
print(df[["Date", "Close", "rsi_14", "macd_12_26_9"]].tail())

# Save to file
df = build_indicators("SPY.csv", output_file="SPY_indicators.csv")

# With train/test split (last 60 days as test set)
train_df, test_df = build_indicators("SPY.csv", test_days=60)

# Split with output files (creates SPY_train.csv and SPY_test.csv)
train_df, test_df = build_indicators("SPY.csv", output_file="SPY.csv", test_days=60)
```

### Command Line Interface

```bash
# Display summary and latest values
python -m backtest.indicators --input_file SPY.csv

# Save indicators to output file
python -m backtest.indicators --input_file SPY.csv --output_file SPY_ind.csv

# Short form
python -m backtest.indicators -i SPY.csv -o SPY_ind.csv

# With train/test split (creates SPY_train.csv and SPY_test.csv)
python -m backtest.indicators -i SPY.csv -o SPY --test 60
```

### CLI Output Example

**Without split:**


```
Processed 252 rows from SPY.csv
Date range: 2023-01-03 to 2023-12-29
Indicators added: 22

Latest indicator values:
  sma_5: 475.2340
  sma_9: 473.1256
  ema_9: 474.5678
  rsi_14: 62.3456
  ...
```

**With train/test split:**

```
Processed 252 rows from SPY.csv
Train set: 192 rows
  Date range: 2023-01-03 to 2023-10-06
Test set: 60 rows
  Date range: 2023-10-09 to 2023-12-29
Indicators added: 22
Output saved to: SPY_train.csv
Output saved to: SPY_test.csv
```

## Installation

```bash
# Install dependencies
pip install pandas numpy pytest pytest-cov

# Or from requirements
pip install -r requirements.txt
```

## Running Tests

```bash
# Run all tests
pytest backtest/

# Run with coverage report
pytest backtest/ --cov=backtest --cov-report=term-missing

# Run with branch coverage
pytest backtest/ --cov=backtest --cov-branch --cov-report=term-missing

# Generate HTML coverage report
pytest backtest/ --cov=backtest --cov-branch --cov-report=html
```

### Coverage Requirements

- Line coverage: >95%
- Branch coverage: >95%

## Code Structure

```
backtest/
├── __init__.py               # Package exports
└── indicators/
    ├── __init__.py           # Subpackage exports
    ├── exceptions.py         # Custom exception classes
    ├── loader.py             # CSV loading and parsing
    ├── validators.py         # Data validation functions
    ├── calculations.py       # Technical indicator calculations
    ├── main.py               # Main API and CLI entry point
    ├── conftest.py           # Pytest fixtures
    ├── test_loader.py        # Loader unit tests
    ├── test_validators.py    # Validator unit tests
    ├── test_calculations.py  # Calculation unit tests
    ├── test_main.py          # API and CLI integration tests
    └── README.md             # This file
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `exceptions.py` | Custom exceptions for clear error handling |
| `loader.py` | CSV file loading, date parsing, sorting |
| `validators.py` | Data quality validation rules |
| `calculations.py` | Pure functions for indicator calculations |
| `main.py` | Main `build_indicators()` API and CLI |

## Train/Test Split

The module supports walk-forward style train/test splitting for backtesting and machine learning workflows:

- **Walk-forward split**: The last N days go into the test set, all prior data into the train set
- **Indicators calculated first**: All indicators are computed on the FULL dataset before splitting, ensuring test set has valid indicator values (no look-ahead bias in the split, but proper indicator warm-up)
- **Automatic file naming**: When output file is specified, creates `{name}_train.csv` and `{name}_test.csv`

```python
# Split last 60 days into test set
train_df, test_df = build_indicators("SPY.csv", test_days=60)

print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
```

## Design Principles

1. **Pure Functions**: All indicator calculations are stateless and side-effect free
2. **Vectorized Operations**: Uses pandas/numpy for efficient large dataset processing
3. **No Global State**: All functions receive explicit inputs
4. **Comprehensive Validation**: Fail fast with clear error messages
5. **Type Safety**: Full type hints for all public functions
6. **Testability**: Modular design enables unit testing at each layer

## Error Handling

The module raises specific exceptions for different error conditions:

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | Input file does not exist |
| `EmptyFileError` | Input file is empty |
| `MissingColumnError` | Required OHLCV column missing |
| `InvalidDataTypeError` | Column contains non-numeric values |
| `DuplicateDateError` | Duplicate trading dates found |
| `NonMonotonicDateError` | Dates not in chronological order |

```python
from backtest import build_indicators
from backtest.indicators.exceptions import MissingColumnError, ValidationError

try:
    df = build_indicators("data.csv")
except MissingColumnError as e:
    print(f"Missing columns: {e.missing_columns}")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## License

MIT License
