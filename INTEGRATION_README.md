# Stock Analyzer Integration

Complete integration of Finviz data download with technical indicator calculations.

## Overview

The `stock_analyzer.py` module combines:
- **finviz.py**: Downloads stock data from Finviz Elite
- **stock_indicators.py**: Calculates technical indicators (RSI, MFI, MACD, MA)

All operations work in **memory** - no CSV files required (though export is available).

## Key Features

### 1. Memory-Based Processing
- Download data directly into pandas DataFrame
- Calculate indicators without saving to disk
- Optional CSV export for results

### 2. Automatic Time Filtering
- Default: Last **1 year** of data (365 days)
- Customizable: Any number of days
- Optimized for recent analysis

### 3. Updated StockIndicators
- **NEW**: Accepts pandas DataFrame directly
- **Still works**: With CSV file paths
- Backward compatible with existing code

## Installation

```bash
pip install pandas numpy pyyaml requests
```

## Quick Start

### Method 1: Quick Analysis

```python
from stock_analyzer import quick_analyze

# Analyze AAPL (last 1 year)
results = quick_analyze('AAPL')

# Custom time period
results = quick_analyze('AAPL', days=180)  # Last 6 months

# With CSV export
results = quick_analyze('AAPL', export=True)
```

### Method 2: StockAnalyzer Class

```python
from stock_analyzer import StockAnalyzer

# Initialize with auth file
analyzer = StockAnalyzer('auth.yaml')

# Download and analyze
results = analyzer.analyze_stock('AAPL', days=365)

# Access results
print(f"Latest RSI: {results['latest']['rsi_14']}")
print(f"Latest Close: ${results['latest']['close']:.2f}")
```

### Method 3: Manual Control

```python
from stock_analyzer import StockAnalyzer

analyzer = StockAnalyzer('auth.yaml')

# Step 1: Download data
df = analyzer.download_stock_data('AAPL', days=365)

# Step 2: Calculate indicators
stock = analyzer.calculate_indicators(df)

# Step 3: Use results
latest = stock.data.iloc[-1]
print(f"RSI: {latest['RSI_14']:.2f}")
```

## Updated StockIndicators Usage

### Accept DataFrame (NEW)

```python
from stock_indicators import StockIndicators
import pandas as pd

# Create or load DataFrame
df = pd.read_csv('data.csv', parse_dates=['Date'])

# Or from memory
df = pd.DataFrame({
    'Date': [...],
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]
})

# Initialize with DataFrame
stock = StockIndicators(df)
stock.calculate_rsi()
stock.calculate_mfi()
stock.calculate_macd()
stock.calculate_ma()
```

### Still Works with CSV

```python
from stock_indicators import StockIndicators

# Initialize with CSV file (backward compatible)
stock = StockIndicators('data.csv')
stock.calculate_rsi()
```

## Configuration

### Authentication File (auth.yaml)

Create a file named `auth.yaml`:

```yaml
auth_token: your_finviz_elite_token_here
```

**Security Note**: Add `auth.yaml` to `.gitignore` to protect your credentials.

## Command Line Usage

Run directly from terminal:

```bash
# Analyze QQQ (default: 365 days)
python stock_analyzer.py QQQ

# Analyze AAPL for last 180 days
python stock_analyzer.py AAPL 180

# Results exported to {TICKER}_indicators.csv
```

## Complete Example

```python
#!/usr/bin/env python3
from stock_analyzer import StockAnalyzer

# Initialize
analyzer = StockAnalyzer('auth.yaml')

# Analyze stock
results = analyzer.analyze_stock(
    stock_name='AAPL',
    days=365,              # Last 1 year
    rsi_period=14,         # RSI parameters
    mfi_period=14,         # MFI parameters
    ma_periods=[20, 50, 200],  # Moving averages
    export_csv='AAPL_analysis.csv'  # Export results
)

if results:
    # Access the StockIndicators object
    stock = results['stock']

    # Access latest values
    latest = results['latest']

    # Access full DataFrame
    df = results['data']

    # Print analysis
    print(f"Ticker: {results['ticker']}")
    print(f"Total Records: {len(df)}")
    print(f"Latest RSI: {latest['rsi_14']:.2f}")
    print(f"Latest MACD: {latest['macd']:.4f}")

    # Further analysis
    if latest['rsi_14'] > 70:
        print("⚠️  Overbought condition")
    elif latest['rsi_14'] < 30:
        print("⚠️  Oversold condition")
```

## Time Period Examples

```python
# Last 30 days (1 month)
results = analyzer.analyze_stock('AAPL', days=30)

# Last 90 days (3 months)
results = analyzer.analyze_stock('AAPL', days=90)

# Last 180 days (6 months)
results = analyzer.analyze_stock('AAPL', days=180)

# Last 365 days (1 year) - DEFAULT
results = analyzer.analyze_stock('AAPL', days=365)

# Last 730 days (2 years)
results = analyzer.analyze_stock('AAPL', days=730)
```

## Results Structure

The `analyze_stock()` method returns a dictionary:

```python
{
    'stock': StockIndicators,      # StockIndicators object
    'latest': dict,                 # Latest indicator values
    'data': pd.DataFrame,           # Full data with indicators
    'ticker': str                   # Stock ticker symbol
}
```

### Latest Values Dictionary

```python
{
    'date': datetime,          # Latest date
    'close': float,            # Close price
    'volume': int,             # Volume
    'rsi_14': float,           # RSI (14-period)
    'mfi_14': float,           # MFI (14-period)
    'macd': float,             # MACD line
    'macd_signal': float,      # MACD signal line
    'macd_histogram': float,   # MACD histogram
    'ma20': float,             # 20-day MA
    'ma50': float,             # 50-day MA
    'ma200': float             # 200-day MA
}
```

## Data Requirements

### Required Columns

Your data must contain:
- **Date**: Trading date
- **Open**: Opening price
- **High**: Highest price
- **Low**: Lowest price
- **Close**: Closing price
- **Volume**: Trading volume

### Data Format

- Date: Any format parseable by pandas
- Prices: Numeric (float)
- Volume: Numeric (int or float)

## Error Handling

```python
from stock_analyzer import StockAnalyzer

analyzer = StockAnalyzer('auth.yaml')

try:
    results = analyzer.analyze_stock('AAPL')

    if results is None:
        print("Analysis failed - check data download")
    else:
        print("Analysis successful")

except FileNotFoundError:
    print("Auth file not found")
except ValueError as e:
    print(f"Invalid data: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

### 1. Limit Time Range
```python
# Faster - analyze only recent data
results = analyzer.analyze_stock('AAPL', days=30)

# Slower - analyze more historical data
results = analyzer.analyze_stock('AAPL', days=730)
```

### 2. Reuse Analyzer
```python
# Good - reuse analyzer instance
analyzer = StockAnalyzer('auth.yaml')
results1 = analyzer.analyze_stock('AAPL')
results2 = analyzer.analyze_stock('GOOGL')
results3 = analyzer.analyze_stock('MSFT')
```

### 3. Batch Processing
```python
analyzer = StockAnalyzer('auth.yaml')
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

analyses = {}
for ticker in tickers:
    print(f"Analyzing {ticker}...")
    analyses[ticker] = analyzer.analyze_stock(ticker, days=365)
```

## Testing

Three test suites are provided:

### 1. Test QQQ Data
```bash
python test_qqq.py
```
Tests all indicators with real QQQ ETF data.

### 2. Test SMA Calculations
```bash
python test_sma.py
```
Validates moving average calculations.

### 3. Test DataFrame Input
```bash
python test_dataframe_input.py
```
Tests new DataFrame input functionality.

## Comparison: Old vs New

### Old Workflow (CSV-based)
```python
# Download to CSV
downloader = StockDownloader('auth.yaml', output_file='data.csv')
downloader.download_stock_detail_data('AAPL')

# Load from CSV
stock = StockIndicators('data.csv')
stock.calculate_rsi()
```

### New Workflow (Memory-based)
```python
# Download and analyze in memory
analyzer = StockAnalyzer('auth.yaml')
results = analyzer.analyze_stock('AAPL')
# No CSV files created!
```

## Benefits of New Approach

1. **Faster**: No disk I/O for intermediate files
2. **Cleaner**: No temporary CSV files
3. **Flexible**: Easy to filter/transform data in memory
4. **Scalable**: Process multiple stocks efficiently
5. **Integrated**: One-step download and analysis

## Backward Compatibility

All existing code still works:

```python
# This still works exactly as before
stock = StockIndicators('mydata.csv')
stock.calculate_rsi()
stock.calculate_mfi()
```

## Notes

### Finviz API
- Requires Finviz Elite subscription
- API structure determines available data
- Check Finviz documentation for data format

### Default Time Period
- Default: 365 days (1 year)
- Balances data quantity vs recency
- Adjust based on your analysis needs

### Data Validation
- Automatic column validation
- Missing data handling
- Type conversion (dates, numbers)

## Troubleshooting

### "Auth file not found"
Create `auth.yaml` with your Finviz token.

### "Missing columns"
Verify Finviz API returns OHLCV data format.

### "No data downloaded"
Check:
1. Finviz Elite subscription is active
2. Auth token is correct
3. Stock ticker is valid

### "Indicators not calculated"
Ensure sufficient historical data:
- RSI: needs 14+ days
- MA50: needs 50+ days
- MA200: needs 200+ days

## See Also

- `STOCK_INDICATORS_README.md` - StockIndicators documentation
- `finviz.py` - Finviz API client documentation
- `stock_indicators.py` - Indicator calculations source code
- `stock_analyzer.py` - Integration source code
