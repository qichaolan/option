# Stock Technical Indicators Calculator

A Python class for parsing stock CSV data and calculating popular technical indicators.

## Features

The `StockIndicators` class provides four main technical analysis functions:

1. **RSI (Relative Strength Index)** - Measures momentum and identifies overbought/oversold conditions
2. **MFI (Money Flow Index)** - Volume-weighted RSI that combines price and volume
3. **MACD (Moving Average Convergence Divergence)** - Trend-following momentum indicator
4. **MA (Moving Averages)** - Simple moving averages for 20, 50, and 200 periods

## Installation

### Requirements

```bash
pip install pandas numpy
```

## CSV File Format

Your CSV file should contain the following columns:

| Column | Description |
|--------|-------------|
| Date   | Trading date (any format parseable by pandas) |
| Open   | Opening price |
| High   | Highest price of the day |
| Low    | Lowest price of the day |
| Close  | Closing price |
| Volume | Trading volume |

Example:
```csv
Date,Open,High,Low,Close,Volume
2024-01-02,185.50,187.20,184.80,186.50,45000000
2024-01-03,186.80,188.50,186.00,187.80,48000000
```

## Usage

### Basic Usage

```python
from stock_indicators import StockIndicators

# Load stock data from CSV
stock = StockIndicators('your_stock_data.csv')

# Calculate indicators
rsi = stock.calculate_rsi(period=14)
mfi = stock.calculate_mfi(period=14)
macd_data = stock.calculate_macd()
ma_data = stock.calculate_ma(periods=[20, 50, 200])

# View results
print(stock.data.tail())  # Last few rows with all indicators
```

### Quick Analysis

Use the convenience function to calculate all indicators at once:

```python
from stock_indicators import analyze_stock_csv

# Calculate all indicators in one call
stock = analyze_stock_csv(
    'your_stock_data.csv',
    rsi_period=14,
    mfi_period=14,
    ma_periods=[20, 50, 200]
)

# Print summary
print(stock.summary())
```

### Get Latest Values

```python
latest = stock.get_latest_indicators()
print(f"Latest Close: ${latest['close']:.2f}")
print(f"RSI: {latest['rsi_14']:.2f}")
print(f"MFI: {latest['mfi_14']:.2f}")
```

### Export Results

```python
# Export data with all calculated indicators to CSV
stock.export_to_csv('stock_data_with_indicators.csv')
```

## Technical Indicators Explained

### 1. RSI (Relative Strength Index)

**Formula:**
```
RSI = 100 - (100 / (1 + RS))
where RS = Average Gain / Average Loss
```

**Interpretation:**
- RSI > 70: Overbought (potential sell signal)
- RSI < 30: Oversold (potential buy signal)
- RSI around 50: Neutral

**Usage:**
```python
rsi = stock.calculate_rsi(period=14)  # 14 is standard
```

### 2. MFI (Money Flow Index)

**Formula:**
```
1. Typical Price = (High + Low + Close) / 3
2. Money Flow = Typical Price × Volume
3. MFI = 100 - (100 / (1 + Money Flow Ratio))
```

**Interpretation:**
- MFI > 80: Overbought
- MFI < 20: Oversold
- Divergence from price can signal reversals

**Usage:**
```python
mfi = stock.calculate_mfi(period=14)  # 14 is standard
```

### 3. MACD (Moving Average Convergence Divergence)

**Components:**
- **MACD Line**: 12-day EMA - 26-day EMA
- **Signal Line**: 9-day EMA of MACD Line
- **Histogram**: MACD Line - Signal Line

**Interpretation:**
- MACD crosses above Signal: Bullish signal
- MACD crosses below Signal: Bearish signal
- Histogram > 0: Bullish momentum
- Histogram < 0: Bearish momentum

**Usage:**
```python
macd_data = stock.calculate_macd(
    fast_period=12,
    slow_period=26,
    signal_period=9
)
# Returns dict with 'macd', 'signal', and 'histogram'
```

### 4. Moving Averages (MA)

**Types:**
- **MA20**: 20-day simple moving average
- **MA50**: 50-day simple moving average
- **MA200**: 200-day simple moving average

**Interpretation:**
- Price > MA: Uptrend
- Price < MA: Downtrend
- MA crossovers signal trend changes
  - Golden Cross: MA50 crosses above MA200 (bullish)
  - Death Cross: MA50 crosses below MA200 (bearish)

**Usage:**
```python
ma_data = stock.calculate_ma(periods=[20, 50, 200])
# Returns dict: {20: Series, 50: Series, 200: Series}
```

## Complete Example

```python
#!/usr/bin/env python3
from stock_indicators import StockIndicators

# Initialize with CSV file
stock = StockIndicators('AAPL_data.csv')

# Calculate all indicators
print("Calculating technical indicators...")
rsi = stock.calculate_rsi(period=14)
mfi = stock.calculate_mfi(period=14)
macd = stock.calculate_macd()
ma = stock.calculate_ma()

# Get latest values
latest = stock.data.iloc[-1]

print(f"\n=== Latest Indicators for {latest['Date']} ===")
print(f"Close Price: ${latest['Close']:.2f}")
print(f"\nMomentum Indicators:")
print(f"  RSI(14): {latest['RSI_14']:.2f}")
print(f"  MFI(14): {latest['MFI_14']:.2f}")
print(f"\nTrend Indicators:")
print(f"  MACD: {latest['MACD']:.4f}")
print(f"  Signal: {latest['MACD_Signal']:.4f}")
print(f"  Histogram: {latest['MACD_Histogram']:.4f}")
print(f"\nMoving Averages:")
print(f"  MA20: ${latest['MA20']:.2f}")
print(f"  MA50: ${latest['MA50']:.2f}")
print(f"  MA200: ${latest['MA200']:.2f}")

# Analysis
if latest['RSI_14'] > 70:
    print("\n⚠️  RSI indicates overbought conditions")
elif latest['RSI_14'] < 30:
    print("\n⚠️  RSI indicates oversold conditions")

if latest['MACD_Histogram'] > 0:
    print("📈 MACD shows bullish momentum")
else:
    print("📉 MACD shows bearish momentum")

# Export results
stock.export_to_csv('analysis_results.csv')
print("\n✅ Results exported to analysis_results.csv")
```

## Running the Example

A complete example script is provided:

```bash
python example_usage.py
```

This will:
1. Load the sample CSV data
2. Calculate all technical indicators
3. Display the latest values
4. Export results to a new CSV file

## File Structure

```
.
├── stock_indicators.py          # Main class implementation
├── example_usage.py              # Example usage script
├── sample_stock_data.csv         # Sample data for testing
└── STOCK_INDICATORS_README.md    # This file
```

## API Reference

### Class: StockIndicators

#### Constructor
```python
StockIndicators(csv_file: str)
```

#### Methods

**calculate_rsi(period: int = 14, column: str = 'Close') -> pd.Series**
- Calculates Relative Strength Index
- Returns: Series of RSI values (0-100)

**calculate_mfi(period: int = 14) -> pd.Series**
- Calculates Money Flow Index
- Returns: Series of MFI values (0-100)

**calculate_macd(fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = 'Close') -> Dict[str, pd.Series]**
- Calculates MACD indicator
- Returns: Dictionary with 'macd', 'signal', and 'histogram' keys

**calculate_ma(periods: List[int] = [20, 50, 200], column: str = 'Close') -> Dict[int, pd.Series]**
- Calculates Simple Moving Averages
- Returns: Dictionary mapping period to MA Series

**get_latest_indicators() -> Dict[str, Union[float, Dict]]**
- Returns the most recent values of all calculated indicators

**export_to_csv(output_file: str) -> None**
- Exports data with indicators to CSV file

**summary() -> str**
- Generates a text summary of the data and indicators

## Notes

- All indicators are stored in the `data` DataFrame as new columns
- Indicators require sufficient historical data (e.g., MA200 needs at least 200 data points)
- NaN values appear for early periods where calculations aren't possible
- Column names are automatically stripped of whitespace

## Error Handling

The class includes comprehensive error handling:

```python
try:
    stock = StockIndicators('data.csv')
    rsi = stock.calculate_rsi()
except FileNotFoundError:
    print("CSV file not found")
except ValueError as e:
    print(f"Data validation error: {e}")
```

## License

This code is provided as-is for educational and analysis purposes.
