#!/usr/bin/env python3
"""
Stock Technical Indicators Calculator

A class-based toolkit for parsing stock CSV data and calculating technical indicators.

Implements:
  • RSI (Relative Strength Index)
  • MFI (Money Flow Index)
  • MACD (Moving Average Convergence Divergence)
  • MA (Moving Averages: 20, 50, 200 periods)

Usage Example:
--------------
>>> from stock_indicators import StockIndicators
>>>
>>> # Initialize with CSV file
>>> stock = StockIndicators('AAPL_data.csv')
>>>
>>> # Calculate indicators
>>> rsi = stock.calculate_rsi(period=14)
>>> mfi = stock.calculate_mfi(period=14)
>>> macd_data = stock.calculate_macd()
>>> ma_data = stock.calculate_ma()
>>>
>>> # Access the data
>>> print(stock.data.head())
>>> print(f"Latest RSI: {rsi[-1]}")
"""

import csv
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class StockIndicators:
    """
    A class for parsing stock data and calculating technical indicators.

    Can accept either a CSV file path or a pandas DataFrame directly.
    The data should contain columns: Date, Open, High, Low, Close, Volume

    Attributes:
        data (pd.DataFrame): Parsed stock data with calculated indicators
        source (str): Description of data source (file path or "DataFrame")
    """

    def __init__(self, data_source: Union[str, pd.DataFrame]):
        """
        Initialize the StockIndicators with a CSV file or DataFrame.

        Args:
            data_source: Either a path to CSV file or a pandas DataFrame

        Raises:
            FileNotFoundError: If CSV file path is provided but doesn't exist
            ValueError: If required columns are missing or invalid data type
        """
        if isinstance(data_source, pd.DataFrame):
            self.source = "DataFrame"
            self.data = self._prepare_dataframe(data_source)
        elif isinstance(data_source, str):
            self.source = data_source
            self.data = self._load_csv(data_source)
        else:
            raise ValueError("data_source must be either a file path (str) or pandas DataFrame")

        self._validate_data()

    def _load_csv(self, csv_file: str) -> pd.DataFrame:
        """
        Load stock data from CSV file.

        Args:
            csv_file: Path to CSV file

        Returns:
            DataFrame containing the stock data

        Raises:
            FileNotFoundError: If the CSV file does not exist
        """
        if not Path(csv_file).exists():
            raise FileNotFoundError(f"CSV file '{csv_file}' not found.")

        try:
            # Read CSV and parse dates
            df = pd.read_csv(csv_file, parse_dates=['Date'])

            # Strip whitespace from column names
            df.columns = df.columns.str.strip()

            # Sort by date (oldest to newest)
            df = df.sort_values('Date').reset_index(drop=True)

            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare a DataFrame for analysis (clean and validate).

        Args:
            df: Input DataFrame

        Returns:
            Cleaned and sorted DataFrame
        """
        # Create a copy to avoid modifying the original
        df = df.copy()

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Ensure Date column is datetime
        if 'Date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])

        # Sort by date (oldest to newest)
        if 'Date' in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)

        return df

    def _validate_data(self) -> None:
        """
        Validate that required columns exist in the data.

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Ensure numeric columns are numeric
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Check for NaN values
        if self.data[numeric_columns].isnull().any().any():
            print("Warning: Some numeric values could not be parsed and were set to NaN.")

    def calculate_rsi(self, period: int = 14, column: str = 'Close') -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI) using Wilder's smoothing method.

        RSI measures the magnitude of recent price changes to evaluate
        overbought or oversold conditions.

        Formula:
            RSI = 100 - (100 / (1 + RS))
            where RS = Average Gain / Average Loss over the period

        Uses Wilder's smoothing:
            - First average: Simple mean of gains/losses over period
            - Subsequent values: (Previous average * (period-1) + Current value) / period

        Args:
            period: Number of periods for RSI calculation (default: 14)
            column: Column to calculate RSI on (default: 'Close')

        Returns:
            Series containing RSI values (0-100)

        Raises:
            ValueError: If period is less than 1 or column doesn't exist
        """
        if period < 1:
            raise ValueError("Period must be at least 1.")
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        # Calculate price changes
        delta = self.data[column].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Use Wilder's smoothing method (RMA - Rolling Moving Average)
        # This is equivalent to EMA with alpha = 1/period
        # ewm(alpha=1/period) is the correct method for RSI, not ewm(span=period)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Store in dataframe
        self.data[f'RSI_{period}'] = rsi

        return rsi

    def calculate_mfi(self, period: int = 14) -> pd.Series:
        """
        Calculate the Money Flow Index (MFI).

        MFI is a momentum indicator that uses price and volume data to identify
        overbought or oversold conditions. It's also known as volume-weighted RSI.

        Formula:
            1. Typical Price = (High + Low + Close) / 3
            2. Raw Money Flow = Typical Price × Volume
            3. Money Flow Ratio = (Positive Money Flow) / (Negative Money Flow)
            4. MFI = 100 - (100 / (1 + Money Flow Ratio))

        Args:
            period: Number of periods for MFI calculation (default: 14)

        Returns:
            Series containing MFI values (0-100)

        Raises:
            ValueError: If period is less than 1
        """
        if period < 1:
            raise ValueError("Period must be at least 1.")

        # Calculate Typical Price
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3.0

        # Calculate Raw Money Flow
        money_flow = typical_price * self.data['Volume']

        # Identify positive and negative money flow
        price_diff = typical_price.diff()

        positive_flow = money_flow.where(price_diff > 0, 0.0)
        negative_flow = money_flow.where(price_diff < 0, 0.0)

        # Calculate the sum of positive and negative money flow over the period
        positive_mf_sum = positive_flow.rolling(window=period).sum()
        negative_mf_sum = negative_flow.rolling(window=period).sum()

        # Calculate Money Flow Ratio
        mf_ratio = positive_mf_sum / negative_mf_sum

        # Calculate MFI
        mfi = 100.0 - (100.0 / (1.0 + mf_ratio))

        # Store in dataframe
        self.data[f'MFI_{period}'] = mfi

        return mfi

    def calculate_macd(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = 'Close'
    ) -> Dict[str, pd.Series]:
        """
        Calculate the Moving Average Convergence Divergence (MACD).

        MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of a security's price.

        Components:
            - MACD Line: Fast EMA - Slow EMA
            - Signal Line: EMA of MACD Line
            - Histogram: MACD Line - Signal Line

        Args:
            fast_period: Period for fast EMA (default: 12)
            slow_period: Period for slow EMA (default: 26)
            signal_period: Period for signal line EMA (default: 9)
            column: Column to calculate MACD on (default: 'Close')

        Returns:
            Dictionary containing 'macd', 'signal', and 'histogram' Series

        Raises:
            ValueError: If periods are invalid or column doesn't exist
        """
        if fast_period < 1 or slow_period < 1 or signal_period < 1:
            raise ValueError("All periods must be at least 1.")
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period.")
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        # Calculate EMAs
        ema_fast = self.data[column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = self.data[column].ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate Signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate Histogram
        histogram = macd_line - signal_line

        # Store in dataframe
        self.data['MACD'] = macd_line
        self.data['MACD_Signal'] = signal_line
        self.data['MACD_Histogram'] = histogram

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_ma(
        self,
        periods: Optional[List[int]] = None,
        column: str = 'Close'
    ) -> Dict[int, pd.Series]:
        """
        Calculate Simple Moving Averages (SMA) for specified periods.

        By default, calculates MA20, MA50, and MA200.

        A moving average smooths price data by creating a constantly updated
        average price over a specific period.

        Args:
            periods: List of periods to calculate (default: [20, 50, 200])
            column: Column to calculate MA on (default: 'Close')

        Returns:
            Dictionary mapping period to MA Series

        Raises:
            ValueError: If any period is less than 1 or column doesn't exist
        """
        if periods is None:
            periods = [20, 50, 200]

        if any(p < 1 for p in periods):
            raise ValueError("All periods must be at least 1.")
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        result = {}

        for period in periods:
            ma = self.data[column].rolling(window=period).mean()
            self.data[f'MA{period}'] = ma
            result[period] = ma

        return result

    def get_latest_indicators(self) -> Dict[str, Union[float, Dict]]:
        """
        Get the most recent values of all calculated indicators.

        Returns:
            Dictionary containing the latest values of all indicators
        """
        latest = {}

        # Get the last row
        last_row = self.data.iloc[-1]

        # Include basic price data
        latest['date'] = last_row['Date']
        latest['close'] = last_row['Close']
        latest['volume'] = last_row['Volume']

        # Include all calculated indicators
        for col in self.data.columns:
            if col in ['RSI_14', 'MFI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                       'MA20', 'MA50', 'MA200']:
                latest[col.lower()] = last_row[col] if not pd.isna(last_row[col]) else None

        return latest

    def export_to_csv(self, output_file: str) -> None:
        """
        Export the data with calculated indicators to a CSV file.

        Args:
            output_file: Path to save the CSV file
        """
        self.data.to_csv(output_file, index=False)
        print(f"Data exported to {output_file}")

    def summary(self) -> str:
        """
        Generate a summary of the stock data and indicators.

        Returns:
            String containing summary information
        """
        summary_lines = [
            f"Stock Data Summary",
            f"=" * 50,
            f"CSV File: {self.csv_file}",
            f"Total Records: {len(self.data)}",
            f"Date Range: {self.data['Date'].min()} to {self.data['Date'].max()}",
            f"",
            f"Latest Values:",
            f"-" * 50,
        ]

        latest = self.get_latest_indicators()
        for key, value in latest.items():
            if value is not None and key != 'date':
                if isinstance(value, float):
                    summary_lines.append(f"  {key.upper()}: {value:.2f}")
                else:
                    summary_lines.append(f"  {key.upper()}: {value}")

        return "\n".join(summary_lines)


# Convenience function for quick usage
def analyze_stock_csv(
    csv_file: str,
    rsi_period: int = 14,
    mfi_period: int = 14,
    ma_periods: Optional[List[int]] = None
) -> StockIndicators:
    """
    Convenience function to load CSV and calculate all indicators at once.

    Args:
        csv_file: Path to the CSV file
        rsi_period: Period for RSI calculation (default: 14)
        mfi_period: Period for MFI calculation (default: 14)
        ma_periods: Periods for MA calculation (default: [20, 50, 200])

    Returns:
        StockIndicators object with all indicators calculated
    """
    stock = StockIndicators(csv_file)
    stock.calculate_rsi(period=rsi_period)
    stock.calculate_mfi(period=mfi_period)
    stock.calculate_macd()
    stock.calculate_ma(periods=ma_periods)

    return stock
