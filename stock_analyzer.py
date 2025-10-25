#!/usr/bin/env python3
"""
Stock Data Downloader and Indicator Calculator

This module integrates Finviz data download with technical indicator calculations.
It downloads stock data from Finviz and calculates technical indicators in memory.

Features:
- Download stock data from Finviz Elite
- Limit data to specified time period (default: 1 year)
- Calculate technical indicators (RSI, MFI, MACD, MA)
- Work entirely in memory without CSV files
- Export results if needed

Usage:
    from stock_analyzer import StockAnalyzer

    # Initialize with auth file
    analyzer = StockAnalyzer('auth.yaml')

    # Download and analyze (default: 1 year of data)
    results = analyzer.analyze_stock('AAPL')

    # Custom time period
    results = analyzer.analyze_stock('AAPL', days=365)
"""

import io
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from finviz import StockDownloader
from stock_indicators import StockIndicators


class StockAnalyzer:
    """
    Integrated stock data downloader and technical indicator calculator.

    Combines Finviz data download with StockIndicators calculations,
    working entirely in memory.
    """

    def __init__(self, auth_file: str):
        """
        Initialize the StockAnalyzer.

        Args:
            auth_file: Path to YAML file containing Finviz authentication token

        Raises:
            FileNotFoundError: If auth_file doesn't exist
            KeyError: If auth_file doesn't contain required keys
        """
        # Initialize the downloader (without output file, returns CSV reader)
        self.downloader = StockDownloader(auth_file, output_file=None)

    def download_stock_data(
        self,
        stock_name: str,
        days: int = 365
    ) -> Optional[pd.DataFrame]:
        """
        Download stock data from Finviz and return as DataFrame.

        Args:
            stock_name: Stock ticker symbol (e.g., 'AAPL', 'QQQ')
            days: Number of days of historical data to retrieve (default: 365)

        Returns:
            DataFrame with stock data, or None if download fails

        Note:
            The Finviz API structure determines what data is available.
            This method filters the data to the specified time period.
        """
        print(f"Downloading {stock_name} data from Finviz...")

        try:
            # Download data using StockDownloader
            csv_reader = self.downloader.download_stock_detail_data(stock_name)

            if csv_reader is None:
                print(f"Failed to download data for {stock_name}")
                return None

            # Convert CSV reader to DataFrame
            df = pd.DataFrame(list(csv_reader))

            if df.empty:
                print(f"No data returned for {stock_name}")
                return None

            print(f"Downloaded {len(df)} records")

            # Check if we have the required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"Warning: Missing columns {missing_cols}")
                print(f"Available columns: {df.columns.tolist()}")
                # Return the data anyway so user can see what's available
                return df

            # Convert Date column to datetime if it exists
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

                # Filter to specified time period
                cutoff_date = datetime.now() - timedelta(days=days)
                df = df[df['Date'] >= cutoff_date]

                print(f"Filtered to {len(df)} records (last {days} days)")

            # Convert numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            print(f"Error downloading data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        rsi_period: int = 14,
        mfi_period: int = 14,
        ma_periods: Optional[list] = None
    ) -> Optional[StockIndicators]:
        """
        Calculate technical indicators on stock data.

        Args:
            data: DataFrame with stock data (Date, Open, High, Low, Close, Volume)
            rsi_period: Period for RSI calculation (default: 14)
            mfi_period: Period for MFI calculation (default: 14)
            ma_periods: Periods for MA calculation (default: [20, 50, 200])

        Returns:
            StockIndicators object with calculated indicators, or None if error
        """
        if ma_periods is None:
            ma_periods = [20, 50, 200]

        try:
            # Initialize StockIndicators with DataFrame (not CSV file)
            stock = StockIndicators(data)

            # Calculate all indicators
            print("\nCalculating technical indicators...")
            stock.calculate_rsi(period=rsi_period)
            print(f"✓ RSI ({rsi_period}-period)")

            stock.calculate_mfi(period=mfi_period)
            print(f"✓ MFI ({mfi_period}-period)")

            stock.calculate_macd()
            print(f"✓ MACD (12, 26, 9)")

            stock.calculate_ma(periods=ma_periods)
            print(f"✓ Moving Averages {ma_periods}")

            return stock

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_stock(
        self,
        stock_name: str,
        days: int = 365,
        rsi_period: int = 14,
        mfi_period: int = 14,
        ma_periods: Optional[list] = None,
        output_csv: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Download stock data and calculate indicators, saving results to CSV.

        Args:
            stock_name: Stock ticker symbol (e.g., 'AAPL')
            days: Number of days of historical data (default: 365)
            rsi_period: Period for RSI calculation (default: 14)
            mfi_period: Period for MFI calculation (default: 14)
            ma_periods: Periods for MA calculation (default: [20, 50, 200])
            output_csv: Path to export results CSV (default: {ticker}_indicators.csv)

        Returns:
            Dictionary containing:
                - 'stock': StockIndicators object
                - 'latest': Latest indicator values
                - 'data': Full DataFrame with indicators
                - 'ticker': Stock ticker symbol
                - 'output_file': Path to CSV file
            Returns None if download or calculation fails
        """
        print(f"Analyzing {stock_name}...")

        # Download data
        df = self.download_stock_data(stock_name, days=days)

        if df is None or df.empty:
            print(f"✗ Failed to download stock data for {stock_name}")
            return None

        # Calculate indicators
        stock = self.calculate_indicators(
            df,
            rsi_period=rsi_period,
            mfi_period=mfi_period,
            ma_periods=ma_periods
        )

        if stock is None:
            print(f"✗ Failed to calculate indicators for {stock_name}")
            return None

        # Get latest values
        latest = stock.get_latest_indicators()

        # Generate output filename if not provided
        if output_csv is None:
            output_csv = f"{stock_name}_indicators.csv"

        # Always export to CSV
        stock.export_to_csv(output_csv)
        print(f"✓ Analysis complete: {len(stock.data)} records with indicators saved to {output_csv}")

        return {
            'stock': stock,
            'latest': latest,
            'data': stock.data,
            'ticker': stock_name,
            'output_file': output_csv
        }


def quick_analyze(
    stock_name: str,
    auth_file: str = 'auth.yaml',
    days: int = 365,
    output_csv: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function for quick stock analysis with CSV export.

    Args:
        stock_name: Stock ticker symbol (e.g., 'AAPL')
        auth_file: Path to authentication file (default: 'auth.yaml')
        days: Number of days of historical data (default: 365)
        output_csv: Output CSV filename (default: {ticker}_indicators.csv)

    Returns:
        Analysis results dictionary or None if failed

    Example:
        >>> results = quick_analyze('AAPL', days=365)
        >>> # Creates AAPL_indicators.csv with all data and indicators
        >>> print(f"Data saved to: {results['output_file']}")
    """
    analyzer = StockAnalyzer(auth_file)

    return analyzer.analyze_stock(
        stock_name,
        days=days,
        output_csv=output_csv
    )


# Example usage
if __name__ == '__main__':
    """
    Example usage of StockAnalyzer.

    Note: Requires valid Finviz Elite authentication file.
    """
    import sys

    # Check if auth file exists
    auth_file = 'auth.yaml'

    if len(sys.argv) > 1:
        stock_ticker = sys.argv[1]
    else:
        stock_ticker = 'QQQ'  # Default to QQQ

    if len(sys.argv) > 2:
        days = int(sys.argv[2])
    else:
        days = 365  # Default to 1 year

    print(f"Analyzing {stock_ticker} for the last {days} days...")
    print(f"Using auth file: {auth_file}")
    print()

    try:
        results = quick_analyze(
            stock_ticker,
            auth_file=auth_file,
            days=days
        )

        if results:
            print(f"\n✓ Analysis completed successfully!")
            print(f"   Total records: {len(results['data'])}")
            print(f"   Output file: {results['output_file']}")
        else:
            print("\n✗ Analysis failed")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nTo use this script:")
        print("1. Create an auth.yaml file with your Finviz Elite token:")
        print("   auth_token: your_token_here")
        print("2. Run: python stock_analyzer.py TICKER [DAYS]")
        print("   Example: python stock_analyzer.py AAPL 365")
