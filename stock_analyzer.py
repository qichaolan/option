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
- Export results to CSV or return as DataFrame
- Two analysis modes: save to CSV or work with DataFrame in memory

Usage:
    from stock_analyzer import StockAnalyzer

    # Initialize with Finviz auth file
    analyzer = StockAnalyzer('finviz_auth.yaml')

    # Option 1: Analyze and save to CSV (default behavior)
    results = analyzer.analyze_stock('AAPL')
    print(f"Data saved to: {results['output_file']}")

    # Option 2: Analyze and return DataFrame (no CSV)
    results = analyzer.analyze_stock_to_dataframe('AAPL', days=365)
    df = results['data']  # Get the DataFrame for further processing
    print(df.head())
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

    def __init__(self, finviz_auth_file: str):
        """
        Initialize the StockAnalyzer.

        Args:
            finviz_auth_file: Path to YAML file containing Finviz Elite authentication token

        Raises:
            FileNotFoundError: If finviz_auth_file doesn't exist
            KeyError: If finviz_auth_file doesn't contain required keys
        """
        # Initialize the downloader (without output file, returns CSV reader)
        self.downloader = StockDownloader(finviz_auth_file, output_file=None)

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

            # Check if response contains HTML (error page) instead of CSV data
            if '<!DOCTYPE html>' in df.columns or '<html>' in str(df.columns).lower():
                print(f"\n✗ Error: Received HTML response instead of CSV data")
                print(f"   This usually indicates an authentication or API access issue:")
                print(f"   - Check that your Finviz Elite authentication token is valid")
                print(f"   - Verify that your Finviz Elite subscription is active")
                print(f"   - Ensure the stock ticker '{stock_name}' is valid")
                print(f"\n   Response preview: {df.columns.tolist()[:3]}")
                return None

            print(f"Downloaded {len(df)} records")

            # Check if we have the required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"\n✗ Error: Missing required columns {missing_cols}")
                print(f"   Available columns: {df.columns.tolist()}")
                print(f"   This may indicate the API endpoint is not returning stock detail data")
                return None

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
        output_csv: Optional[str] = None,
        use_ai: bool = False,
        ai_config: str = 'openai_config.yaml',
        ai_prompts_folder: str = 'prompts',
        ai_prompt_name: str = 'default_analysis'
    ) -> Optional[Dict[str, Any]]:
        """
        Download stock data and calculate indicators, saving results to CSV.
        Optionally analyze with OpenAI.

        Args:
            stock_name: Stock ticker symbol (e.g., 'AAPL')
            days: Number of days of historical data (default: 365)
            rsi_period: Period for RSI calculation (default: 14)
            mfi_period: Period for MFI calculation (default: 14)
            ma_periods: Periods for MA calculation (default: [20, 50, 200])
            output_csv: Path to export results CSV (default: {ticker}_indicators.csv)
            use_ai: Whether to use OpenAI for analysis (default: False)
            ai_config: Path to OpenAI config file (default: openai_config.yaml)
            ai_prompts_folder: Path to prompts folder (default: prompts)
            ai_prompt_name: Name of AI analysis prompt file (default: default_analysis)

        Returns:
            Dictionary containing:
                - 'stock': StockIndicators object
                - 'latest': Latest indicator values
                - 'data': Full DataFrame with indicators
                - 'ticker': Stock ticker symbol
                - 'output_file': Path to CSV file
                - 'ai_analysis': AI analysis text (if use_ai=True)
                - 'ai_analysis_file': Path to AI analysis file (if use_ai=True)
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

        # Prepare results
        results = {
            'stock': stock,
            'latest': latest,
            'data': stock.data,
            'ticker': stock_name,
            'output_file': output_csv
        }

        # Optional AI analysis
        if use_ai:
            try:
                from openai_analyzer import OpenAIAnalyzer

                print("\n" + "=" * 70)
                print("AI Analysis (OpenAI)")
                print("=" * 70)

                # Initialize AI analyzer
                ai_analyzer = OpenAIAnalyzer(ai_config, ai_prompts_folder)

                # Perform analysis
                ai_analysis = ai_analyzer.analyze_stock_indicators(
                    ticker=stock_name,
                    latest_data=latest,
                    prompt_name=ai_prompt_name
                )

                if ai_analysis:
                    # Save analysis to file
                    ai_file = ai_analyzer.save_analysis(ai_analysis, stock_name)

                    # Add to results
                    results['ai_analysis'] = ai_analysis
                    results['ai_analysis_file'] = ai_file

                    # Display summary
                    print("\n" + "=" * 70)
                    print("AI Analysis Preview:")
                    print("=" * 70)
                    # Show first 500 characters
                    preview = ai_analysis[:500] + "..." if len(ai_analysis) > 500 else ai_analysis
                    print(preview)
                    print("\n" + "=" * 70)
                else:
                    print("✗ AI analysis failed")

            except ImportError:
                print("\n✗ OpenAI analyzer not available. Install with: pip install openai")
            except FileNotFoundError as e:
                print(f"\n✗ {e}")
            except Exception as e:
                print(f"\n✗ AI analysis error: {e}")

        return results

    def analyze_stock_to_dataframe(
        self,
        stock_name: str,
        days: int = 365,
        rsi_period: int = 14,
        mfi_period: int = 14,
        ma_periods: Optional[list] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Download stock data and calculate indicators, returning results as DataFrame.

        This method is similar to analyze_stock but does NOT save to CSV.
        It returns the data in memory as a DataFrame for further processing.

        Args:
            stock_name: Stock ticker symbol (e.g., 'AAPL')
            days: Number of days of historical data (default: 365)
            rsi_period: Period for RSI calculation (default: 14)
            mfi_period: Period for MFI calculation (default: 14)
            ma_periods: Periods for MA calculation (default: [20, 50, 200])

        Returns:
            Dictionary containing:
                - 'stock': StockIndicators object
                - 'latest': Latest indicator values
                - 'data': Full DataFrame with indicators
                - 'ticker': Stock ticker symbol
            Returns None if download or calculation fails
        """
        print(f"Analyzing {stock_name} (in-memory)...")

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

        print(f"✓ Analysis complete: {len(stock.data)} records with indicators in DataFrame")

        # Return results (no CSV export)
        return {
            'stock': stock,
            'latest': latest,
            'data': stock.data,
            'ticker': stock_name
        }


def quick_analyze(
    stock_name: str,
    finviz_auth_file: str = 'finviz_auth.yaml',
    days: int = 365,
    output_csv: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function for quick stock analysis with CSV export.

    Args:
        stock_name: Stock ticker symbol (e.g., 'AAPL')
        finviz_auth_file: Path to Finviz authentication file (default: 'finviz_auth.yaml')
        days: Number of days of historical data (default: 365)
        output_csv: Output CSV filename (default: {ticker}_indicators.csv)

    Returns:
        Analysis results dictionary or None if failed

    Example:
        >>> results = quick_analyze('AAPL', days=365)
        >>> # Creates AAPL_indicators.csv with all data and indicators
        >>> print(f"Data saved to: {results['output_file']}")
    """
    analyzer = StockAnalyzer(finviz_auth_file)

    return analyzer.analyze_stock(
        stock_name,
        days=days,
        output_csv=output_csv
    )


def quick_analyze_to_dataframe(
    stock_name: str,
    finviz_auth_file: str = 'finviz_auth.yaml',
    days: int = 365
) -> Optional[Dict[str, Any]]:
    """
    Convenience function for quick stock analysis returning DataFrame.

    This function does NOT save to CSV - it returns the data in memory.

    Args:
        stock_name: Stock ticker symbol (e.g., 'AAPL')
        finviz_auth_file: Path to Finviz authentication file (default: 'finviz_auth.yaml')
        days: Number of days of historical data (default: 365)

    Returns:
        Analysis results dictionary or None if failed

    Example:
        >>> results = quick_analyze_to_dataframe('AAPL', days=365)
        >>> df = results['data']  # Get the DataFrame
        >>> print(df.head())
        >>> print(f"Latest RSI: {results['latest']['rsi_14']}")
    """
    analyzer = StockAnalyzer(finviz_auth_file)

    return analyzer.analyze_stock_to_dataframe(
        stock_name,
        days=days
    )


# Example usage
if __name__ == '__main__':
    """
    Command-line interface for StockAnalyzer.
    """
    import sys
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Download stock data from Finviz and calculate technical indicators',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s AAPL                                  # Analyze AAPL with defaults
  %(prog)s AAPL --days 180                       # Analyze AAPL for last 180 days
  %(prog)s AAPL --finviz-auth custom_finviz.yaml # Use custom Finviz auth file
  %(prog)s QQQ --days 90 --finviz-auth my_auth.yaml --output qqq.csv
        '''
    )

    parser.add_argument(
        'ticker',
        type=str,
        help='Stock ticker symbol (e.g., AAPL, QQQ, MSFT)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days of historical data (default: 365)'
    )

    parser.add_argument(
        '--finviz-auth',
        type=str,
        default='finviz_auth.yaml',
        metavar='FILE',
        dest='finviz_auth_file',
        help='Path to Finviz authentication YAML file (default: finviz_auth.yaml)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        metavar='FILE',
        help='Output CSV filename (default: {TICKER}_indicators.csv)'
    )

    parser.add_argument(
        '--ai',
        action='store_true',
        help='Enable OpenAI analysis of indicators (DEPRECATED - use openai_analyzer.py instead)'
    )

    parser.add_argument(
        '--ai-config',
        type=str,
        default='openai_config.yaml',
        metavar='FILE',
        help='Path to OpenAI config file (default: openai_config.yaml)'
    )

    parser.add_argument(
        '--ai-prompt',
        type=str,
        default='default_analysis',
        metavar='TYPE',
        help='Type of AI analysis prompt (default: default_analysis)'
    )

    args = parser.parse_args()

    # Display configuration
    print("Stock Analyzer")
    print("=" * 70)
    print(f"Ticker: {args.ticker}")
    print(f"Period: {args.days} days")
    print(f"Finviz auth: {args.finviz_auth_file}")
    if args.output:
        print(f"Output file: {args.output}")
    else:
        print(f"Output file: {args.ticker}_indicators.csv (auto-generated)")
    if args.ai:
        print(f"AI Analysis: Enabled (DEPRECATED)")
        print(f"AI Config: {args.ai_config}")
        print(f"AI Prompt: {args.ai_prompt}")
    print("=" * 70)
    print()

    try:
        analyzer = StockAnalyzer(args.finviz_auth_file)

        results = analyzer.analyze_stock(
            stock_name=args.ticker,
            days=args.days,
            output_csv=args.output,
            use_ai=args.ai,
            ai_config=args.ai_config,
            ai_prompt_name=args.ai_prompt
        )

        if results:
            print(f"\n✓ Analysis completed successfully!")
            print(f"   Total records: {len(results['data'])}")
            print(f"   Output file: {results['output_file']}")

            if 'ai_analysis_file' in results:
                print(f"   AI Analysis file: {results['ai_analysis_file']}")
        else:
            print("\n✗ Analysis failed")
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nMake sure '{args.finviz_auth_file}' exists with your Finviz Elite token:")
        print("   auth_token: your_token_here")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
