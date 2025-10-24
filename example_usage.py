#!/usr/bin/env python3
"""
Example usage of the StockIndicators class.

This script demonstrates how to use the StockIndicators class to:
1. Load stock data from a CSV file
2. Calculate technical indicators (RSI, MFI, MACD, MA)
3. Display the results
"""

from stock_indicators import StockIndicators, analyze_stock_csv


def main():
    """Main function demonstrating StockIndicators usage."""

    # Example 1: Basic usage with all indicators
    print("=" * 60)
    print("Example 1: Loading and analyzing stock data")
    print("=" * 60)

    # Replace 'sample_stock_data.csv' with your actual CSV file
    csv_file = 'sample_stock_data.csv'

    try:
        # Initialize the class
        stock = StockIndicators(csv_file)

        print(f"\nLoaded {len(stock.data)} records from {csv_file}")
        print(f"Date range: {stock.data['Date'].min()} to {stock.data['Date'].max()}\n")

        # Calculate all indicators
        print("Calculating technical indicators...")

        # 1. Calculate RSI (14-period)
        rsi = stock.calculate_rsi(period=14)
        print(f"✓ RSI calculated (14-period)")

        # 2. Calculate MFI (14-period)
        mfi = stock.calculate_mfi(period=14)
        print(f"✓ MFI calculated (14-period)")

        # 3. Calculate MACD
        macd_data = stock.calculate_macd()
        print(f"✓ MACD calculated (12, 26, 9)")

        # 4. Calculate Moving Averages (20, 50, 200)
        ma_data = stock.calculate_ma(periods=[20, 50, 200])
        print(f"✓ Moving Averages calculated (20, 50, 200)\n")

        # Display latest values
        print("=" * 60)
        print("Latest Indicator Values:")
        print("=" * 60)

        latest = stock.data.iloc[-1]
        print(f"Date: {latest['Date']}")
        print(f"Close Price: ${latest['Close']:.2f}")
        print(f"Volume: {latest['Volume']:,.0f}")
        print()

        if not pd.isna(latest['RSI_14']):
            print(f"RSI (14): {latest['RSI_14']:.2f}")
            if latest['RSI_14'] > 70:
                print("  → Overbought condition")
            elif latest['RSI_14'] < 30:
                print("  → Oversold condition")

        if not pd.isna(latest['MFI_14']):
            print(f"MFI (14): {latest['MFI_14']:.2f}")
            if latest['MFI_14'] > 80:
                print("  → Overbought condition")
            elif latest['MFI_14'] < 20:
                print("  → Oversold condition")

        print()
        if not pd.isna(latest['MACD']):
            print(f"MACD: {latest['MACD']:.4f}")
            print(f"Signal: {latest['MACD_Signal']:.4f}")
            print(f"Histogram: {latest['MACD_Histogram']:.4f}")

        print()
        if not pd.isna(latest['MA20']):
            print(f"MA20: ${latest['MA20']:.2f}")
        if not pd.isna(latest['MA50']):
            print(f"MA50: ${latest['MA50']:.2f}")
        if not pd.isna(latest['MA200']):
            print(f"MA200: ${latest['MA200']:.2f}")

        # Export to CSV
        print("\n" + "=" * 60)
        output_file = 'stock_data_with_indicators.csv'
        stock.export_to_csv(output_file)

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        print("\nTo use this example:")
        print("1. Create a CSV file named 'sample_stock_data.csv'")
        print("2. Include columns: Date, Open, High, Low, Close, Volume")
        print("3. Run this script again")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Quick analysis using convenience function
    print("\n\n" + "=" * 60)
    print("Example 2: Using convenience function")
    print("=" * 60)

    try:
        # This calculates all indicators in one call
        stock_quick = analyze_stock_csv(
            csv_file,
            rsi_period=14,
            mfi_period=14,
            ma_periods=[20, 50, 200]
        )

        print("\nAll indicators calculated!")
        print(stock_quick.summary())

    except FileNotFoundError:
        print(f"Skipping - CSV file not found")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    # Import pandas here for the display
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    main()
