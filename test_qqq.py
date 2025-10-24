#!/usr/bin/env python3
"""
Test script for StockIndicators class using QQQ data from data/qqq.csv
"""

import pandas as pd
from stock_indicators import StockIndicators, analyze_stock_csv
from datetime import datetime, timedelta

def convert_excel_date(excel_date):
    """Convert Excel serial date to datetime."""
    try:
        # Excel epoch starts on 1899-12-30
        base_date = datetime(1899, 12, 30)
        return base_date + timedelta(days=int(excel_date))
    except:
        return excel_date

def main():
    print("=" * 70)
    print("Testing StockIndicators with QQQ Data")
    print("=" * 70)

    csv_file = 'data/qqq.csv'

    try:
        # First, let's fix the date format in the CSV
        print(f"\nLoading {csv_file}...")
        df = pd.read_csv(csv_file)

        print(f"Original data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())

        # Convert Excel dates to proper datetime
        print("\nConverting Excel serial dates to datetime...")
        df['Date'] = df['Date'].apply(convert_excel_date)

        # Save the corrected CSV
        corrected_file = 'data/qqq_corrected.csv'
        df.to_csv(corrected_file, index=False)
        print(f"Saved corrected dates to {corrected_file}")

        # Now use StockIndicators
        print("\n" + "=" * 70)
        print("Calculating Technical Indicators")
        print("=" * 70)

        stock = StockIndicators(corrected_file)

        print(f"\nTotal records: {len(stock.data)}")
        print(f"Date range: {stock.data['Date'].min()} to {stock.data['Date'].max()}")

        # Calculate all indicators
        print("\nCalculating indicators...")
        rsi = stock.calculate_rsi(period=14)
        print("✓ RSI (14-period)")

        mfi = stock.calculate_mfi(period=14)
        print("✓ MFI (14-period)")

        macd_data = stock.calculate_macd()
        print("✓ MACD (12, 26, 9)")

        ma_data = stock.calculate_ma(periods=[20, 50, 200])
        print("✓ Moving Averages (20, 50, 200)")

        # Display latest values
        print("\n" + "=" * 70)
        print("Latest QQQ Indicator Values")
        print("=" * 70)

        latest = stock.data.iloc[-1]
        print(f"\nDate: {latest['Date']}")
        print(f"Close Price: ${latest['Close']:.2f}")
        print(f"Volume: {latest['Volume']:,.0f}")

        print("\n--- Momentum Indicators ---")
        if not pd.isna(latest['RSI_14']):
            print(f"RSI (14): {latest['RSI_14']:.2f}")
            if latest['RSI_14'] > 70:
                print("  → Status: OVERBOUGHT")
            elif latest['RSI_14'] < 30:
                print("  → Status: OVERSOLD")
            else:
                print("  → Status: NEUTRAL")

        if not pd.isna(latest['MFI_14']):
            print(f"MFI (14): {latest['MFI_14']:.2f}")
            if latest['MFI_14'] > 80:
                print("  → Status: OVERBOUGHT")
            elif latest['MFI_14'] < 20:
                print("  → Status: OVERSOLD")
            else:
                print("  → Status: NEUTRAL")

        print("\n--- Trend Indicators ---")
        if not pd.isna(latest['MACD']):
            print(f"MACD Line: {latest['MACD']:.4f}")
            print(f"Signal Line: {latest['MACD_Signal']:.4f}")
            print(f"Histogram: {latest['MACD_Histogram']:.4f}")

            if latest['MACD_Histogram'] > 0:
                print("  → Momentum: BULLISH (histogram positive)")
            else:
                print("  → Momentum: BEARISH (histogram negative)")

        print("\n--- Moving Averages ---")
        if not pd.isna(latest['MA20']):
            print(f"MA20:  ${latest['MA20']:.2f}")
        if not pd.isna(latest['MA50']):
            print(f"MA50:  ${latest['MA50']:.2f}")
        if not pd.isna(latest['MA200']):
            print(f"MA200: ${latest['MA200']:.2f}")

        # Trend analysis based on MA
        if not pd.isna(latest['MA50']) and not pd.isna(latest['MA200']):
            print("\n--- Trend Analysis ---")
            close_price = latest['Close']

            if close_price > latest['MA200']:
                print(f"Price is ABOVE MA200 → Long-term UPTREND")
            else:
                print(f"Price is BELOW MA200 → Long-term DOWNTREND")

            if latest['MA50'] > latest['MA200']:
                print(f"MA50 > MA200 → Bullish alignment")
            else:
                print(f"MA50 < MA200 → Bearish alignment")

        # Display last 10 rows with indicators
        print("\n" + "=" * 70)
        print("Last 10 Days of QQQ Data with Indicators")
        print("=" * 70)

        display_cols = ['Date', 'Close', 'RSI_14', 'MFI_14', 'MACD', 'MA20', 'MA50', 'MA200']
        print(stock.data[display_cols].tail(10).to_string(index=False))

        # Export results
        output_file = 'data/qqq_with_indicators.csv'
        stock.export_to_csv(output_file)
        print(f"\n✓ Full data with indicators exported to {output_file}")

        # Summary
        print("\n" + "=" * 70)
        print("Summary Statistics")
        print("=" * 70)

        print(f"\nRSI Range: {rsi.min():.2f} - {rsi.max():.2f}")
        print(f"MFI Range: {mfi.min():.2f} - {mfi.max():.2f}")
        print(f"Close Price Range: ${stock.data['Close'].min():.2f} - ${stock.data['Close'].max():.2f}")

        # Count signals
        if not stock.data['RSI_14'].isna().all():
            oversold_days = (stock.data['RSI_14'] < 30).sum()
            overbought_days = (stock.data['RSI_14'] > 70).sum()
            print(f"\nRSI Signals:")
            print(f"  Oversold days (RSI < 30): {oversold_days}")
            print(f"  Overbought days (RSI > 70): {overbought_days}")

        print("\n" + "=" * 70)
        print("Test completed successfully!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure {csv_file} exists in the data folder")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
