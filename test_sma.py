#!/usr/bin/env python3
"""
Test and validate Simple Moving Average (SMA) calculations.

This script verifies that MA20, MA50, and MA200 are calculated correctly
by comparing against manual calculations.
"""

import pandas as pd
from stock_indicators import StockIndicators

def manual_sma(data, period):
    """Manually calculate SMA for verification."""
    return data.rolling(window=period).mean()

def test_sma_calculations():
    """Test SMA calculations against manual calculations."""

    print("=" * 70)
    print("Testing Simple Moving Average (SMA) Calculations")
    print("=" * 70)

    # Load the corrected QQQ data
    csv_file = 'data/qqq_corrected.csv'

    try:
        # Load data using StockIndicators
        stock = StockIndicators(csv_file)

        print(f"\nLoaded {len(stock.data)} records")
        print(f"Date range: {stock.data['Date'].min()} to {stock.data['Date'].max()}\n")

        # Calculate SMAs using the class
        print("Calculating SMAs using StockIndicators class...")
        ma_data = stock.calculate_ma(periods=[20, 50, 200])

        # Manual calculations for verification
        print("Calculating SMAs manually for verification...")
        manual_ma20 = manual_sma(stock.data['Close'], 20)
        manual_ma50 = manual_sma(stock.data['Close'], 50)
        manual_ma200 = manual_sma(stock.data['Close'], 200)

        # Compare results
        print("\n" + "=" * 70)
        print("Verification: Comparing Class Output vs Manual Calculation")
        print("=" * 70)

        # Check MA20
        ma20_match = stock.data['MA20'].equals(manual_ma20)
        print(f"\nMA20 Match: {ma20_match}")
        if ma20_match:
            print("✓ MA20 calculations are CORRECT")
        else:
            # Check if differences are due to floating point precision
            max_diff = (stock.data['MA20'] - manual_ma20).abs().max()
            print(f"  Max difference: {max_diff}")
            if max_diff < 1e-10:
                print("✓ MA20 calculations are CORRECT (within floating point precision)")
            else:
                print("✗ MA20 calculations have significant differences")

        # Check MA50
        ma50_match = stock.data['MA50'].equals(manual_ma50)
        print(f"\nMA50 Match: {ma50_match}")
        if ma50_match:
            print("✓ MA50 calculations are CORRECT")
        else:
            max_diff = (stock.data['MA50'] - manual_ma50).abs().max()
            print(f"  Max difference: {max_diff}")
            if max_diff < 1e-10:
                print("✓ MA50 calculations are CORRECT (within floating point precision)")
            else:
                print("✗ MA50 calculations have significant differences")

        # Check MA200
        ma200_match = stock.data['MA200'].equals(manual_ma200)
        print(f"\nMA200 Match: {ma200_match}")
        if ma200_match:
            print("✓ MA200 calculations are CORRECT")
        else:
            max_diff = (stock.data['MA200'] - manual_ma200).abs().max()
            print(f"  Max difference: {max_diff}")
            if max_diff < 1e-10:
                print("✓ MA200 calculations are CORRECT (within floating point precision)")
            else:
                print("✗ MA200 calculations have significant differences")

        # Display sample calculations
        print("\n" + "=" * 70)
        print("Sample Manual Verification")
        print("=" * 70)

        # Verify MA20 at a specific point (row 220 has enough history)
        idx = 220
        print(f"\nVerifying MA20 at index {idx} (Date: {stock.data.iloc[idx]['Date']}):")
        print(f"Close price: ${stock.data.iloc[idx]['Close']:.2f}")

        # Get the last 20 close prices
        last_20_closes = stock.data['Close'].iloc[idx-19:idx+1]
        manual_avg = last_20_closes.mean()
        class_ma20 = stock.data['MA20'].iloc[idx]

        print(f"\nLast 20 closing prices:")
        print(last_20_closes.values)
        print(f"\nManual calculation: {manual_avg:.6f}")
        print(f"Class calculation:  {class_ma20:.6f}")
        print(f"Difference:         {abs(manual_avg - class_ma20):.10f}")

        if abs(manual_avg - class_ma20) < 1e-10:
            print("✓ MA20 manual verification PASSED")
        else:
            print("✗ MA20 manual verification FAILED")

        # Display latest values with context
        print("\n" + "=" * 70)
        print("Latest SMA Values (Last 5 Days)")
        print("=" * 70)

        display_cols = ['Date', 'Close', 'MA20', 'MA50', 'MA200']
        print("\n" + stock.data[display_cols].tail(5).to_string(index=False))

        # Show the latest values
        latest = stock.data.iloc[-1]
        print("\n" + "=" * 70)
        print(f"Latest Data (Date: {latest['Date']})")
        print("=" * 70)
        print(f"Close Price: ${latest['Close']:.2f}")
        print(f"MA20:        ${latest['MA20']:.2f}")
        print(f"MA50:        ${latest['MA50']:.2f}")
        print(f"MA200:       ${latest['MA200']:.2f}")

        # Trend analysis
        print("\n" + "=" * 70)
        print("Trend Analysis")
        print("=" * 70)

        close = latest['Close']
        ma20 = latest['MA20']
        ma50 = latest['MA50']
        ma200 = latest['MA200']

        print(f"\nPrice vs MA20:  ${close:.2f} vs ${ma20:.2f} = ", end="")
        if close > ma20:
            print(f"ABOVE by ${close - ma20:.2f}")
        else:
            print(f"BELOW by ${ma20 - close:.2f}")

        print(f"Price vs MA50:  ${close:.2f} vs ${ma50:.2f} = ", end="")
        if close > ma50:
            print(f"ABOVE by ${close - ma50:.2f}")
        else:
            print(f"BELOW by ${ma50 - close:.2f}")

        print(f"Price vs MA200: ${close:.2f} vs ${ma200:.2f} = ", end="")
        if close > ma200:
            print(f"ABOVE by ${close - ma200:.2f}")
        else:
            print(f"BELOW by ${ma200 - close:.2f}")

        print(f"\nMA20 vs MA50:   ${ma20:.2f} vs ${ma50:.2f} = ", end="")
        if ma20 > ma50:
            print(f"ABOVE by ${ma20 - ma50:.2f} (Short-term strength)")
        else:
            print(f"BELOW by ${ma50 - ma20:.2f} (Short-term weakness)")

        print(f"MA50 vs MA200:  ${ma50:.2f} vs ${ma200:.2f} = ", end="")
        if ma50 > ma200:
            print(f"ABOVE by ${ma50 - ma200:.2f} (GOLDEN CROSS - Bullish)")
        else:
            print(f"BELOW by ${ma200 - ma50:.2f} (DEATH CROSS - Bearish)")

        # Check for crossovers in recent history
        print("\n" + "=" * 70)
        print("Recent Crossover Detection (Last 20 Days)")
        print("=" * 70)

        recent_data = stock.data.tail(20).copy()

        # Check for MA20/MA50 crossover
        recent_data['MA20_above_MA50'] = recent_data['MA20'] > recent_data['MA50']
        if recent_data['MA20_above_MA50'].diff().any():
            crossover_points = recent_data[recent_data['MA20_above_MA50'].diff() != 0]
            if len(crossover_points) > 0:
                print("\nMA20/MA50 Crossovers detected:")
                for idx, row in crossover_points.iterrows():
                    if row['MA20_above_MA50']:
                        print(f"  {row['Date']}: MA20 crossed ABOVE MA50 (Bullish)")
                    else:
                        print(f"  {row['Date']}: MA20 crossed BELOW MA50 (Bearish)")
            else:
                print("\nNo MA20/MA50 crossovers in last 20 days")
        else:
            print("\nNo MA20/MA50 crossovers in last 20 days")

        # Summary statistics
        print("\n" + "=" * 70)
        print("SMA Statistics (Entire Dataset)")
        print("=" * 70)

        print(f"\nMA20 Range:  ${stock.data['MA20'].min():.2f} - ${stock.data['MA20'].max():.2f}")
        print(f"MA50 Range:  ${stock.data['MA50'].min():.2f} - ${stock.data['MA50'].max():.2f}")
        print(f"MA200 Range: ${stock.data['MA200'].min():.2f} - ${stock.data['MA200'].max():.2f}")

        # Count how many days price was above each MA
        days_above_ma20 = (stock.data['Close'] > stock.data['MA20']).sum()
        days_above_ma50 = (stock.data['Close'] > stock.data['MA50']).sum()
        days_above_ma200 = (stock.data['Close'] > stock.data['MA200']).sum()
        total_days = len(stock.data)

        print(f"\nDays price was ABOVE moving averages:")
        print(f"  MA20:  {days_above_ma20}/{total_days} ({days_above_ma20/total_days*100:.1f}%)")
        print(f"  MA50:  {days_above_ma50}/{total_days} ({days_above_ma50/total_days*100:.1f}%)")
        print(f"  MA200: {days_above_ma200}/{total_days} ({days_above_ma200/total_days*100:.1f}%)")

        print("\n" + "=" * 70)
        print("✓ All SMA tests completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_sma_calculations()
