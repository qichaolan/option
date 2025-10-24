#!/usr/bin/env python3
"""
Test StockIndicators with DataFrame input (from memory, not CSV).

This test verifies that StockIndicators can accept data directly
from memory as a pandas DataFrame, without requiring a CSV file.
"""

import pandas as pd
from stock_indicators import StockIndicators
from datetime import datetime, timedelta


def test_dataframe_input():
    """Test StockIndicators with DataFrame input."""

    print("=" * 70)
    print("Testing StockIndicators with DataFrame Input")
    print("=" * 70)

    # Create sample data in memory
    print("\n1. Creating sample stock data in memory...")

    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    n = len(dates)

    # Simulate stock price data
    base_price = 100.0
    prices = []
    for i in range(n):
        # Simple random walk simulation
        change = (i % 10 - 5) * 0.5
        price = base_price + change + (i * 0.1)
        prices.append(price)

    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': [p - (p * 0.01) for p in prices],
        'High': [p + (p * 0.02) for p in prices],
        'Low': [p - (p * 0.015) for p in prices],
        'Close': prices,
        'Volume': [1000000 + (i * 10000) for i in range(n)]
    })

    print(f"✓ Created {len(df)} records in memory")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

    # Test 1: Initialize StockIndicators with DataFrame
    print("\n2. Initializing StockIndicators with DataFrame...")

    try:
        stock = StockIndicators(df)
        print("✓ StockIndicators initialized successfully")
        print(f"  Source: {stock.source}")
        print(f"  Records: {len(stock.data)}")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False

    # Test 2: Calculate indicators
    print("\n3. Calculating technical indicators...")

    try:
        rsi = stock.calculate_rsi(period=14)
        print("✓ RSI calculated")

        mfi = stock.calculate_mfi(period=14)
        print("✓ MFI calculated")

        macd = stock.calculate_macd()
        print("✓ MACD calculated")

        ma = stock.calculate_ma(periods=[20, 50, 200])
        print("✓ Moving Averages calculated")
    except Exception as e:
        print(f"✗ Failed to calculate indicators: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Verify results
    print("\n4. Verifying results...")

    latest = stock.data.iloc[-1]

    print(f"\n  Latest Date: {latest['Date']}")
    print(f"  Close Price: ${latest['Close']:.2f}")

    # Check that indicators were calculated
    indicators_ok = True

    if pd.isna(latest['RSI_14']):
        print("  ✗ RSI not calculated")
        indicators_ok = False
    else:
        print(f"  ✓ RSI: {latest['RSI_14']:.2f}")

    if pd.isna(latest['MFI_14']):
        print("  ✗ MFI not calculated")
        indicators_ok = False
    else:
        print(f"  ✓ MFI: {latest['MFI_14']:.2f}")

    if pd.isna(latest['MACD']):
        print("  ✗ MACD not calculated")
        indicators_ok = False
    else:
        print(f"  ✓ MACD: {latest['MACD']:.4f}")

    if pd.isna(latest['MA20']):
        print("  ✗ MA20 not calculated")
        indicators_ok = False
    else:
        print(f"  ✓ MA20: ${latest['MA20']:.2f}")

    # Test 4: Compare DataFrame input vs CSV input
    print("\n5. Comparing DataFrame vs CSV input...")

    # Save to CSV
    csv_file = 'test_temp.csv'
    df.to_csv(csv_file, index=False)

    # Load from CSV
    stock_csv = StockIndicators(csv_file)
    stock_csv.calculate_rsi(period=14)
    stock_csv.calculate_mfi(period=14)
    stock_csv.calculate_macd()
    stock_csv.calculate_ma(periods=[20, 50, 200])

    # Compare results
    latest_csv = stock_csv.data.iloc[-1]

    print(f"\n  DataFrame RSI: {latest['RSI_14']:.6f}")
    print(f"  CSV RSI:       {latest_csv['RSI_14']:.6f}")

    rsi_match = abs(latest['RSI_14'] - latest_csv['RSI_14']) < 1e-10
    mfi_match = abs(latest['MFI_14'] - latest_csv['MFI_14']) < 1e-10
    macd_match = abs(latest['MACD'] - latest_csv['MACD']) < 1e-10

    if rsi_match and mfi_match and macd_match:
        print("\n  ✓ DataFrame and CSV methods produce identical results")
    else:
        print("\n  ✗ DataFrame and CSV methods produce different results")
        indicators_ok = False

    # Clean up temp file
    import os
    os.remove(csv_file)

    # Test 5: Test with real QQQ data
    print("\n6. Testing with real QQQ data from CSV...")

    # Load QQQ data as DataFrame
    qqq_df = pd.read_csv('data/qqq_corrected.csv', parse_dates=['Date'])

    # Use only last year of data
    one_year_ago = qqq_df['Date'].max() - timedelta(days=365)
    qqq_df_1yr = qqq_df[qqq_df['Date'] >= one_year_ago].copy()

    print(f"  Original QQQ records: {len(qqq_df)}")
    print(f"  Last year records: {len(qqq_df_1yr)}")

    # Calculate indicators on 1-year subset
    stock_qqq = StockIndicators(qqq_df_1yr)
    stock_qqq.calculate_rsi(period=14)
    stock_qqq.calculate_mfi(period=14)
    stock_qqq.calculate_macd()
    stock_qqq.calculate_ma(periods=[20, 50, 200])

    latest_qqq = stock_qqq.data.iloc[-1]

    print(f"\n  QQQ Latest (1-year subset):")
    print(f"    Date: {latest_qqq['Date']}")
    print(f"    Close: ${latest_qqq['Close']:.2f}")
    print(f"    RSI: {latest_qqq['RSI_14']:.2f}")
    print(f"    MFI: {latest_qqq['MFI_14']:.2f}")

    # Summary
    print("\n" + "=" * 70)
    if indicators_ok:
        print("✓ All DataFrame input tests PASSED")
    else:
        print("✗ Some DataFrame input tests FAILED")
    print("=" * 70)

    return indicators_ok


def test_error_handling():
    """Test error handling for invalid inputs."""

    print("\n" + "=" * 70)
    print("Testing Error Handling")
    print("=" * 70)

    # Test 1: Invalid data type
    print("\n1. Testing invalid data type...")
    try:
        stock = StockIndicators(12345)  # Invalid type
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test 2: DataFrame with missing columns
    print("\n2. Testing DataFrame with missing columns...")
    df_bad = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=10),
        'Close': [100 + i for i in range(10)]
        # Missing: Open, High, Low, Volume
    })

    try:
        stock = StockIndicators(df_bad)
        print("✗ Should have raised ValueError for missing columns")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    print("\n" + "=" * 70)
    print("✓ All error handling tests PASSED")
    print("=" * 70)

    return True


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("StockIndicators DataFrame Input Test Suite")
    print("=" * 70)

    # Run tests
    test1_passed = test_dataframe_input()
    test2_passed = test_error_handling()

    # Final summary
    print("\n\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    if test1_passed and test2_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")

    print("=" * 70)
