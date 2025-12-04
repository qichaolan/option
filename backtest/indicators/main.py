#!/usr/bin/env python3
"""
Stock Indicator Generation Module.

A production-quality module for calculating technical indicators from OHLCV data.

Usage (Python API):
    from backtest import build_indicators
    df = build_indicators("SPY.csv")

    # With train/test split
    train_df, test_df = build_indicators("SPY.csv", test_days=60)

Usage (CLI):
    python -m backtest.indicators --input_file SPY.csv --output_file SPY_ind.csv
    python -m backtest.indicators -i SPY.csv -o SPY --test 60  # Creates SPY_train.csv and SPY_test.csv

Indicators calculated:
    Trend/Momentum: SMA (5,9,20,50,200), EMA (9,21,50), MACD, RSI, MFI
    Volatility: ATR, Historical Volatility
    Volume: OBV, Volume SMA
    Market Structure: Pivot Points, Bollinger Bands
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from backtest.indicators.calculations import add_all_indicators
from backtest.indicators.exceptions import IndicatorError
from backtest.indicators.loader import load_and_prepare
from backtest.indicators.validators import validate_all


def split_train_test(
    df: pd.DataFrame,
    test_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and test sets using walk-forward style.

    The latest N days go into the test set, the rest into the train set.
    Indicators are calculated on the FULL dataset first, then split,
    ensuring test set has valid indicator values.

    Args:
        df: DataFrame with indicators already calculated.
        test_days: Number of days for test set.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If test_days >= total rows.
    """
    total_rows = len(df)

    if test_days >= total_rows:
        raise ValueError(
            f"test_days ({test_days}) must be less than total rows ({total_rows})"
        )

    if test_days <= 0:
        raise ValueError(f"test_days must be positive, got {test_days}")

    split_idx = total_rows - test_days
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def build_indicators(
    input_file: str,
    output_file: Optional[str] = None,
    test_days: Optional[int] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load OHLCV data from CSV and calculate all technical indicators.

    This is the main API function for programmatic usage.

    Args:
        input_file: Path to input CSV file with OHLCV data.
        output_file: Optional path to save output CSV. If None, only returns DataFrame.
            When test_days is provided, this is used as the base name for train/test files.
        test_days: Optional number of days for test set. If provided, splits data
            into train and test sets (last N days = test, rest = train).

    Returns:
        If test_days is None: DataFrame with original data plus all calculated indicators.
        If test_days is provided: Tuple of (train_df, test_df).

    Raises:
        FileNotFoundError: If input file does not exist.
        EmptyFileError: If input file is empty.
        LoaderError: If CSV parsing fails.
        ValidationError: If data validation fails.
        ValueError: If test_days is invalid.

    Example:
        >>> df = build_indicators("SPY.csv")
        >>> print(df.columns.tolist())
        ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'sma_5', ...]

        >>> train_df, test_df = build_indicators("SPY.csv", test_days=60)
    """
    # Load and prepare data
    df = load_and_prepare(input_file)

    # Validate data
    df = validate_all(df)

    # Calculate indicators on FULL dataset first
    df = add_all_indicators(df)

    # Handle train/test split if requested
    if test_days is not None:
        train_df, test_df = split_train_test(df, test_days)

        # Optionally save to files
        if output_file:
            # Generate train/test file names from base output_file
            output_path = Path(output_file)
            stem = output_path.stem
            suffix = output_path.suffix or ".csv"
            parent = output_path.parent

            train_path = parent / f"{stem}_train{suffix}"
            test_path = parent / f"{stem}_test{suffix}"

            save_to_csv(train_df, str(train_path))
            save_to_csv(test_df, str(test_path))

        return train_df, test_df

    # No split - return single DataFrame
    if output_file:
        save_to_csv(df, output_file)

    return df


def save_to_csv(df: pd.DataFrame, output_file: str) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        df: DataFrame to save.
        output_file: Output file path.

    Raises:
        IOError: If file cannot be written.
    """
    output_path = Path(output_file)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="indicators",
        description="Calculate technical indicators from OHLCV CSV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backtest.indicators --input_file SPY.csv
  python -m backtest.indicators --input_file SPY.csv --output_file SPY_indicators.csv
  python -m backtest.indicators -i data.csv -o output.csv

Indicators calculated:
  Trend/Momentum: sma_5, sma_9, sma_20, sma_50, sma_200, ema_9, ema_21, ema_50,
                  macd_12_26_9, macd_signal_12_26_9, macd_hist_12_26_9, rsi_14, mfi_14
  Volatility:     atr_14, hv_20
  Volume:         obv, vol_sma_20
  Structure:      pivot_high_3, pivot_low_3, bb_mid_20_2, bb_upper_20_2, bb_lower_20_2
""",
    )

    parser.add_argument(
        "--input_file",
        "-i",
        required=True,
        type=str,
        help="Path to input CSV file with OHLCV data",
    )

    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default=None,
        help="Path to output CSV file (optional, prints summary if not provided)",
    )

    parser.add_argument(
        "--test",
        "-t",
        type=int,
        default=None,
        metavar="N",
        help="Split last N days into test set (creates _train and _test files)",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for CLI.

    Args:
        args: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    try:
        result = build_indicators(
            input_file=parsed_args.input_file,
            output_file=parsed_args.output_file,
            test_days=parsed_args.test,
        )

        # Handle train/test split output
        if parsed_args.test is not None:
            train_df, test_df = result
            total_rows = len(train_df) + len(test_df)

            print(f"Processed {total_rows} rows from {parsed_args.input_file}")
            print(f"Train set: {len(train_df)} rows")
            print(f"  Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
            print(f"Test set: {len(test_df)} rows")
            print(f"  Date range: {test_df['Date'].min()} to {test_df['Date'].max()}")
            print(f"Indicators added: {len(train_df.columns) - 6}")

            if parsed_args.output_file:
                output_path = Path(parsed_args.output_file)
                stem = output_path.stem
                suffix = output_path.suffix or ".csv"
                parent = output_path.parent
                print(f"Output saved to: {parent / f'{stem}_train{suffix}'}")
                print(f"Output saved to: {parent / f'{stem}_test{suffix}'}")
        else:
            df = result

            # Print summary
            print(f"Processed {len(df)} rows from {parsed_args.input_file}")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"Indicators added: {len(df.columns) - 6}")

            if parsed_args.output_file:
                print(f"Output saved to: {parsed_args.output_file}")
            else:
                print("\nLatest indicator values:")
                latest = df.iloc[-1]
                indicator_cols = [
                    col
                    for col in df.columns
                    if col not in ["Date", "Open", "High", "Low", "Close", "Volume"]
                ]
                for col in indicator_cols[:10]:
                    val = latest[col]
                    if pd.notna(val):
                        print(f"  {col}: {val:.4f}")
                if len(indicator_cols) > 10:
                    print(f"  ... and {len(indicator_cols) - 10} more")

        return 0

    except IndicatorError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
