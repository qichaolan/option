"""
Data Validation Module.

This module provides pure functions to validate OHLCV data.
Each validation function checks a specific aspect of data quality.
"""

import pandas as pd

from backtest.indicators.exceptions import (
    DuplicateDateError,
    EmptyDataError,
    InvalidDataTypeError,
    MissingColumnError,
    NonMonotonicDateError,
)

# Required columns for OHLCV data
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def validate_not_empty(df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame is not empty.

    Args:
        df: Input DataFrame.

    Raises:
        EmptyDataError: If the DataFrame has no rows.
    """
    if df.empty or len(df) == 0:
        raise EmptyDataError()


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Validate that all required columns are present.

    Args:
        df: Input DataFrame.

    Raises:
        MissingColumnError: If any required columns are missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise MissingColumnError(missing)


def validate_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and convert numeric columns to proper types.

    This function attempts to convert OHLCV columns to numeric types.
    It returns a new DataFrame with converted columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with numeric columns properly typed.

    Raises:
        InvalidDataTypeError: If a column cannot be converted to numeric.
    """
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue

        # Try to convert to numeric
        original_values = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check if any values failed to convert (became NaN but weren't originally)
        original_nulls = original_values.isna()
        new_nulls = df[col].isna()
        conversion_failures = new_nulls & ~original_nulls

        if conversion_failures.any():
            # Find first failing value for error message
            first_fail_idx = conversion_failures.idxmax()
            first_fail_value = original_values.iloc[first_fail_idx]
            raise InvalidDataTypeError(
                col, f"Cannot convert '{first_fail_value}' to numeric"
            )

    return df


def validate_no_missing_ohlcv(df: pd.DataFrame) -> None:
    """
    Validate that there are no missing values in OHLCV columns.

    Args:
        df: Input DataFrame.

    Raises:
        InvalidDataTypeError: If any OHLCV column has missing values.
    """
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue

        if df[col].isna().any():
            null_count = df[col].isna().sum()
            raise InvalidDataTypeError(col, f"Contains {null_count} missing value(s)")


def validate_no_duplicate_dates(df: pd.DataFrame) -> None:
    """
    Validate that there are no duplicate dates.

    Args:
        df: Input DataFrame.

    Raises:
        DuplicateDateError: If duplicate dates are found.
    """
    if "Date" not in df.columns:
        return

    duplicates = df[df["Date"].duplicated(keep=False)]["Date"].unique()
    if len(duplicates) > 0:
        raise DuplicateDateError(list(duplicates))


def validate_chronological_order(df: pd.DataFrame) -> None:
    """
    Validate that dates are in strictly increasing chronological order.

    This check is performed after sorting, so it verifies that
    there are no date inversions in the original data that would
    indicate data quality issues.

    Args:
        df: Input DataFrame (should already be sorted).

    Raises:
        NonMonotonicDateError: If dates are not monotonically increasing.
    """
    if "Date" not in df.columns or len(df) < 2:
        return

    # Check if dates are strictly increasing
    date_diff = df["Date"].diff().dropna()

    # All differences should be positive (strictly increasing)
    if (date_diff <= pd.Timedelta(0)).any():
        # Find first non-increasing pair
        bad_idx = (date_diff <= pd.Timedelta(0)).idxmax()
        prev_date = df.loc[bad_idx - 1, "Date"]
        curr_date = df.loc[bad_idx, "Date"]
        raise NonMonotonicDateError(f"Date {curr_date} <= previous date {prev_date}")


def validate_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all validations on the DataFrame.

    This function runs all validation checks in the proper order
    and returns a validated DataFrame with proper types.

    Args:
        df: Input DataFrame.

    Returns:
        Validated DataFrame with proper types.

    Raises:
        EmptyDataError: If the DataFrame is empty.
        MissingColumnError: If required columns are missing.
        InvalidDataTypeError: If columns have invalid types.
        DuplicateDateError: If duplicate dates exist.
        NonMonotonicDateError: If dates are not chronological.
    """
    # Check emptiness first
    validate_not_empty(df)

    # Check required columns
    validate_required_columns(df)

    # Convert and validate numeric columns
    df = validate_numeric_columns(df)

    # Check for missing OHLCV values
    validate_no_missing_ohlcv(df)

    # Check for duplicate dates
    validate_no_duplicate_dates(df)

    # Check chronological order (after sorting)
    validate_chronological_order(df)

    return df
