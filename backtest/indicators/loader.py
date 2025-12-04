"""
CSV Data Loader Module.

This module provides functions to load and parse OHLCV data from CSV files.
All functions are pure and stateless for easy testing.
"""

from pathlib import Path

import pandas as pd

from backtest.indicators.exceptions import EmptyFileError
from backtest.indicators.exceptions import FileNotFoundError as CustomFileNotFoundError
from backtest.indicators.exceptions import LoaderError


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file and return a raw DataFrame.

    This function handles file existence checks, basic parsing,
    and initial data loading without validation.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Raw DataFrame with parsed data.

    Raises:
        FileNotFoundError: If the file does not exist.
        EmptyFileError: If the file is empty.
        LoaderError: If the file cannot be parsed.
    """
    path = Path(file_path)

    # Check file existence
    if not path.exists():
        raise CustomFileNotFoundError(file_path)

    # Check file is not empty
    if path.stat().st_size == 0:
        raise EmptyFileError(file_path)

    try:
        # Read CSV with minimal processing
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise EmptyFileError(file_path)
    except Exception as e:
        raise LoaderError(f"Failed to parse CSV: {e}")

    # Check for empty dataframe (headers only)
    if df.empty:
        raise EmptyFileError(file_path)

    return df


def parse_dates(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """
    Parse the date column to datetime format.

    Args:
        df: Input DataFrame.
        date_column: Name of the date column.

    Returns:
        DataFrame with parsed date column.

    Raises:
        LoaderError: If date parsing fails.
    """
    if date_column not in df.columns:
        # Will be caught by validation later
        return df

    df = df.copy()

    try:
        df[date_column] = pd.to_datetime(df[date_column])
    except Exception as e:
        raise LoaderError(f"Failed to parse dates in column '{date_column}': {e}")

    return df


def sort_by_date(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """
    Sort DataFrame by date in ascending order.

    Args:
        df: Input DataFrame.
        date_column: Name of the date column.

    Returns:
        Sorted DataFrame with reset index.
    """
    if date_column not in df.columns:
        return df

    df = df.sort_values(date_column, ascending=True).reset_index(drop=True)
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from column names.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with cleaned column names.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def load_and_prepare(file_path: str) -> pd.DataFrame:
    """
    Load CSV and prepare data for validation and indicator calculation.

    This is the main entry point for loading data. It combines:
    - File loading
    - Column name cleaning
    - Date parsing
    - Chronological sorting

    Args:
        file_path: Path to the CSV file.

    Returns:
        Prepared DataFrame ready for validation.

    Raises:
        FileNotFoundError: If the file does not exist.
        EmptyFileError: If the file is empty.
        LoaderError: If the file cannot be parsed.
    """
    df = load_csv(file_path)
    df = clean_column_names(df)
    df = parse_dates(df)
    df = sort_by_date(df)
    return df
