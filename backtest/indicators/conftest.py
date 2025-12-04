"""
Pytest configuration and fixtures for backtest tests.

This module provides shared fixtures and test data generators
for all test modules.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    np.random.seed(42)

    # Generate realistic price data
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.abs(np.random.randn(100) * 0.5)
    low = close - np.abs(np.random.randn(100) * 0.5)
    open_price = low + (high - low) * np.random.rand(100)
    volume = np.random.randint(1000000, 5000000, 100)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


@pytest.fixture
def minimal_ohlcv_data() -> pd.DataFrame:
    """Create minimal OHLCV data (10 rows) for edge case testing."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="B")

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5],
            "High": [101.0, 102.0, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5, 106.0, 105.5],
            "Low": [99.0, 100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5],
            "Close": [100.5, 101.5, 102.5, 101.0, 103.5, 102.0, 104.5, 103.0, 105.5, 104.0],
            "Volume": [1000000, 1100000, 1200000, 900000, 1300000, 1000000, 1400000, 1100000, 1500000, 1200000],
        }
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_csv_file(temp_dir, sample_ohlcv_data) -> Path:
    """Create a valid CSV file for testing."""
    file_path = temp_dir / "valid.csv"
    sample_ohlcv_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def minimal_csv_file(temp_dir, minimal_ohlcv_data) -> Path:
    """Create a minimal CSV file for testing."""
    file_path = temp_dir / "minimal.csv"
    minimal_ohlcv_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def empty_file(temp_dir) -> Path:
    """Create an empty file."""
    file_path = temp_dir / "empty.csv"
    file_path.touch()
    return file_path


@pytest.fixture
def headers_only_file(temp_dir) -> Path:
    """Create a CSV file with headers only."""
    file_path = temp_dir / "headers_only.csv"
    file_path.write_text("Date,Open,High,Low,Close,Volume\n")
    return file_path


@pytest.fixture
def missing_column_file(temp_dir) -> Path:
    """Create a CSV file missing required columns."""
    file_path = temp_dir / "missing_column.csv"
    file_path.write_text(
        "Date,Open,High,Low,Close\n"
        "2023-01-01,100,101,99,100.5\n"
    )
    return file_path


@pytest.fixture
def invalid_dtype_file(temp_dir) -> Path:
    """Create a CSV file with invalid data types."""
    file_path = temp_dir / "invalid_dtype.csv"
    file_path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2023-01-01,100,101,99,100.5,1000000\n"
        "2023-01-02,abc,102,100,101.5,1100000\n"
    )
    return file_path


@pytest.fixture
def duplicate_dates_file(temp_dir) -> Path:
    """Create a CSV file with duplicate dates."""
    file_path = temp_dir / "duplicate_dates.csv"
    file_path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2023-01-01,100,101,99,100.5,1000000\n"
        "2023-01-01,101,102,100,101.5,1100000\n"
        "2023-01-03,102,103,101,102.5,1200000\n"
    )
    return file_path


@pytest.fixture
def non_monotonic_dates_file(temp_dir) -> Path:
    """Create a CSV file with non-monotonic dates."""
    file_path = temp_dir / "non_monotonic.csv"
    file_path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2023-01-03,102,103,101,102.5,1200000\n"
        "2023-01-01,100,101,99,100.5,1000000\n"
        "2023-01-02,101,102,100,101.5,1100000\n"
    )
    return file_path


@pytest.fixture
def missing_values_file(temp_dir) -> Path:
    """Create a CSV file with missing OHLCV values."""
    file_path = temp_dir / "missing_values.csv"
    file_path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2023-01-01,100,101,99,100.5,1000000\n"
        "2023-01-02,,102,100,101.5,1100000\n"
        "2023-01-03,102,103,101,102.5,1200000\n"
    )
    return file_path


@pytest.fixture
def whitespace_columns_file(temp_dir) -> Path:
    """Create a CSV file with whitespace in column names."""
    file_path = temp_dir / "whitespace_columns.csv"
    file_path.write_text(
        " Date , Open , High , Low , Close , Volume \n"
        "2023-01-01,100,101,99,100.5,1000000\n"
        "2023-01-02,101,102,100,101.5,1100000\n"
    )
    return file_path


@pytest.fixture
def unsorted_dates_file(temp_dir) -> Path:
    """Create a CSV file with unsorted dates (but valid data)."""
    file_path = temp_dir / "unsorted_dates.csv"
    file_path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2023-01-03,102,103,101,102.5,1200000\n"
        "2023-01-01,100,101,99,100.5,1000000\n"
        "2023-01-02,101,102,100,101.5,1100000\n"
    )
    return file_path


@pytest.fixture
def large_dataset(temp_dir) -> Path:
    """Create a large dataset for performance testing."""
    dates = pd.date_range(start="2010-01-01", periods=5000, freq="B")
    np.random.seed(42)

    close = 100 + np.cumsum(np.random.randn(5000) * 0.5)
    high = close + np.abs(np.random.randn(5000) * 0.5)
    low = close - np.abs(np.random.randn(5000) * 0.5)
    open_price = low + (high - low) * np.random.rand(5000)
    volume = np.random.randint(1000000, 5000000, 5000)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )

    file_path = temp_dir / "large_dataset.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def adj_close_file(temp_dir) -> Path:
    """Create a CSV file with Adj Close column (like Yahoo Finance)."""
    file_path = temp_dir / "with_adj_close.csv"
    file_path.write_text(
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2023-01-01,100,101,99,100.5,100.5,1000000\n"
        "2023-01-02,101,102,100,101.5,101.5,1100000\n"
        "2023-01-03,102,103,101,102.5,102.5,1200000\n"
    )
    return file_path
