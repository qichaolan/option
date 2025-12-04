"""
Tests for the loader module.

This module tests all CSV loading and parsing functionality including:
- Valid CSV loading
- Missing file handling
- Empty file handling
- Date parsing
- Column name cleaning
- Sorting
"""

import pandas as pd
import pytest

from backtest.indicators.exceptions import EmptyFileError
from backtest.indicators.exceptions import FileNotFoundError as CustomFileNotFoundError
from backtest.indicators.exceptions import LoaderError
from backtest.indicators.loader import (
    clean_column_names,
    load_and_prepare,
    load_csv,
    parse_dates,
    sort_by_date,
)


class TestLoadCsv:
    """Tests for load_csv function."""

    def test_load_valid_csv(self, valid_csv_file):
        """Test loading a valid CSV file."""
        df = load_csv(str(valid_csv_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "Date" in df.columns
        assert "Close" in df.columns

    def test_load_missing_file(self, temp_dir):
        """Test loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(CustomFileNotFoundError) as exc_info:
            load_csv(str(temp_dir / "nonexistent.csv"))
        assert "nonexistent.csv" in str(exc_info.value)

    def test_load_empty_file(self, empty_file):
        """Test loading an empty file raises EmptyFileError."""
        with pytest.raises(EmptyFileError) as exc_info:
            load_csv(str(empty_file))
        assert "empty.csv" in str(exc_info.value)

    def test_load_headers_only_file(self, headers_only_file):
        """Test loading a file with headers only raises EmptyFileError."""
        with pytest.raises(EmptyFileError):
            load_csv(str(headers_only_file))

    def test_load_with_adj_close(self, adj_close_file):
        """Test loading a file with Adj Close column works."""
        df = load_csv(str(adj_close_file))
        assert len(df) == 3
        assert "Adj Close" in df.columns

    def test_load_whitespace_only_file(self, temp_dir):
        """Test loading a file with only whitespace raises EmptyFileError."""
        file_path = temp_dir / "whitespace.csv"
        file_path.write_text("   \n   \n")
        with pytest.raises(EmptyFileError):
            load_csv(str(file_path))


class TestParseDates:
    """Tests for parse_dates function."""

    def test_parse_valid_dates(self, sample_ohlcv_data):
        """Test parsing dates from a DataFrame."""
        # Create a copy with string dates
        df = sample_ohlcv_data.copy()
        df["Date"] = df["Date"].astype(str)

        result = parse_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_parse_missing_date_column(self, sample_ohlcv_data):
        """Test that missing date column is handled gracefully."""
        df = sample_ohlcv_data.drop(columns=["Date"])
        result = parse_dates(df)
        assert "Date" not in result.columns

    def test_parse_invalid_dates(self, temp_dir):
        """Test parsing invalid dates raises LoaderError."""
        file_path = temp_dir / "invalid_dates.csv"
        file_path.write_text(
            "Date,Open,High,Low,Close,Volume\n"
            "not-a-date,100,101,99,100.5,1000000\n"
        )
        df = load_csv(str(file_path))
        with pytest.raises(LoaderError) as exc_info:
            parse_dates(df)
        assert "Failed to parse dates" in str(exc_info.value)

    def test_parse_various_date_formats(self, temp_dir):
        """Test parsing various date formats."""
        file_path = temp_dir / "various_dates.csv"
        file_path.write_text(
            "Date,Open,High,Low,Close,Volume\n"
            "2023-01-01,100,101,99,100.5,1000000\n"
            "2023-01-02,101,102,100,101.5,1100000\n"
        )
        df = load_csv(str(file_path))
        result = parse_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_parse_already_datetime(self, sample_ohlcv_data):
        """Test parsing when dates are already datetime."""
        result = parse_dates(sample_ohlcv_data)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])


class TestSortByDate:
    """Tests for sort_by_date function."""

    def test_sort_unsorted_data(self, temp_dir):
        """Test sorting unsorted data."""
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"]),
                "Close": [103, 101, 102],
            }
        )
        result = sort_by_date(df)
        assert result["Date"].iloc[0] == pd.Timestamp("2023-01-01")
        assert result["Date"].iloc[-1] == pd.Timestamp("2023-01-03")
        assert result["Close"].iloc[0] == 101

    def test_sort_already_sorted(self, sample_ohlcv_data):
        """Test sorting already sorted data."""
        result = sort_by_date(sample_ohlcv_data)
        assert len(result) == len(sample_ohlcv_data)
        assert result["Date"].is_monotonic_increasing

    def test_sort_missing_date_column(self):
        """Test sorting without date column returns unchanged DataFrame."""
        df = pd.DataFrame({"Close": [100, 101, 102]})
        result = sort_by_date(df)
        assert "Date" not in result.columns
        assert list(result["Close"]) == [100, 101, 102]

    def test_sort_resets_index(self, temp_dir):
        """Test that sorting resets the index."""
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"]),
                "Close": [103, 101, 102],
            }
        )
        result = sort_by_date(df)
        assert list(result.index) == [0, 1, 2]


class TestCleanColumnNames:
    """Tests for clean_column_names function."""

    def test_clean_whitespace(self, whitespace_columns_file):
        """Test cleaning whitespace from column names."""
        df = load_csv(str(whitespace_columns_file))
        result = clean_column_names(df)
        assert "Date" in result.columns
        assert "Open" in result.columns
        assert " Date " not in result.columns

    def test_clean_no_whitespace(self, sample_ohlcv_data):
        """Test cleaning when no whitespace present."""
        result = clean_column_names(sample_ohlcv_data)
        assert list(result.columns) == list(sample_ohlcv_data.columns)

    def test_clean_trailing_spaces(self, temp_dir):
        """Test cleaning trailing spaces from column names."""
        file_path = temp_dir / "trailing.csv"
        file_path.write_text(
            "Date  ,Open  ,High  ,Low  ,Close  ,Volume  \n"
            "2023-01-01,100,101,99,100.5,1000000\n"
        )
        df = load_csv(str(file_path))
        result = clean_column_names(df)
        assert "Date" in result.columns
        assert "Date  " not in result.columns


class TestLoadAndPrepare:
    """Tests for load_and_prepare function (integration)."""

    def test_load_and_prepare_valid(self, valid_csv_file):
        """Test full loading and preparation of valid file."""
        df = load_and_prepare(str(valid_csv_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])
        assert df["Date"].is_monotonic_increasing

    def test_load_and_prepare_unsorted(self, unsorted_dates_file):
        """Test loading and sorting unsorted file."""
        df = load_and_prepare(str(unsorted_dates_file))
        assert df["Date"].is_monotonic_increasing
        assert df.iloc[0]["Date"] == pd.Timestamp("2023-01-01")

    def test_load_and_prepare_whitespace_columns(self, whitespace_columns_file):
        """Test loading file with whitespace in column names."""
        df = load_and_prepare(str(whitespace_columns_file))
        assert "Date" in df.columns
        assert " Date " not in df.columns

    def test_load_and_prepare_nonexistent(self, temp_dir):
        """Test loading nonexistent file."""
        with pytest.raises(CustomFileNotFoundError):
            load_and_prepare(str(temp_dir / "nonexistent.csv"))

    def test_load_and_prepare_empty(self, empty_file):
        """Test loading empty file."""
        with pytest.raises(EmptyFileError):
            load_and_prepare(str(empty_file))

    def test_load_and_prepare_large_dataset(self, large_dataset):
        """Test loading large dataset performs well."""
        df = load_and_prepare(str(large_dataset))
        assert len(df) == 5000
        assert df["Date"].is_monotonic_increasing

    def test_load_and_prepare_with_adj_close(self, adj_close_file):
        """Test loading file with extra columns."""
        df = load_and_prepare(str(adj_close_file))
        assert "Adj Close" in df.columns
        assert len(df) == 3
