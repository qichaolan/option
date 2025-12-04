"""
Tests for the validators module.

This module tests all data validation functionality including:
- Empty data validation
- Required column validation
- Numeric type validation
- Missing value detection
- Duplicate date detection
- Chronological order validation
"""

import pandas as pd
import pytest

from backtest.indicators.exceptions import (
    DuplicateDateError,
    EmptyDataError,
    InvalidDataTypeError,
    MissingColumnError,
    NonMonotonicDateError,
)
from backtest.indicators.validators import (
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    validate_all,
    validate_chronological_order,
    validate_no_duplicate_dates,
    validate_no_missing_ohlcv,
    validate_not_empty,
    validate_numeric_columns,
    validate_required_columns,
)


class TestValidateNotEmpty:
    """Tests for validate_not_empty function."""

    def test_valid_non_empty(self, sample_ohlcv_data):
        """Test that non-empty DataFrame passes validation."""
        validate_not_empty(sample_ohlcv_data)  # Should not raise

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises EmptyDataError."""
        df = pd.DataFrame()
        with pytest.raises(EmptyDataError):
            validate_not_empty(df)

    def test_zero_rows(self):
        """Test that DataFrame with zero rows raises EmptyDataError."""
        df = pd.DataFrame(columns=["Date", "Close"])
        with pytest.raises(EmptyDataError):
            validate_not_empty(df)

    def test_single_row(self, minimal_ohlcv_data):
        """Test that single-row DataFrame passes."""
        single_row = minimal_ohlcv_data.head(1)
        validate_not_empty(single_row)  # Should not raise


class TestValidateRequiredColumns:
    """Tests for validate_required_columns function."""

    def test_all_columns_present(self, sample_ohlcv_data):
        """Test that all required columns pass validation."""
        validate_required_columns(sample_ohlcv_data)  # Should not raise

    def test_missing_single_column(self, sample_ohlcv_data):
        """Test that missing single column raises MissingColumnError."""
        df = sample_ohlcv_data.drop(columns=["Volume"])
        with pytest.raises(MissingColumnError) as exc_info:
            validate_required_columns(df)
        assert "Volume" in exc_info.value.missing_columns

    def test_missing_multiple_columns(self, sample_ohlcv_data):
        """Test that missing multiple columns raises MissingColumnError."""
        df = sample_ohlcv_data.drop(columns=["Open", "High", "Volume"])
        with pytest.raises(MissingColumnError) as exc_info:
            validate_required_columns(df)
        assert set(exc_info.value.missing_columns) == {"Open", "High", "Volume"}

    def test_missing_date_column(self, sample_ohlcv_data):
        """Test that missing Date column raises error."""
        df = sample_ohlcv_data.drop(columns=["Date"])
        with pytest.raises(MissingColumnError) as exc_info:
            validate_required_columns(df)
        assert "Date" in exc_info.value.missing_columns

    def test_extra_columns_allowed(self, sample_ohlcv_data):
        """Test that extra columns are allowed."""
        df = sample_ohlcv_data.copy()
        df["ExtraColumn"] = 123
        validate_required_columns(df)  # Should not raise

    def test_required_columns_constant(self):
        """Test that REQUIRED_COLUMNS contains expected values."""
        assert "Date" in REQUIRED_COLUMNS
        assert "Open" in REQUIRED_COLUMNS
        assert "High" in REQUIRED_COLUMNS
        assert "Low" in REQUIRED_COLUMNS
        assert "Close" in REQUIRED_COLUMNS
        assert "Volume" in REQUIRED_COLUMNS


class TestValidateNumericColumns:
    """Tests for validate_numeric_columns function."""

    def test_valid_numeric_data(self, sample_ohlcv_data):
        """Test that valid numeric data passes validation."""
        result = validate_numeric_columns(sample_ohlcv_data)
        for col in NUMERIC_COLUMNS:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_string_values_fail(self):
        """Test that non-numeric strings raise InvalidDataTypeError."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01"],
                "Open": ["abc"],
                "High": [101],
                "Low": [99],
                "Close": [100.5],
                "Volume": [1000000],
            }
        )
        with pytest.raises(InvalidDataTypeError) as exc_info:
            validate_numeric_columns(df)
        assert exc_info.value.column == "Open"
        assert "abc" in str(exc_info.value)

    def test_mixed_valid_invalid(self):
        """Test that mixed valid/invalid values raise error."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "Open": [100, "invalid"],
                "High": [101, 102],
                "Low": [99, 100],
                "Close": [100.5, 101.5],
                "Volume": [1000000, 1100000],
            }
        )
        with pytest.raises(InvalidDataTypeError) as exc_info:
            validate_numeric_columns(df)
        assert exc_info.value.column == "Open"

    def test_numeric_strings_convert(self):
        """Test that numeric strings are converted successfully."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01"],
                "Open": ["100.5"],
                "High": ["101"],
                "Low": ["99"],
                "Close": ["100.5"],
                "Volume": ["1000000"],
            }
        )
        result = validate_numeric_columns(df)
        assert result["Open"].iloc[0] == 100.5
        assert result["Volume"].iloc[0] == 1000000

    def test_preserves_original(self, sample_ohlcv_data):
        """Test that original DataFrame is not modified."""
        original_values = sample_ohlcv_data["Close"].copy()
        validate_numeric_columns(sample_ohlcv_data)
        assert (sample_ohlcv_data["Close"] == original_values).all()

    def test_missing_column_skipped(self):
        """Test that missing columns are skipped."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01"],
                "Close": [100.5],
            }
        )
        result = validate_numeric_columns(df)
        assert "Close" in result.columns


class TestValidateNoMissingOhlcv:
    """Tests for validate_no_missing_ohlcv function."""

    def test_no_missing_values(self, sample_ohlcv_data):
        """Test that data without missing values passes."""
        validate_no_missing_ohlcv(sample_ohlcv_data)  # Should not raise

    def test_missing_open(self):
        """Test that missing Open values raise error."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "Open": [100, float("nan")],
                "High": [101, 102],
                "Low": [99, 100],
                "Close": [100.5, 101.5],
                "Volume": [1000000, 1100000],
            }
        )
        with pytest.raises(InvalidDataTypeError) as exc_info:
            validate_no_missing_ohlcv(df)
        assert exc_info.value.column == "Open"
        assert "1 missing" in str(exc_info.value)

    def test_missing_volume(self):
        """Test that missing Volume values raise error."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000000, float("nan"), float("nan")],
            }
        )
        with pytest.raises(InvalidDataTypeError) as exc_info:
            validate_no_missing_ohlcv(df)
        assert exc_info.value.column == "Volume"
        assert "2 missing" in str(exc_info.value)

    def test_multiple_missing_columns(self):
        """Test that first missing column is reported."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01"],
                "Open": [float("nan")],
                "High": [float("nan")],
                "Low": [99],
                "Close": [100.5],
                "Volume": [1000000],
            }
        )
        with pytest.raises(InvalidDataTypeError) as exc_info:
            validate_no_missing_ohlcv(df)
        # Should report first column with missing values
        assert exc_info.value.column in ["Open", "High"]


class TestValidateNoDuplicateDates:
    """Tests for validate_no_duplicate_dates function."""

    def test_no_duplicates(self, sample_ohlcv_data):
        """Test that data without duplicates passes."""
        validate_no_duplicate_dates(sample_ohlcv_data)  # Should not raise

    def test_single_duplicate(self):
        """Test that duplicate dates raise DuplicateDateError."""
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
                "Close": [100, 101, 102],
            }
        )
        with pytest.raises(DuplicateDateError) as exc_info:
            validate_no_duplicate_dates(df)
        assert pd.Timestamp("2023-01-01") in exc_info.value.duplicate_dates

    def test_multiple_duplicates(self):
        """Test that multiple duplicate dates are detected."""
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]
                ),
                "Close": [100, 101, 102, 103],
            }
        )
        with pytest.raises(DuplicateDateError) as exc_info:
            validate_no_duplicate_dates(df)
        assert len(exc_info.value.duplicate_dates) == 2

    def test_missing_date_column(self):
        """Test that missing Date column is handled."""
        df = pd.DataFrame({"Close": [100, 101, 102]})
        validate_no_duplicate_dates(df)  # Should not raise

    def test_duplicate_dates_error_message(self):
        """Test that error message is properly formatted."""
        dates = pd.to_datetime(["2023-01-01"] * 10)
        df = pd.DataFrame({"Date": dates, "Close": range(10)})
        with pytest.raises(DuplicateDateError) as exc_info:
            validate_no_duplicate_dates(df)
        error_message = str(exc_info.value)
        assert "Duplicate dates found" in error_message

    def test_many_duplicate_dates_truncated(self):
        """Test that >5 duplicate dates shows 'and X more' in message."""
        # Create 10 different dates, each duplicated
        dates = []
        for i in range(10):
            dates.extend([f"2023-01-{i+1:02d}"] * 2)
        df = pd.DataFrame({
            "Date": pd.to_datetime(dates),
            "Close": range(20),
        })
        with pytest.raises(DuplicateDateError) as exc_info:
            validate_no_duplicate_dates(df)
        error_message = str(exc_info.value)
        # Should show first 5 and "and X more"
        assert "and" in error_message and "more" in error_message


class TestValidateChronologicalOrder:
    """Tests for validate_chronological_order function."""

    def test_chronological_order(self, sample_ohlcv_data):
        """Test that chronologically ordered data passes."""
        validate_chronological_order(sample_ohlcv_data)  # Should not raise

    def test_non_chronological(self):
        """Test that non-chronological data raises error."""
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-02", "2023-01-01", "2023-01-03"]),
                "Close": [101, 100, 102],
            }
        )
        with pytest.raises(NonMonotonicDateError):
            validate_chronological_order(df)

    def test_equal_dates(self):
        """Test that equal consecutive dates raise error."""
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
                "Close": [100, 101, 102],
            }
        )
        with pytest.raises(NonMonotonicDateError) as exc_info:
            validate_chronological_order(df)
        assert "<= previous date" in str(exc_info.value)

    def test_missing_date_column(self):
        """Test that missing Date column is handled."""
        df = pd.DataFrame({"Close": [100, 101, 102]})
        validate_chronological_order(df)  # Should not raise

    def test_single_row(self, minimal_ohlcv_data):
        """Test that single row data passes."""
        single_row = minimal_ohlcv_data.head(1)
        validate_chronological_order(single_row)  # Should not raise

    def test_two_rows_valid(self):
        """Test that two chronological rows pass."""
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "Close": [100, 101],
            }
        )
        validate_chronological_order(df)  # Should not raise


class TestValidateAll:
    """Tests for validate_all function (integration)."""

    def test_valid_data(self, sample_ohlcv_data):
        """Test that valid data passes all validation."""
        result = validate_all(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)

    def test_empty_data_first(self):
        """Test that empty data is checked first."""
        df = pd.DataFrame()
        with pytest.raises(EmptyDataError):
            validate_all(df)

    def test_missing_columns_second(self):
        """Test that missing columns are checked after empty."""
        df = pd.DataFrame({"Close": [100, 101, 102]})
        with pytest.raises(MissingColumnError):
            validate_all(df)

    def test_invalid_types_third(self):
        """Test that invalid types are checked after columns."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "Open": ["abc", 101],
                "High": [101, 102],
                "Low": [99, 100],
                "Close": [100.5, 101.5],
                "Volume": [1000000, 1100000],
            }
        )
        with pytest.raises(InvalidDataTypeError):
            validate_all(df)

    def test_returns_typed_dataframe(self, sample_ohlcv_data):
        """Test that returned DataFrame has proper types."""
        result = validate_all(sample_ohlcv_data)
        for col in NUMERIC_COLUMNS:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_preserves_extra_columns(self, sample_ohlcv_data):
        """Test that extra columns are preserved."""
        df = sample_ohlcv_data.copy()
        df["ExtraColumn"] = "extra"
        result = validate_all(df)
        assert "ExtraColumn" in result.columns
        assert result["ExtraColumn"].iloc[0] == "extra"

    def test_duplicate_dates_error(self):
        """Test that duplicate dates raise error in validate_all."""
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000000, 1100000, 1200000],
            }
        )
        with pytest.raises(DuplicateDateError):
            validate_all(df)

    def test_missing_values_error(self):
        """Test that missing OHLCV values raise error in validate_all."""
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "Open": [100, float("nan")],
                "High": [101, 102],
                "Low": [99, 100],
                "Close": [100.5, 101.5],
                "Volume": [1000000, 1100000],
            }
        )
        with pytest.raises(InvalidDataTypeError) as exc_info:
            validate_all(df)
        assert "missing" in str(exc_info.value)
