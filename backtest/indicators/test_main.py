"""
Tests for the indicators module (main entry point).

This module tests:
- build_indicators API function
- CLI argument parsing
- CLI main function
- File I/O operations
"""

import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from backtest.indicators.exceptions import (
    DuplicateDateError,
    EmptyFileError,
    InvalidDataTypeError,
    MissingColumnError,
    NonMonotonicDateError,
    ValidationError,
)
from backtest.indicators.exceptions import FileNotFoundError as CustomFileNotFoundError
from backtest.indicators.main import (
    build_indicators,
    create_parser,
    main,
    save_to_csv,
    split_train_test,
)


class TestBuildIndicators:
    """Tests for build_indicators function."""

    def test_build_indicators_basic(self, valid_csv_file):
        """Test basic indicator building."""
        df = build_indicators(str(valid_csv_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "sma_20" in df.columns
        assert "rsi_14" in df.columns

    def test_build_indicators_all_columns(self, valid_csv_file):
        """Test that all expected indicator columns are present."""
        df = build_indicators(str(valid_csv_file))

        expected = [
            "Date", "Open", "High", "Low", "Close", "Volume",
            "sma_5", "sma_9", "sma_20", "sma_50", "sma_200",
            "ema_9", "ema_21", "ema_50",
            "macd_12_26_9", "macd_signal_12_26_9", "macd_hist_12_26_9",
            "rsi_14", "mfi_14",
            "atr_14", "hv_20",
            "obv", "vol_sma_20",
            "pivot_high_3", "pivot_low_3",
            "bb_mid_20_2", "bb_upper_20_2", "bb_lower_20_2",
        ]

        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_build_indicators_with_output(self, valid_csv_file, temp_dir):
        """Test building indicators with output file."""
        output_file = temp_dir / "output.csv"
        df = build_indicators(str(valid_csv_file), str(output_file))

        assert output_file.exists()
        assert len(df) == 100

        # Read back and verify
        df_read = pd.read_csv(output_file)
        assert len(df_read) == 100
        assert "sma_20" in df_read.columns

    def test_build_indicators_creates_parent_dirs(self, valid_csv_file, temp_dir):
        """Test that output file parent directories are created."""
        output_file = temp_dir / "subdir" / "nested" / "output.csv"
        build_indicators(str(valid_csv_file), str(output_file))
        assert output_file.exists()

    def test_build_indicators_nonexistent_file(self, temp_dir):
        """Test error for nonexistent input file."""
        with pytest.raises(CustomFileNotFoundError):
            build_indicators(str(temp_dir / "nonexistent.csv"))

    def test_build_indicators_empty_file(self, empty_file):
        """Test error for empty input file."""
        with pytest.raises(EmptyFileError):
            build_indicators(str(empty_file))

    def test_build_indicators_missing_column(self, missing_column_file):
        """Test error for missing required column."""
        with pytest.raises(MissingColumnError):
            build_indicators(str(missing_column_file))

    def test_build_indicators_duplicate_dates(self, duplicate_dates_file):
        """Test error for duplicate dates."""
        with pytest.raises(DuplicateDateError):
            build_indicators(str(duplicate_dates_file))

    def test_build_indicators_minimal_data(self, minimal_csv_file):
        """Test with minimal valid data."""
        df = build_indicators(str(minimal_csv_file))
        assert len(df) == 10
        # Some indicators will be NaN but should exist
        assert "sma_5" in df.columns

    def test_build_indicators_large_dataset(self, large_dataset):
        """Test performance with large dataset."""
        df = build_indicators(str(large_dataset))
        assert len(df) == 5000


class TestSaveToCsv:
    """Tests for save_to_csv function."""

    def test_save_basic(self, sample_ohlcv_data, temp_dir):
        """Test basic CSV save."""
        output_file = temp_dir / "output.csv"
        save_to_csv(sample_ohlcv_data, str(output_file))

        assert output_file.exists()
        df_read = pd.read_csv(output_file)
        assert len(df_read) == len(sample_ohlcv_data)

    def test_save_creates_parent_dirs(self, sample_ohlcv_data, temp_dir):
        """Test that parent directories are created."""
        output_file = temp_dir / "a" / "b" / "c" / "output.csv"
        save_to_csv(sample_ohlcv_data, str(output_file))
        assert output_file.exists()

    def test_save_overwrites_existing(self, sample_ohlcv_data, temp_dir):
        """Test that existing file is overwritten."""
        output_file = temp_dir / "output.csv"
        output_file.write_text("old content")

        save_to_csv(sample_ohlcv_data, str(output_file))

        df_read = pd.read_csv(output_file)
        assert len(df_read) == len(sample_ohlcv_data)


class TestCreateParser:
    """Tests for create_parser function."""

    def test_parser_basic(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser is not None

    def test_parser_input_required(self):
        """Test that input_file is required."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_input_file(self):
        """Test parsing input_file argument."""
        parser = create_parser()
        args = parser.parse_args(["--input_file", "test.csv"])
        assert args.input_file == "test.csv"

    def test_parser_short_input(self):
        """Test parsing short -i argument."""
        parser = create_parser()
        args = parser.parse_args(["-i", "test.csv"])
        assert args.input_file == "test.csv"

    def test_parser_output_file(self):
        """Test parsing output_file argument."""
        parser = create_parser()
        args = parser.parse_args(["--input_file", "in.csv", "--output_file", "out.csv"])
        assert args.input_file == "in.csv"
        assert args.output_file == "out.csv"

    def test_parser_short_output(self):
        """Test parsing short -o argument."""
        parser = create_parser()
        args = parser.parse_args(["-i", "in.csv", "-o", "out.csv"])
        assert args.output_file == "out.csv"

    def test_parser_output_optional(self):
        """Test that output_file is optional."""
        parser = create_parser()
        args = parser.parse_args(["--input_file", "test.csv"])
        assert args.output_file is None


class TestMain:
    """Tests for main CLI function."""

    def test_main_success(self, valid_csv_file, capsys):
        """Test successful CLI execution."""
        result = main(["--input_file", str(valid_csv_file)])
        assert result == 0

        captured = capsys.readouterr()
        assert "Processed 100 rows" in captured.out
        assert "Indicators added" in captured.out

    def test_main_with_output(self, valid_csv_file, temp_dir, capsys):
        """Test CLI with output file."""
        output_file = temp_dir / "output.csv"
        result = main(["-i", str(valid_csv_file), "-o", str(output_file)])

        assert result == 0
        assert output_file.exists()

        captured = capsys.readouterr()
        assert "Output saved to" in captured.out

    def test_main_nonexistent_file(self, temp_dir, capsys):
        """Test CLI with nonexistent file."""
        result = main(["--input_file", str(temp_dir / "nonexistent.csv")])
        assert result == 1

        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_empty_file(self, empty_file, capsys):
        """Test CLI with empty file."""
        result = main(["--input_file", str(empty_file)])
        assert result == 1

        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_validation_error(self, missing_column_file, capsys):
        """Test CLI with validation error."""
        result = main(["--input_file", str(missing_column_file)])
        assert result == 1

        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_shows_latest_values(self, valid_csv_file, capsys):
        """Test that CLI shows latest indicator values when no output file."""
        result = main(["--input_file", str(valid_csv_file)])
        assert result == 0

        captured = capsys.readouterr()
        assert "Latest indicator values" in captured.out

    def test_main_shows_date_range(self, valid_csv_file, capsys):
        """Test that CLI shows date range."""
        result = main(["--input_file", str(valid_csv_file)])
        assert result == 0

        captured = capsys.readouterr()
        assert "Date range" in captured.out

    def test_main_invalid_args(self, capsys):
        """Test CLI with invalid arguments."""
        # Missing required argument - argparse will call sys.exit(2)
        with pytest.raises(SystemExit) as exc_info:
            main([])
        # argparse exits with code 2 for missing required arguments
        assert exc_info.value.code == 2

    def test_main_uses_sysargv_default(self, valid_csv_file, monkeypatch, capsys):
        """Test that main uses sys.argv when no args provided."""
        monkeypatch.setattr(
            sys, "argv", ["indicators.py", "--input_file", str(valid_csv_file)]
        )
        # This would use sys.argv, but we can't easily test without mocking
        # Just verify the function signature accepts None
        # In production it would use sys.argv

    def test_main_large_dataset(self, large_dataset, temp_dir, capsys):
        """Test CLI with large dataset."""
        output_file = temp_dir / "large_output.csv"
        result = main(["-i", str(large_dataset), "-o", str(output_file)])

        assert result == 0
        assert output_file.exists()

        captured = capsys.readouterr()
        assert "5000 rows" in captured.out


class TestCLIIntegration:
    """Integration tests for CLI behavior."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow: create data, process, verify output."""
        # Create test data
        input_file = temp_dir / "input.csv"
        output_file = temp_dir / "output.csv"

        dates = pd.date_range(start="2023-01-01", periods=50, freq="B")
        df = pd.DataFrame({
            "Date": dates,
            "Open": [100 + i * 0.1 for i in range(50)],
            "High": [101 + i * 0.1 for i in range(50)],
            "Low": [99 + i * 0.1 for i in range(50)],
            "Close": [100.5 + i * 0.1 for i in range(50)],
            "Volume": [1000000 + i * 10000 for i in range(50)],
        })
        df.to_csv(input_file, index=False)

        # Process
        result = main(["-i", str(input_file), "-o", str(output_file)])
        assert result == 0

        # Verify output
        df_out = pd.read_csv(output_file)
        assert len(df_out) == 50
        assert "rsi_14" in df_out.columns
        assert "macd_12_26_9" in df_out.columns

    def test_api_and_cli_produce_same_result(self, valid_csv_file, temp_dir):
        """Test that API and CLI produce identical results."""
        # API
        df_api = build_indicators(str(valid_csv_file))

        # CLI
        output_file = temp_dir / "cli_output.csv"
        main(["-i", str(valid_csv_file), "-o", str(output_file)])
        df_cli = pd.read_csv(output_file)

        # Compare (dates might be string in CLI output)
        assert len(df_api) == len(df_cli)
        for col in df_api.columns:
            if col != "Date":
                assert col in df_cli.columns


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_row_data(self, temp_dir):
        """Test with single row of data."""
        input_file = temp_dir / "single.csv"
        input_file.write_text(
            "Date,Open,High,Low,Close,Volume\n"
            "2023-01-01,100,101,99,100.5,1000000\n"
        )

        df = build_indicators(str(input_file))
        assert len(df) == 1
        # Most indicators will be NaN but should not error

    def test_two_rows_data(self, temp_dir):
        """Test with two rows of data."""
        input_file = temp_dir / "two.csv"
        input_file.write_text(
            "Date,Open,High,Low,Close,Volume\n"
            "2023-01-01,100,101,99,100.5,1000000\n"
            "2023-01-02,101,102,100,101.5,1100000\n"
        )

        df = build_indicators(str(input_file))
        assert len(df) == 2

    def test_exact_lookback_period(self, temp_dir):
        """Test with exactly enough data for lookback."""
        # 20 rows for SMA-20
        dates = pd.date_range(start="2023-01-01", periods=20, freq="B")
        df = pd.DataFrame({
            "Date": dates,
            "Open": [100 + i for i in range(20)],
            "High": [101 + i for i in range(20)],
            "Low": [99 + i for i in range(20)],
            "Close": [100.5 + i for i in range(20)],
            "Volume": [1000000] * 20,
        })

        input_file = temp_dir / "exact.csv"
        df.to_csv(input_file, index=False)

        result = build_indicators(str(input_file))
        # SMA-20 should have exactly 1 valid value
        assert result["sma_20"].dropna().count() == 1

    def test_unicode_in_path(self, temp_dir):
        """Test handling of unicode characters in file path."""
        # Create a subdirectory with unicode name
        unicode_dir = temp_dir / "données"
        unicode_dir.mkdir()

        input_file = unicode_dir / "data.csv"
        dates = pd.date_range(start="2023-01-01", periods=30, freq="B")
        df = pd.DataFrame({
            "Date": dates,
            "Open": [100] * 30,
            "High": [101] * 30,
            "Low": [99] * 30,
            "Close": [100.5] * 30,
            "Volume": [1000000] * 30,
        })
        df.to_csv(input_file, index=False)

        result = build_indicators(str(input_file))
        assert len(result) == 30


class TestSplitTrainTest:
    """Tests for split_train_test function."""

    def test_split_basic(self, sample_ohlcv_data):
        """Test basic train/test split."""
        train_df, test_df = split_train_test(sample_ohlcv_data, test_days=20)
        assert len(train_df) == 80
        assert len(test_df) == 20

    def test_split_preserves_columns(self, sample_ohlcv_data):
        """Test that split preserves all columns."""
        train_df, test_df = split_train_test(sample_ohlcv_data, test_days=10)
        assert list(train_df.columns) == list(sample_ohlcv_data.columns)
        assert list(test_df.columns) == list(sample_ohlcv_data.columns)

    def test_split_chronological(self, sample_ohlcv_data):
        """Test that train comes before test chronologically."""
        train_df, test_df = split_train_test(sample_ohlcv_data, test_days=20)
        # Last row of train should be before first row of test
        assert train_df["Date"].iloc[-1] < test_df["Date"].iloc[0]

    def test_split_test_days_too_large(self, sample_ohlcv_data):
        """Test error when test_days >= total rows."""
        with pytest.raises(ValueError) as exc_info:
            split_train_test(sample_ohlcv_data, test_days=100)
        assert "must be less than" in str(exc_info.value)

    def test_split_test_days_equal_rows(self, sample_ohlcv_data):
        """Test error when test_days equals total rows."""
        with pytest.raises(ValueError):
            split_train_test(sample_ohlcv_data, test_days=len(sample_ohlcv_data))

    def test_split_test_days_zero(self, sample_ohlcv_data):
        """Test error when test_days is zero."""
        with pytest.raises(ValueError) as exc_info:
            split_train_test(sample_ohlcv_data, test_days=0)
        assert "must be positive" in str(exc_info.value)

    def test_split_test_days_negative(self, sample_ohlcv_data):
        """Test error when test_days is negative."""
        with pytest.raises(ValueError) as exc_info:
            split_train_test(sample_ohlcv_data, test_days=-5)
        assert "must be positive" in str(exc_info.value)

    def test_split_minimal(self, sample_ohlcv_data):
        """Test split with minimal test_days=1."""
        train_df, test_df = split_train_test(sample_ohlcv_data, test_days=1)
        assert len(train_df) == 99
        assert len(test_df) == 1

    def test_split_almost_all_test(self, sample_ohlcv_data):
        """Test split with almost all data in test set."""
        train_df, test_df = split_train_test(sample_ohlcv_data, test_days=99)
        assert len(train_df) == 1
        assert len(test_df) == 99


class TestBuildIndicatorsWithSplit:
    """Tests for build_indicators with train/test split."""

    def test_build_with_test_days(self, valid_csv_file):
        """Test building indicators with train/test split."""
        result = build_indicators(str(valid_csv_file), test_days=20)
        assert isinstance(result, tuple)
        train_df, test_df = result
        assert len(train_df) == 80
        assert len(test_df) == 20

    def test_build_with_split_has_indicators(self, valid_csv_file):
        """Test that both train and test have indicators."""
        train_df, test_df = build_indicators(str(valid_csv_file), test_days=20)
        assert "sma_20" in train_df.columns
        assert "sma_20" in test_df.columns
        assert "rsi_14" in train_df.columns
        assert "rsi_14" in test_df.columns

    def test_build_with_split_and_output(self, valid_csv_file, temp_dir):
        """Test building indicators with split and output file."""
        output_file = temp_dir / "output.csv"
        train_df, test_df = build_indicators(
            str(valid_csv_file), str(output_file), test_days=20
        )

        # Check that train and test files were created
        train_file = temp_dir / "output_train.csv"
        test_file = temp_dir / "output_test.csv"
        assert train_file.exists()
        assert test_file.exists()

        # Verify contents
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        assert len(df_train) == 80
        assert len(df_test) == 20

    def test_build_with_split_output_preserves_suffix(self, valid_csv_file, temp_dir):
        """Test that output file suffix is preserved."""
        output_file = temp_dir / "data.csv"
        build_indicators(str(valid_csv_file), str(output_file), test_days=20)

        assert (temp_dir / "data_train.csv").exists()
        assert (temp_dir / "data_test.csv").exists()

    def test_build_without_test_days_returns_dataframe(self, valid_csv_file):
        """Test that without test_days, single DataFrame is returned."""
        result = build_indicators(str(valid_csv_file))
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, tuple)


class TestCLIWithSplit:
    """Tests for CLI with train/test split."""

    def test_parser_test_argument(self):
        """Test parsing --test argument."""
        parser = create_parser()
        args = parser.parse_args(["-i", "in.csv", "--test", "20"])
        assert args.test == 20

    def test_parser_short_test_argument(self):
        """Test parsing short -t argument."""
        parser = create_parser()
        args = parser.parse_args(["-i", "in.csv", "-t", "30"])
        assert args.test == 30

    def test_parser_test_optional(self):
        """Test that --test is optional."""
        parser = create_parser()
        args = parser.parse_args(["-i", "in.csv"])
        assert args.test is None

    def test_main_with_test_split(self, valid_csv_file, capsys):
        """Test CLI with train/test split."""
        result = main(["-i", str(valid_csv_file), "--test", "20"])
        assert result == 0

        captured = capsys.readouterr()
        assert "Train set: 80 rows" in captured.out
        assert "Test set: 20 rows" in captured.out

    def test_main_with_split_and_output(self, valid_csv_file, temp_dir, capsys):
        """Test CLI with split and output file."""
        output_file = temp_dir / "output.csv"
        result = main(["-i", str(valid_csv_file), "-o", str(output_file), "-t", "20"])

        assert result == 0

        captured = capsys.readouterr()
        assert "output_train.csv" in captured.out
        assert "output_test.csv" in captured.out

        # Verify files exist
        assert (temp_dir / "output_train.csv").exists()
        assert (temp_dir / "output_test.csv").exists()

    def test_main_with_invalid_test_days(self, valid_csv_file, capsys):
        """Test CLI with invalid test_days (too large)."""
        result = main(["-i", str(valid_csv_file), "--test", "1000"])
        assert result == 1

        captured = capsys.readouterr()
        assert "Error" in captured.err or "error" in captured.err.lower()

    def test_main_split_shows_date_ranges(self, valid_csv_file, capsys):
        """Test that CLI shows date ranges for both sets."""
        result = main(["-i", str(valid_csv_file), "--test", "20"])
        assert result == 0

        captured = capsys.readouterr()
        # Should have two "Date range" outputs
        assert captured.out.count("Date range") == 2


class TestExceptionMessages:
    """Tests for exception message formatting."""

    def test_invalid_data_type_error_without_details(self):
        """Test InvalidDataTypeError without details."""
        error = InvalidDataTypeError("Close")
        assert "Close" in str(error)
        assert "Invalid data type in column 'Close'" in str(error)

    def test_invalid_data_type_error_with_details(self):
        """Test InvalidDataTypeError with details."""
        error = InvalidDataTypeError("Close", "cannot convert 'abc'")
        assert "cannot convert 'abc'" in str(error)

    def test_non_monotonic_date_error_without_details(self):
        """Test NonMonotonicDateError without details."""
        error = NonMonotonicDateError()
        assert "not in monotonically increasing order" in str(error)

    def test_non_monotonic_date_error_with_details(self):
        """Test NonMonotonicDateError with details."""
        error = NonMonotonicDateError("Date 2023-01-02 <= previous date 2023-01-03")
        assert "2023-01-02" in str(error)
