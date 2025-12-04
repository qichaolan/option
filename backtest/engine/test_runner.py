"""
Tests for the main runner module.

This module tests the main API and CLI including:
- run_backtest function
- CLI argument parsing
- Integration tests
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from backtest.engine.exceptions import (
    DataError,
    FileNotFoundError,
    InsufficientDataError,
    InvalidParameterError,
)
from backtest.engine.runner import (
    BacktestResult,
    create_parser,
    load_data,
    main,
    run_backtest,
    validate_data,
)


class TestLoadData:
    """Tests for load_data function."""

    def test_load_valid_data(self, sample_indicator_csv):
        """Test loading valid CSV data."""
        df = load_data(str(sample_indicator_csv))
        assert isinstance(df, pd.DataFrame)
        assert "Date" in df.columns
        assert "Close" in df.columns

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_data(str(temp_dir / "nonexistent.csv"))
        assert "nonexistent.csv" in str(exc_info.value)

    def test_load_empty_file(self, temp_dir):
        """Test loading empty file raises error."""
        file_path = temp_dir / "empty.csv"
        file_path.write_text("Date,Close\n")
        with pytest.raises(InsufficientDataError):
            load_data(str(file_path))

    def test_load_missing_date_column(self, temp_dir):
        """Test loading file without Date column raises error."""
        file_path = temp_dir / "no_date.csv"
        file_path.write_text("Price,Volume\n100,1000\n")
        with pytest.raises(DataError) as exc_info:
            load_data(str(file_path))
        assert "Date" in str(exc_info.value)

    def test_load_missing_close_column(self, temp_dir):
        """Test loading file without Close column raises error."""
        file_path = temp_dir / "no_close.csv"
        file_path.write_text("Date,Open\n2023-01-01,100\n")
        with pytest.raises(DataError) as exc_info:
            load_data(str(file_path))
        assert "Close" in str(exc_info.value)

    def test_date_parsing(self, temp_dir):
        """Test date parsing works correctly."""
        file_path = temp_dir / "dates.csv"
        file_path.write_text("Date,Close\n2023-01-15,100\n2023-01-16,101\n")
        df = load_data(str(file_path))
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    def test_load_malformed_csv(self, temp_dir):
        """Test loading malformed CSV raises error."""
        file_path = temp_dir / "malformed.csv"
        # Write binary garbage
        file_path.write_bytes(b"\x00\x01\x02\x03")
        # The file is parsed, but has no proper data, so raises InsufficientDataError
        with pytest.raises(InsufficientDataError):
            load_data(str(file_path))


class TestValidateData:
    """Tests for validate_data function."""

    def test_validate_valid_data(self):
        """Test validation passes for valid data."""
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Close": [100.0, 101.0, 102.0],
        })
        # Should not raise
        validate_data(df)

    def test_validate_duplicate_dates(self):
        """Test validation detects duplicate dates."""
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
            "Close": [100.0, 101.0, 102.0],
        })
        with pytest.raises(DataError) as exc_info:
            validate_data(df)
        assert "Duplicate" in str(exc_info.value) or "duplicate" in str(exc_info.value)

    def test_validate_unsorted_dates_gets_sorted(self):
        """Test unsorted dates get auto-sorted."""
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"]),
            "Close": [103.0, 100.0, 102.0],
        })
        # Should not raise but sort the data
        validate_data(df)
        # Data should now be sorted
        assert df["Date"].is_monotonic_increasing

    def test_validate_nan_in_close(self):
        """Test validation detects NaN in Close column."""
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Close": [100.0, float("nan"), 102.0],
        })
        with pytest.raises(DataError) as exc_info:
            validate_data(df)
        assert "NaN" in str(exc_info.value)

    def test_validate_non_positive_close(self):
        """Test validation detects non-positive Close prices."""
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Close": [100.0, 0.0, 102.0],
        })
        with pytest.raises(DataError) as exc_info:
            validate_data(df)
        assert "non-positive" in str(exc_info.value)

    def test_validate_negative_close(self):
        """Test validation detects negative Close prices."""
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Close": [100.0, -10.0, 102.0],
        })
        with pytest.raises(DataError) as exc_info:
            validate_data(df)
        assert "non-positive" in str(exc_info.value)


class TestRunBacktest:
    """Tests for run_backtest function."""

    def test_run_backtest_basic(self, sample_indicator_csv, simple_strategy_yaml):
        """Test basic backtest run."""
        result = run_backtest(
            data_file=str(sample_indicator_csv),
            strategy_files=str(simple_strategy_yaml),
            initial_capital=100000,
        )

        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 100000
        assert len(result.strategies) == 1
        assert len(result.benchmarks) == 3

    def test_run_backtest_multiple_strategies(
        self, sample_indicator_csv, simple_strategy_yaml, multi_strategy_yaml
    ):
        """Test backtest with multiple strategy files."""
        result = run_backtest(
            data_file=str(sample_indicator_csv),
            strategy_files=[str(simple_strategy_yaml), str(multi_strategy_yaml)],
        )

        assert len(result.strategies) == 3  # 1 + 2

    def test_run_backtest_custom_capital(self, sample_indicator_csv, simple_strategy_yaml):
        """Test backtest with custom initial capital."""
        result = run_backtest(
            data_file=str(sample_indicator_csv),
            strategy_files=str(simple_strategy_yaml),
            initial_capital=50000,
        )

        assert result.initial_capital == 50000
        assert result.portfolio.initial_capital == 50000

    def test_run_backtest_invalid_capital(self, sample_indicator_csv, simple_strategy_yaml):
        """Test backtest with invalid capital raises error."""
        with pytest.raises(InvalidParameterError) as exc_info:
            run_backtest(
                data_file=str(sample_indicator_csv),
                strategy_files=str(simple_strategy_yaml),
                initial_capital=0,
            )
        assert "initial_capital" in str(exc_info.value)

    def test_run_backtest_negative_capital(self, sample_indicator_csv, simple_strategy_yaml):
        """Test backtest with negative capital raises error."""
        with pytest.raises(InvalidParameterError):
            run_backtest(
                data_file=str(sample_indicator_csv),
                strategy_files=str(simple_strategy_yaml),
                initial_capital=-1000,
            )

    def test_run_backtest_with_dataframe(self, sample_indicator_csv, simple_strategy_yaml):
        """Test backtest accepts DataFrame directly."""
        df = pd.read_csv(sample_indicator_csv)
        result = run_backtest(
            data_file=df,
            strategy_files=str(simple_strategy_yaml),
            initial_capital=100000,
        )

        assert isinstance(result, BacktestResult)
        assert result.data_file == "<DataFrame>"
        assert result.portfolio.final_value > 0

    def test_run_backtest_saves_output(
        self, sample_indicator_csv, simple_strategy_yaml, temp_dir
    ):
        """Test backtest saves output files."""
        output_file = temp_dir / "results.csv"
        trades_file = temp_dir / "trades.csv"

        result = run_backtest(
            data_file=str(sample_indicator_csv),
            strategy_files=str(simple_strategy_yaml),
            output_file=str(output_file),
            trades_file=str(trades_file),
        )

        assert output_file.exists()
        df = pd.read_csv(output_file)
        assert "Date" in df.columns
        assert "Portfolio_Value" in df.columns

    def test_run_backtest_date_range(self, sample_indicator_csv, simple_strategy_yaml):
        """Test backtest records date range."""
        result = run_backtest(
            data_file=str(sample_indicator_csv),
            strategy_files=str(simple_strategy_yaml),
        )

        assert result.date_range[0] < result.date_range[1]
        assert result.num_rows > 0


class TestBacktestResult:
    """Tests for BacktestResult class."""

    def test_summary_output(self, sample_indicator_csv, simple_strategy_yaml):
        """Test summary generation."""
        result = run_backtest(
            data_file=str(sample_indicator_csv),
            strategy_files=str(simple_strategy_yaml),
        )

        summary = result.summary()

        assert "BACKTEST RESULTS" in summary
        assert "PERFORMANCE COMPARISON" in summary
        assert "TRADING SUMMARY" in summary
        assert result.data_file in summary

    def test_comparison_dataframe(self, sample_indicator_csv, simple_strategy_yaml):
        """Test comparison DataFrame."""
        result = run_backtest(
            data_file=str(sample_indicator_csv),
            strategy_files=str(simple_strategy_yaml),
        )

        assert not result.comparison.empty
        assert len(result.comparison) == 4  # Strategy + 3 benchmarks
        assert "Strategy" in result.comparison.columns


class TestCreateParser:
    """Tests for CLI argument parser."""

    def test_parser_required_args(self):
        """Test parser requires data and strategies."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])

        with pytest.raises(SystemExit):
            parser.parse_args(["--data", "test.csv"])

    def test_parser_all_args(self):
        """Test parser accepts all arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "--data", "test.csv",
            "--strategies", "strategy.yaml",
            "--capital", "50000",
            "--output", "results.csv",
            "--trades", "trades.csv",
        ])

        assert args.data == "test.csv"
        assert args.strategies == ["strategy.yaml"]
        assert args.capital == 50000
        assert args.output == "results.csv"
        assert args.trades == "trades.csv"

    def test_parser_short_args(self):
        """Test parser accepts short argument forms."""
        parser = create_parser()
        args = parser.parse_args([
            "-d", "test.csv",
            "-s", "s1.yaml", "s2.yaml",
            "-c", "25000",
            "-o", "out.csv",
            "-t", "log.csv",
        ])

        assert args.data == "test.csv"
        assert args.strategies == ["s1.yaml", "s2.yaml"]
        assert args.capital == 25000

    def test_parser_default_capital(self):
        """Test parser default capital."""
        parser = create_parser()
        args = parser.parse_args([
            "-d", "test.csv",
            "-s", "strategy.yaml",
        ])

        assert args.capital == 100000


class TestMain:
    """Tests for main CLI function."""

    def test_main_success(self, sample_indicator_csv, simple_strategy_yaml, capsys):
        """Test successful CLI run."""
        exit_code = main([
            "--data", str(sample_indicator_csv),
            "--strategies", str(simple_strategy_yaml),
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out

    def test_main_file_not_found(self, simple_strategy_yaml, capsys):
        """Test CLI with missing data file."""
        exit_code = main([
            "--data", "nonexistent.csv",
            "--strategies", str(simple_strategy_yaml),
        ])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_invalid_strategy(self, sample_indicator_csv, temp_dir, capsys):
        """Test CLI with invalid strategy file."""
        strategy_file = temp_dir / "bad.yaml"
        strategy_file.write_text("invalid: yaml: content")

        exit_code = main([
            "--data", str(sample_indicator_csv),
            "--strategies", str(strategy_file),
        ])

        assert exit_code == 1

    def test_main_with_output_files(
        self, sample_indicator_csv, simple_strategy_yaml, temp_dir, capsys
    ):
        """Test CLI saves output files."""
        output_file = temp_dir / "results.csv"
        trades_file = temp_dir / "trades.csv"

        exit_code = main([
            "--data", str(sample_indicator_csv),
            "--strategies", str(simple_strategy_yaml),
            "--output", str(output_file),
            "--trades", str(trades_file),
        ])

        assert exit_code == 0
        assert output_file.exists()

        captured = capsys.readouterr()
        assert "results.csv" in captured.out


class TestIntegration:
    """Integration tests for the backtest engine."""

    def test_full_pipeline_trending_data(self, trending_up_data, simple_strategy_yaml, temp_dir):
        """Test full pipeline with trending data."""
        # Save trending data to CSV
        data_file = temp_dir / "trending.csv"
        trending_up_data.to_csv(data_file, index=False)

        result = run_backtest(
            data_file=str(data_file),
            strategy_files=str(simple_strategy_yaml),
            initial_capital=100000,
        )

        # Should have at least one trade
        assert result.portfolio.num_trades > 0
        # Trending up data with RSI strategy should generate signals
        assert "BUY" in result.portfolio.signals
        assert "SELL" in result.portfolio.signals

    def test_full_pipeline_oscillating_data(
        self, oscillating_data, simple_strategy_yaml, temp_dir
    ):
        """Test full pipeline with oscillating data."""
        data_file = temp_dir / "oscillating.csv"
        oscillating_data.to_csv(data_file, index=False)

        result = run_backtest(
            data_file=str(data_file),
            strategy_files=str(simple_strategy_yaml),
        )

        # Multiple cycles should generate multiple trades
        assert result.portfolio.num_buys > 0
        assert result.portfolio.num_sells > 0

    def test_indicator_comparison_strategy(
        self, sample_indicator_csv, indicator_comparison_strategy
    ):
        """Test strategy with indicator-to-indicator comparison."""
        result = run_backtest(
            data_file=str(sample_indicator_csv),
            strategy_files=str(indicator_comparison_strategy),
        )

        # Should complete without error
        assert isinstance(result, BacktestResult)

    def test_multi_strategy_weighting(
        self, sample_indicator_csv, multi_strategy_yaml
    ):
        """Test multi-strategy weighted aggregation."""
        result = run_backtest(
            data_file=str(sample_indicator_csv),
            strategy_files=str(multi_strategy_yaml),
        )

        # Weights should be normalized
        total_weight = sum(s.weight for s in result.strategies)
        assert abs(total_weight - 1.0) < 0.0001

    def test_benchmark_comparison(self, sample_indicator_csv, simple_strategy_yaml):
        """Test benchmark comparison is included."""
        result = run_backtest(
            data_file=str(sample_indicator_csv),
            strategy_files=str(simple_strategy_yaml),
        )

        # Should have 3 benchmarks
        assert len(result.benchmarks) == 3

        # Comparison should include strategy + benchmarks
        assert len(result.comparison) == 4

    def test_unexpected_error(self, sample_indicator_csv, temp_dir, capsys):
        """Test handling of unexpected errors."""
        # Create a strategy that will cause issues by having invalid YAML
        bad_yaml = temp_dir / "bad.yaml"
        bad_yaml.write_text("strategies: [}")  # Invalid YAML

        exit_code = main([
            "-d", str(sample_indicator_csv),
            "-s", str(bad_yaml),
        ])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err or "error" in captured.err.lower()
