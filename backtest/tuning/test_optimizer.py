"""
Tests for grid search optimizer.

This module tests:
- Parameter combination generation
- Single backtest execution
- Full parameter search workflow
- CLI interface
- Result ranking and output
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from backtest.tuning.exceptions import TuningError
from backtest.tuning.optimizer import (
    ParameterResult,
    SearchResult,
    build_path_mapping,
    create_parser,
    generate_parameter_combinations,
    main,
    run_parameter_search,
    run_single_backtest,
)
from backtest.tuning.param_config import ParamConfig, ParameterSpec


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_indicator_csv(temp_dir):
    """Create sample indicator data CSV."""
    # Create 100 days of data with clear trends
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # Create trending data
    base_price = 100
    prices = [base_price + i * 0.5 + (i % 10 - 5) * 0.2 for i in range(100)]

    df = pd.DataFrame({
        "Date": dates,
        "Open": [p - 0.5 for p in prices],
        "High": [p + 1 for p in prices],
        "Low": [p - 1 for p in prices],
        "Close": prices,
        "Volume": [1000000 + i * 1000 for i in range(100)],
    })

    # Add indicators
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_21"] = df["Close"].ewm(span=21).mean()

    # Add RSI (simplified calculation)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Fill NaN values
    df = df.bfill().ffill()

    file_path = temp_dir / "indicators.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def simple_strategy_yaml(temp_dir):
    """Create simple strategy YAML file."""
    yaml_content = """
strategies:
  - name: "RSI_Strategy"
    weight: 1.0
    combine: "all"
    rules:
      - indicator: "rsi_14"
        operator: "<"
        value: 30
        action: "buy"
        strength: 1.0
      - indicator: "rsi_14"
        operator: ">"
        value: 70
        action: "sell"
        strength: 1.0
"""
    file_path = temp_dir / "strategy.yaml"
    file_path.write_text(yaml_content)
    return file_path


@pytest.fixture
def param_config_yaml(temp_dir):
    """Create parameter config YAML file."""
    yaml_content = """
parameters:
  - name: rsi_oversold
    path: strategies[0].rules[0].value
    start: 25
    end: 35
    step: 5

  - name: rsi_overbought
    path: strategies[0].rules[1].value
    start: 65
    end: 75
    step: 5
"""
    file_path = temp_dir / "params.yaml"
    file_path.write_text(yaml_content)
    return file_path


class TestParameterResult:
    """Tests for ParameterResult dataclass."""

    def test_create_result(self):
        """Test creating parameter result."""
        result = ParameterResult(
            parameters={"rsi_oversold": 30, "rsi_overbought": 70},
            strategy_final_value=105000.0,
            total_return_pct=5.0,
            excess_vs_lumpsum=2.0,
            excess_vs_dca=1.5,
            num_trades=10,
            num_buys=5,
            num_sells=5,
        )

        assert result.parameters["rsi_oversold"] == 30
        assert result.strategy_final_value == 105000.0

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ParameterResult(
            parameters={"rsi_oversold": 30},
            strategy_final_value=105000.0,
            total_return_pct=5.0,
            excess_vs_lumpsum=2.0,
            excess_vs_dca=1.5,
            num_trades=10,
            num_buys=5,
            num_sells=5,
        )

        d = result.to_dict()
        assert "parameters" in d
        assert d["strategy_final_value"] == 105000.0


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self):
        """Test creating search result."""
        top_result = ParameterResult(
            parameters={"rsi_oversold": 30},
            strategy_final_value=110000.0,
            total_return_pct=10.0,
            excess_vs_lumpsum=5.0,
            excess_vs_dca=3.0,
            num_trades=15,
            num_buys=8,
            num_sells=7,
        )

        result = SearchResult(
            top_results=[top_result],
            total_combinations=10,
            evaluated_combinations=10,
            best_parameters={"rsi_oversold": 30},
            best_final_value=110000.0,
            search_space={"rsi_oversold": [25, 30, 35]},
        )

        assert result.total_combinations == 10
        assert result.best_final_value == 110000.0

    def test_summary_output(self):
        """Test summary string generation."""
        top_result = ParameterResult(
            parameters={"rsi_oversold": 30},
            strategy_final_value=110000.0,
            total_return_pct=10.0,
            excess_vs_lumpsum=5.0,
            excess_vs_dca=3.0,
            num_trades=15,
            num_buys=8,
            num_sells=7,
        )

        result = SearchResult(
            top_results=[top_result],
            total_combinations=10,
            evaluated_combinations=10,
            best_parameters={"rsi_oversold": 30},
            best_final_value=110000.0,
            search_space={"rsi_oversold": [25, 30, 35]},
        )

        summary = result.summary()
        assert "PARAMETER SEARCH RESULTS" in summary
        assert "TOP PERFORMING" in summary
        assert "BEST PARAMETERS" in summary
        assert "110,000" in summary

    def test_to_dict(self):
        """Test converting search result to dictionary."""
        top_result = ParameterResult(
            parameters={"rsi_oversold": 30},
            strategy_final_value=110000.0,
            total_return_pct=10.0,
            excess_vs_lumpsum=5.0,
            excess_vs_dca=3.0,
            num_trades=15,
            num_buys=8,
            num_sells=7,
        )

        result = SearchResult(
            top_results=[top_result],
            total_combinations=10,
            evaluated_combinations=10,
            best_parameters={"rsi_oversold": 30},
            best_final_value=110000.0,
            search_space={"rsi_oversold": [25, 30, 35]},
        )

        d = result.to_dict()
        assert "top_results" in d
        assert "best_parameters" in d
        assert len(d["top_results"]) == 1


class TestGenerateParameterCombinations:
    """Tests for generate_parameter_combinations function."""

    def test_single_parameter(self):
        """Test generating combinations for single parameter."""
        specs = [
            ParameterSpec("p1", "path1", 10.0, 30.0, 10.0),
        ]
        config = ParamConfig(parameters=specs)

        combos = generate_parameter_combinations(config)

        assert len(combos) == 3
        assert {"p1": 10.0} in combos
        assert {"p1": 20.0} in combos
        assert {"p1": 30.0} in combos

    def test_multiple_parameters(self):
        """Test generating combinations for multiple parameters."""
        specs = [
            ParameterSpec("p1", "path1", 10.0, 20.0, 10.0),  # 2 values
            ParameterSpec("p2", "path2", 50.0, 70.0, 10.0),  # 3 values
        ]
        config = ParamConfig(parameters=specs)

        combos = generate_parameter_combinations(config)

        assert len(combos) == 6  # 2 * 3
        assert {"p1": 10.0, "p2": 50.0} in combos
        assert {"p1": 10.0, "p2": 60.0} in combos
        assert {"p1": 10.0, "p2": 70.0} in combos
        assert {"p1": 20.0, "p2": 50.0} in combos
        assert {"p1": 20.0, "p2": 60.0} in combos
        assert {"p1": 20.0, "p2": 70.0} in combos

    def test_empty_parameters(self):
        """Test generating combinations with no parameters."""
        config = ParamConfig(parameters=[])
        combos = generate_parameter_combinations(config)

        assert len(combos) == 1
        assert combos[0] == {}


class TestBuildPathMapping:
    """Tests for build_path_mapping function."""

    def test_build_mapping(self):
        """Test building path mapping from params."""
        specs = [
            ParameterSpec("rsi_oversold", "strategies[0].rules[0].value", 20.0, 40.0, 5.0),
            ParameterSpec("rsi_overbought", "strategies[0].rules[1].value", 60.0, 80.0, 5.0),
        ]
        config = ParamConfig(parameters=specs)

        param_values = {"rsi_oversold": 30, "rsi_overbought": 70}
        mapping = build_path_mapping(config, param_values)

        assert mapping["strategies[0].rules[0].value"] == 30
        assert mapping["strategies[0].rules[1].value"] == 70

    def test_build_mapping_partial(self):
        """Test building mapping with partial params."""
        specs = [
            ParameterSpec("p1", "path1", 10.0, 20.0, 5.0),
            ParameterSpec("p2", "path2", 50.0, 60.0, 5.0),
        ]
        config = ParamConfig(parameters=specs)

        # Only provide one parameter
        param_values = {"p1": 15}
        mapping = build_path_mapping(config, param_values)

        assert mapping["path1"] == 15
        assert "path2" not in mapping


class TestRunSingleBacktest:
    """Tests for run_single_backtest function."""

    def test_run_backtest(self, sample_indicator_csv, simple_strategy_yaml, temp_dir):
        """Test running single backtest."""
        from backtest.tuning.yaml_utils import load_yaml

        strategy_yaml = load_yaml(str(simple_strategy_yaml))

        final_value, metrics = run_single_backtest(
            data=str(sample_indicator_csv),
            strategy_yaml=strategy_yaml,
            initial_capital=100000.0,
            temp_dir=temp_dir,
        )

        assert final_value > 0
        assert "strategy_final_value" in metrics
        assert "total_return_pct" in metrics
        assert "num_trades" in metrics


class TestRunParameterSearch:
    """Tests for run_parameter_search function."""

    def test_basic_search(
        self, sample_indicator_csv, simple_strategy_yaml, param_config_yaml
    ):
        """Test basic parameter search."""
        result = run_parameter_search(
            data_path=str(sample_indicator_csv),
            strategy_template_path=str(simple_strategy_yaml),
            param_config_path=str(param_config_yaml),
            initial_capital=100000.0,
            top_n=3,
        )

        assert isinstance(result, SearchResult)
        assert result.total_combinations == 9  # 3 * 3
        assert len(result.top_results) <= 3
        assert result.best_final_value > 0

    def test_search_with_verbose(
        self, sample_indicator_csv, simple_strategy_yaml, param_config_yaml, capsys
    ):
        """Test parameter search with verbose output."""
        result = run_parameter_search(
            data_path=str(sample_indicator_csv),
            strategy_template_path=str(simple_strategy_yaml),
            param_config_path=str(param_config_yaml),
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "Starting parameter search" in captured.out
        assert "Completed" in captured.out

    def test_search_results_sorted(
        self, sample_indicator_csv, simple_strategy_yaml, param_config_yaml
    ):
        """Test that results are sorted by final value."""
        result = run_parameter_search(
            data_path=str(sample_indicator_csv),
            strategy_template_path=str(simple_strategy_yaml),
            param_config_path=str(param_config_yaml),
            top_n=5,
        )

        # Results should be sorted by final value (descending)
        for i in range(len(result.top_results) - 1):
            assert (
                result.top_results[i].strategy_final_value
                >= result.top_results[i + 1].strategy_final_value
            )

    def test_search_file_not_found(self, sample_indicator_csv, simple_strategy_yaml):
        """Test error when param config not found."""
        with pytest.raises(FileNotFoundError):
            run_parameter_search(
                data_path=str(sample_indicator_csv),
                strategy_template_path=str(simple_strategy_yaml),
                param_config_path="/nonexistent/params.yaml",
            )

    def test_search_data_not_found(self, simple_strategy_yaml, param_config_yaml):
        """Test error when data file not found."""
        with pytest.raises(FileNotFoundError):
            run_parameter_search(
                data_path="/nonexistent/data.csv",
                strategy_template_path=str(simple_strategy_yaml),
                param_config_path=str(param_config_yaml),
            )

    def test_search_invalid_capital(
        self, sample_indicator_csv, simple_strategy_yaml, param_config_yaml
    ):
        """Test error with invalid capital."""
        with pytest.raises(TuningError):
            run_parameter_search(
                data_path=str(sample_indicator_csv),
                strategy_template_path=str(simple_strategy_yaml),
                param_config_path=str(param_config_yaml),
                initial_capital=0,
            )

    def test_search_invalid_top_n(
        self, sample_indicator_csv, simple_strategy_yaml, param_config_yaml
    ):
        """Test error with invalid top_n."""
        with pytest.raises(TuningError):
            run_parameter_search(
                data_path=str(sample_indicator_csv),
                strategy_template_path=str(simple_strategy_yaml),
                param_config_path=str(param_config_yaml),
                top_n=0,
            )

    def test_search_parallel_execution(
        self, sample_indicator_csv, simple_strategy_yaml, param_config_yaml
    ):
        """Test parallel parameter search with n_jobs > 1."""
        result = run_parameter_search(
            data_path=str(sample_indicator_csv),
            strategy_template_path=str(simple_strategy_yaml),
            param_config_path=str(param_config_yaml),
            initial_capital=100000.0,
            top_n=3,
            n_jobs=2,
        )

        assert isinstance(result, SearchResult)
        assert result.total_combinations == 9  # 3 * 3
        assert len(result.top_results) <= 3
        assert result.best_final_value > 0

    def test_search_parallel_rejects_dataframe(
        self, sample_indicator_csv, simple_strategy_yaml, param_config_yaml
    ):
        """Test that parallel execution rejects DataFrame input."""
        import pandas as pd

        df = pd.read_csv(sample_indicator_csv)

        with pytest.raises(TuningError) as exc_info:
            run_parameter_search(
                data_path=df,
                strategy_template_path=str(simple_strategy_yaml),
                param_config_path=str(param_config_yaml),
                n_jobs=2,
            )
        assert "DataFrame" in str(exc_info.value)

    def test_search_with_dataframe(
        self, sample_indicator_csv, simple_strategy_yaml, param_config_yaml
    ):
        """Test parameter search with DataFrame input (sequential only)."""
        import pandas as pd

        df = pd.read_csv(sample_indicator_csv)

        result = run_parameter_search(
            data_path=df,
            strategy_template_path=str(simple_strategy_yaml),
            param_config_path=str(param_config_yaml),
            initial_capital=100000.0,
            top_n=3,
            n_jobs=1,  # Must be sequential for DataFrame
        )

        assert isinstance(result, SearchResult)
        assert result.total_combinations == 9
        assert result.best_final_value > 0


class TestCreateParser:
    """Tests for CLI argument parser."""

    def test_parser_required_args(self):
        """Test parser requires data, strategy, params."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])

        with pytest.raises(SystemExit):
            parser.parse_args(["--data", "test.csv"])

        with pytest.raises(SystemExit):
            parser.parse_args(["--data", "test.csv", "--strategy", "s.yaml"])

    def test_parser_all_args(self):
        """Test parser accepts all arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "--data", "data.csv",
            "--strategy", "strategy.yaml",
            "--params", "params.yaml",
            "--capital", "50000",
            "--top-n", "10",
            "--verbose",
            "--output", "results.json",
        ])

        assert args.data == "data.csv"
        assert args.strategy == "strategy.yaml"
        assert args.params == "params.yaml"
        assert args.capital == 50000
        assert args.top_n == 10
        assert args.verbose is True
        assert args.output == "results.json"

    def test_parser_short_args(self):
        """Test parser accepts short argument forms."""
        parser = create_parser()
        args = parser.parse_args([
            "-d", "data.csv",
            "-s", "strategy.yaml",
            "-p", "params.yaml",
            "-c", "25000",
            "-n", "3",
            "-v",
            "-o", "out.json",
        ])

        assert args.data == "data.csv"
        assert args.capital == 25000
        assert args.top_n == 3
        assert args.verbose is True

    def test_parser_defaults(self):
        """Test parser default values."""
        parser = create_parser()
        args = parser.parse_args([
            "-d", "data.csv",
            "-s", "strategy.yaml",
            "-p", "params.yaml",
        ])

        assert args.capital == 100000.0
        assert args.top_n == 5
        assert args.verbose is False
        assert args.output is None


class TestMain:
    """Tests for main CLI function."""

    def test_main_success(
        self, sample_indicator_csv, simple_strategy_yaml, param_config_yaml, capsys
    ):
        """Test successful CLI run."""
        exit_code = main([
            "--data", str(sample_indicator_csv),
            "--strategy", str(simple_strategy_yaml),
            "--params", str(param_config_yaml),
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        assert "PARAMETER SEARCH RESULTS" in captured.out
        assert "TOP PERFORMING" in captured.out

    def test_main_with_output_file(
        self, sample_indicator_csv, simple_strategy_yaml, param_config_yaml, temp_dir, capsys
    ):
        """Test CLI saves output to JSON file."""
        output_file = temp_dir / "results.json"

        exit_code = main([
            "--data", str(sample_indicator_csv),
            "--strategy", str(simple_strategy_yaml),
            "--params", str(param_config_yaml),
            "--output", str(output_file),
        ])

        assert exit_code == 0
        assert output_file.exists()

        # Verify JSON content
        with open(output_file) as f:
            data = json.load(f)
        assert "top_results" in data
        assert "best_parameters" in data

    def test_main_file_not_found(self, simple_strategy_yaml, param_config_yaml, capsys):
        """Test CLI with missing data file."""
        exit_code = main([
            "--data", "nonexistent.csv",
            "--strategy", str(simple_strategy_yaml),
            "--params", str(param_config_yaml),
        ])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_verbose(
        self, sample_indicator_csv, simple_strategy_yaml, param_config_yaml, capsys
    ):
        """Test CLI with verbose flag."""
        exit_code = main([
            "--data", str(sample_indicator_csv),
            "--strategy", str(simple_strategy_yaml),
            "--params", str(param_config_yaml),
            "--verbose",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Testing" in captured.out


class TestIntegration:
    """Integration tests for parameter optimization."""

    def test_full_optimization_workflow(self, temp_dir):
        """Test complete optimization workflow."""
        # Create indicator data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = [100 + i * 0.3 for i in range(100)]

        df = pd.DataFrame({
            "Date": dates,
            "Open": prices,
            "High": [p + 1 for p in prices],
            "Low": [p - 1 for p in prices],
            "Close": prices,
            "Volume": [1000000] * 100,
        })

        # Add RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df["RSI_14"] = 100 - (100 / (1 + rs))
        df = df.fillna(50)  # Fill NaN with neutral RSI

        data_file = temp_dir / "data.csv"
        df.to_csv(data_file, index=False)

        # Create strategy
        strategy_yaml = """
strategies:
  - name: "RSI_Tunable"
    weight: 1.0
    combine: "any"
    rules:
      - indicator: "rsi_14"
        operator: "<"
        value: 30
        action: "buy"
        strength: 1.0
      - indicator: "rsi_14"
        operator: ">"
        value: 70
        action: "sell"
        strength: 1.0
"""
        strategy_file = temp_dir / "strategy.yaml"
        strategy_file.write_text(strategy_yaml)

        # Create param config
        param_config = """
parameters:
  - name: buy_threshold
    path: strategies[0].rules[0].value
    start: 20
    end: 40
    step: 10
"""
        param_file = temp_dir / "params.yaml"
        param_file.write_text(param_config)

        # Run optimization
        result = run_parameter_search(
            data_path=str(data_file),
            strategy_template_path=str(strategy_file),
            param_config_path=str(param_file),
            initial_capital=100000.0,
            top_n=3,
        )

        # Verify results
        assert result.total_combinations == 3
        assert result.evaluated_combinations > 0
        assert len(result.top_results) > 0
        assert result.best_final_value > 0

    def test_different_parameter_types(self, sample_indicator_csv, temp_dir):
        """Test optimization with different parameter value types."""
        # Create strategy with float thresholds
        strategy_yaml = """
strategies:
  - name: "Mixed_Strategy"
    weight: 1.0
    combine: "any"
    rules:
      - indicator: "rsi_14"
        operator: "<"
        value: 30.5
        action: "buy"
        strength: 1.0
"""
        strategy_file = temp_dir / "strategy.yaml"
        strategy_file.write_text(strategy_yaml)

        # Create param config with float values
        param_config = """
parameters:
  - name: threshold
    path: strategies[0].rules[0].value
    start: 25.5
    end: 35.5
    step: 2.5
"""
        param_file = temp_dir / "params.yaml"
        param_file.write_text(param_config)

        result = run_parameter_search(
            data_path=str(sample_indicator_csv),
            strategy_template_path=str(strategy_file),
            param_config_path=str(param_file),
        )

        # Values should include floats
        assert len(result.search_space["threshold"]) > 1
