"""
Tests for the CLI interface.
"""

import os
import sys
import tempfile
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from backtest.daily_scorer.__main__ import create_parser, main
from backtest.daily_scorer.exceptions import ConfigurationError
from backtest.daily_scorer.scorer import ScoreResult
from datetime import datetime


class TestCreateParser:
    """Tests for argument parser creation."""

    def test_parser_requires_symbol(self):
        """Should require --symbol argument."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--strategy", "test.yaml"])

    def test_parser_requires_strategy(self):
        """Should require --strategy argument."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--symbol", "SPY"])

    def test_parser_accepts_multiple_strategies(self):
        """Should accept multiple strategy files."""
        parser = create_parser()
        args = parser.parse_args([
            "--symbol", "SPY",
            "--strategy", "strat1.yaml", "strat2.yaml"
        ])
        assert args.strategy == ["strat1.yaml", "strat2.yaml"]

    def test_parser_default_values(self):
        """Should have correct default values."""
        parser = create_parser()
        args = parser.parse_args([
            "--symbol", "SPY",
            "--strategy", "test.yaml"
        ])

        assert args.cache_dir == "./cache"
        assert args.normalization == "zscore"
        assert args.lookback == 365
        assert args.refresh is False
        assert args.clear_cache is False
        assert args.show_all is False
        assert args.quiet is False

    def test_parser_accepts_all_options(self):
        """Should accept all command-line options."""
        parser = create_parser()
        args = parser.parse_args([
            "--symbol", "AAPL",
            "--strategy", "test.yaml",
            "--cache-dir", "/tmp/cache",
            "--normalization", "minmax",
            "--lookback", "180",
            "--refresh",
            "--clear-cache",
            "--show-all",
            "--quiet",
        ])

        assert args.symbol == "AAPL"
        assert args.cache_dir == "/tmp/cache"
        assert args.normalization == "minmax"
        assert args.lookback == 180
        assert args.refresh is True
        assert args.clear_cache is True
        assert args.show_all is True
        assert args.quiet is True

    def test_parser_short_options(self):
        """Should accept short options."""
        parser = create_parser()
        args = parser.parse_args([
            "-s", "SPY",
            "--strategy", "test.yaml",
            "-q",
        ])

        assert args.symbol == "SPY"
        assert args.quiet is True

    def test_parser_normalization_choices(self):
        """Should only accept valid normalization choices."""
        parser = create_parser()

        # Valid choices
        for choice in ["none", "minmax", "zscore"]:
            args = parser.parse_args([
                "--symbol", "SPY",
                "--strategy", "test.yaml",
                "--normalization", choice,
            ])
            assert args.normalization == choice

        # Invalid choice
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--symbol", "SPY",
                "--strategy", "test.yaml",
                "--normalization", "invalid",
            ])


class TestMainFunction:
    """Tests for main CLI entry point."""

    @pytest.fixture
    def mock_strategy_file(self):
        """Create a temporary strategy file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            content = """strategies:
  - name: TestStrategy
    weight: 1.0
    combine: any
    rules:
      - indicator: rsi_14
        operator: "<"
        value: 30
        action: buy
        strength: 1.0
"""
            f.write(content)
            f.flush()
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_main_no_scores_without_refresh(
        self, mock_strategy_file, temp_cache_dir, capsys
    ):
        """Should report no scores without refresh."""
        result = main([
            "--symbol", "SPY",
            "--strategy", mock_strategy_file,
            "--cache-dir", temp_cache_dir,
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "No scores available" in captured.out

    def test_main_quiet_no_scores(
        self, mock_strategy_file, temp_cache_dir, capsys
    ):
        """Should output N/A in quiet mode with no scores."""
        result = main([
            "--symbol", "SPY",
            "--strategy", mock_strategy_file,
            "--cache-dir", temp_cache_dir,
            "--quiet",
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "N/A" in captured.out

    def test_main_clear_cache(
        self, mock_strategy_file, temp_cache_dir, capsys
    ):
        """Should clear cache when requested."""
        result = main([
            "--symbol", "SPY",
            "--strategy", mock_strategy_file,
            "--cache-dir", temp_cache_dir,
            "--clear-cache",
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "Cleared cache" in captured.out

    def test_main_show_all_empty(
        self, mock_strategy_file, temp_cache_dir, capsys
    ):
        """Should show empty message for no cached scores."""
        result = main([
            "--symbol", "SPY",
            "--strategy", mock_strategy_file,
            "--cache-dir", temp_cache_dir,
            "--show-all",
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "No cached scores" in captured.out

    def test_main_show_all_with_scores(
        self, mock_strategy_file, temp_cache_dir, capsys
    ):
        """Should show all cached scores."""
        from backtest.daily_scorer.scorer import DailyScorer

        # Create scorer and add some scores
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )
        scorer.cache.add_scores(
            [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            [0.5, 0.6],
            [0.75, 0.8],
        )

        result = main([
            "--symbol", "SPY",
            "--strategy", mock_strategy_file,
            "--cache-dir", temp_cache_dir,
            "--show-all",
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "2024-01-01" in captured.out
        assert "2024-01-02" in captured.out

    def test_main_refresh_success(
        self, mock_strategy_file, temp_cache_dir, capsys
    ):
        """Should refresh and display results."""
        mock_result = ScoreResult(
            date=datetime(2024, 1, 15),
            signal_raw=0.5,
            signal_0_1=0.75,
            symbol="SPY",
            is_cached=False,
        )

        with patch("backtest.daily_scorer.__main__.DailyScorer") as MockScorer:
            mock_scorer_instance = MagicMock()
            mock_scorer_instance.refresh.return_value = mock_result
            MockScorer.return_value = mock_scorer_instance

            result = main([
                "--symbol", "SPY",
                "--strategy", mock_strategy_file,
                "--cache-dir", temp_cache_dir,
                "--refresh",
            ])

        assert result == 0
        captured = capsys.readouterr()
        assert "SPY" in captured.out
        assert "2024-01-15" in captured.out

    def test_main_refresh_quiet(
        self, mock_strategy_file, temp_cache_dir, capsys
    ):
        """Should output only score in quiet mode."""
        mock_result = ScoreResult(
            date=datetime(2024, 1, 15),
            signal_raw=0.5,
            signal_0_1=0.7500,
            symbol="SPY",
            is_cached=False,
        )

        with patch("backtest.daily_scorer.__main__.DailyScorer") as MockScorer:
            mock_scorer_instance = MagicMock()
            mock_scorer_instance.refresh.return_value = mock_result
            MockScorer.return_value = mock_scorer_instance

            result = main([
                "--symbol", "SPY",
                "--strategy", mock_strategy_file,
                "--cache-dir", temp_cache_dir,
                "--refresh",
                "--quiet",
            ])

        assert result == 0
        captured = capsys.readouterr()
        assert "0.7500" in captured.out

    def test_main_interpretation_bearish(
        self, mock_strategy_file, temp_cache_dir, capsys
    ):
        """Should show BEARISH interpretation for low scores."""
        mock_result = ScoreResult(
            date=datetime(2024, 1, 15),
            signal_raw=-0.8,
            signal_0_1=0.1,
            symbol="SPY",
            is_cached=False,
        )

        with patch("backtest.daily_scorer.__main__.DailyScorer") as MockScorer:
            mock_scorer_instance = MagicMock()
            mock_scorer_instance.refresh.return_value = mock_result
            mock_scorer_instance.cache.__len__ = MagicMock(return_value=10)
            MockScorer.return_value = mock_scorer_instance

            result = main([
                "--symbol", "SPY",
                "--strategy", mock_strategy_file,
                "--cache-dir", temp_cache_dir,
                "--refresh",
            ])

        captured = capsys.readouterr()
        assert "BEARISH" in captured.out

    def test_main_interpretation_bullish(
        self, mock_strategy_file, temp_cache_dir, capsys
    ):
        """Should show BULLISH interpretation for high scores."""
        mock_result = ScoreResult(
            date=datetime(2024, 1, 15),
            signal_raw=0.8,
            signal_0_1=0.9,
            symbol="SPY",
            is_cached=False,
        )

        with patch("backtest.daily_scorer.__main__.DailyScorer") as MockScorer:
            mock_scorer_instance = MagicMock()
            mock_scorer_instance.refresh.return_value = mock_result
            mock_scorer_instance.cache.__len__ = MagicMock(return_value=10)
            MockScorer.return_value = mock_scorer_instance

            result = main([
                "--symbol", "SPY",
                "--strategy", mock_strategy_file,
                "--cache-dir", temp_cache_dir,
                "--refresh",
            ])

        captured = capsys.readouterr()
        assert "BULLISH" in captured.out

    def test_main_interpretation_neutral(
        self, mock_strategy_file, temp_cache_dir, capsys
    ):
        """Should show NEUTRAL interpretation for mid scores."""
        mock_result = ScoreResult(
            date=datetime(2024, 1, 15),
            signal_raw=0.0,
            signal_0_1=0.5,
            symbol="SPY",
            is_cached=False,
        )

        with patch("backtest.daily_scorer.__main__.DailyScorer") as MockScorer:
            mock_scorer_instance = MagicMock()
            mock_scorer_instance.refresh.return_value = mock_result
            mock_scorer_instance.cache.__len__ = MagicMock(return_value=10)
            MockScorer.return_value = mock_scorer_instance

            result = main([
                "--symbol", "SPY",
                "--strategy", mock_strategy_file,
                "--cache-dir", temp_cache_dir,
                "--refresh",
            ])

        captured = capsys.readouterr()
        assert "NEUTRAL" in captured.out

    def test_main_configuration_error(
        self, temp_cache_dir, capsys
    ):
        """Should return error code for configuration errors."""
        result = main([
            "--symbol", "SPY",
            "--strategy", "nonexistent.yaml",
            "--cache-dir", temp_cache_dir,
        ])

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_unexpected_error(
        self, mock_strategy_file, temp_cache_dir, capsys
    ):
        """Should handle unexpected errors gracefully."""
        with patch("backtest.daily_scorer.__main__.DailyScorer") as MockScorer:
            MockScorer.side_effect = RuntimeError("Unexpected!")

            result = main([
                "--symbol", "SPY",
                "--strategy", mock_strategy_file,
                "--cache-dir", temp_cache_dir,
            ])

        assert result == 1
        captured = capsys.readouterr()
        assert "Unexpected error" in captured.err
