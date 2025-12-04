"""
Tests for the main DailyScorer class.
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtest.daily_scorer.cache import ScoreCache
from backtest.daily_scorer.exceptions import (
    ConfigurationError,
    InsufficientDataError,
)
from backtest.daily_scorer.scorer import DailyScorer, ScoreResult


@pytest.fixture
def mock_stock_data():
    """Create mock stock data with enough rows for indicators."""
    n_days = 300
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    np.random.seed(42)

    close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    high = close + np.abs(np.random.randn(n_days)) * 2
    low = close - np.abs(np.random.randn(n_days)) * 2
    open_price = close + np.random.randn(n_days) * 0.5

    return pd.DataFrame({
        "Date": dates,
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.random.randint(1000000, 5000000, n_days),
    })


@pytest.fixture
def mock_strategy_file():
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
        f.flush()  # Ensure content is written before yielding
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestDailyScorerInit:
    """Tests for DailyScorer initialization."""

    def test_init_creates_cache(self, mock_strategy_file, temp_cache_dir):
        """Should create cache on initialization."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        assert scorer.symbol == "SPY"
        assert isinstance(scorer.cache, ScoreCache)

    def test_init_uppercases_symbol(self, mock_strategy_file, temp_cache_dir):
        """Should uppercase the symbol."""
        scorer = DailyScorer(
            symbol="spy",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        assert scorer.symbol == "SPY"

    def test_init_accepts_list_of_strategies(self, mock_strategy_file, temp_cache_dir):
        """Should accept list of strategy files."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=[mock_strategy_file],
            cache_dir=temp_cache_dir,
        )

        assert len(scorer._strategy_files) == 1

    def test_init_raises_for_missing_strategy(self, temp_cache_dir):
        """Should raise ConfigurationError for missing strategy file."""
        with pytest.raises(ConfigurationError, match="not found"):
            DailyScorer(
                symbol="SPY",
                strategy_files="nonexistent.yaml",
                cache_dir=temp_cache_dir,
            )

    def test_repr(self, mock_strategy_file, temp_cache_dir):
        """Should have informative repr."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        repr_str = repr(scorer)
        assert "SPY" in repr_str
        assert "zscore" in repr_str


class TestScoreResult:
    """Tests for ScoreResult dataclass."""

    def test_str_representation(self):
        """Should have readable string representation."""
        result = ScoreResult(
            date=datetime(2024, 1, 15),
            signal_raw=0.5,
            signal_0_1=0.75,
            symbol="SPY",
            is_cached=True,
        )

        str_repr = str(result)
        assert "SPY" in str_repr
        assert "2024-01-15" in str_repr
        assert "0.5000" in str_repr
        assert "cached" in str_repr


class TestDailyScorerOperations:
    """Tests for DailyScorer operations."""

    def test_get_latest_score_empty_cache(self, mock_strategy_file, temp_cache_dir):
        """Should return None for empty cache."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        result = scorer.get_latest_score()
        assert result is None

    def test_get_score_by_date(self, mock_strategy_file, temp_cache_dir):
        """Should return score for specific date."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        # Add a score manually
        scorer.cache.add_scores(
            [datetime(2024, 1, 15)],
            [0.5],
            [0.75],
        )

        result = scorer.get_score(datetime(2024, 1, 15))

        assert result is not None
        assert result.signal_raw == 0.5
        assert result.signal_0_1 == 0.75
        assert result.is_cached

    def test_get_score_not_found(self, mock_strategy_file, temp_cache_dir):
        """Should return None for uncached date."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        result = scorer.get_score(datetime(2024, 1, 15))
        assert result is None

    def test_get_all_scores(self, mock_strategy_file, temp_cache_dir):
        """Should return all cached scores as DataFrame."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        # Add scores
        dates = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        scorer.cache.add_scores(dates, [0.5, 0.6], [0.75, 0.8])

        df = scorer.get_all_scores()

        assert len(df) == 2
        assert "date" in df.columns
        assert "signal_raw" in df.columns

    def test_get_all_scores_empty(self, mock_strategy_file, temp_cache_dir):
        """Should return empty DataFrame for empty cache."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        df = scorer.get_all_scores()

        assert len(df) == 0
        assert "date" in df.columns

    def test_clear_cache(self, mock_strategy_file, temp_cache_dir):
        """Should clear all cached scores."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        # Add and then clear
        scorer.cache.add_scores([datetime(2024, 1, 1)], [0.5], [0.75])
        assert len(scorer.cache) == 1

        scorer.clear_cache()

        assert len(scorer.cache) == 0


class TestDailyScorerRefresh:
    """Tests for refresh functionality."""

    def test_refresh_downloads_and_scores(
        self, mock_stock_data, mock_strategy_file, temp_cache_dir
    ):
        """Should download data, compute indicators, and score."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        with patch.object(scorer, "_download_data", return_value=mock_stock_data):
            result = scorer.refresh()

        assert result is not None
        assert isinstance(result, ScoreResult)
        assert len(scorer.cache) > 0

    def test_refresh_skips_already_scored(
        self, mock_stock_data, mock_strategy_file, temp_cache_dir
    ):
        """Should not re-score already cached dates."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        # Pre-populate cache with some dates
        existing_date = mock_stock_data["Date"].iloc[100].to_pydatetime()
        scorer.cache.add_scores([existing_date], [0.5], [0.75])

        with patch.object(scorer, "_download_data", return_value=mock_stock_data):
            result = scorer.refresh()

        # Check the pre-existing score wasn't overwritten
        score = scorer.cache.get_score(existing_date)
        assert score["signal_raw"] == 0.5

    def test_refresh_filters_null_closes(self, mock_strategy_file, temp_cache_dir):
        """Should filter out rows with null Close prices."""
        # Create data with some null closes
        n_days = 300
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        data = pd.DataFrame({
            "Date": dates,
            "Open": [100.0] * n_days,
            "High": [101.0] * n_days,
            "Low": [99.0] * n_days,
            "Close": [100.5] * n_days,
            "Volume": [1000000] * n_days,
        })
        # Set some closes to null
        data.loc[data.index[-5:], "Close"] = None

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        with patch.object(scorer, "_download_data", return_value=data):
            result = scorer.refresh()

        # Should still work, just with fewer dates
        assert result is not None

    def test_refresh_raises_on_insufficient_data(
        self, mock_strategy_file, temp_cache_dir
    ):
        """Should raise InsufficientDataError for too little data."""
        # Create data with only 10 days
        small_data = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10),
            "Open": [100.0] * 10,
            "High": [101.0] * 10,
            "Low": [99.0] * 10,
            "Close": [100.5] * 10,
            "Volume": [1000000] * 10,
        })

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        with patch.object(scorer, "_download_data", return_value=small_data):
            with pytest.raises(InsufficientDataError):
                scorer.refresh()

    def test_refresh_returns_none_for_empty_data(
        self, mock_strategy_file, temp_cache_dir
    ):
        """Should return None if no valid data."""
        empty_data = pd.DataFrame({
            "Date": [],
            "Open": [],
            "High": [],
            "Low": [],
            "Close": [],
            "Volume": [],
        })

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        with patch.object(scorer, "_download_data", return_value=empty_data):
            result = scorer.refresh()

        assert result is None


class TestDailyScorerFilterUnscored:
    """Tests for _filter_unscored_dates method."""

    def test_filter_returns_all_when_cache_empty(
        self, mock_strategy_file, temp_cache_dir
    ):
        """Should return all dates when cache is empty."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=5),
            "Close": [100.0] * 5,
        })

        filtered = scorer._filter_unscored_dates(df)
        assert len(filtered) == 5

    def test_filter_excludes_scored_dates(
        self, mock_strategy_file, temp_cache_dir
    ):
        """Should exclude dates that are already scored."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        # Add some dates to cache
        scorer.cache.add_scores(
            [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            [0.5, 0.6],
            [0.75, 0.8],
        )

        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=5),
            "Close": [100.0] * 5,
        })

        filtered = scorer._filter_unscored_dates(df)
        # Should only have dates 3, 4, 5
        assert len(filtered) == 3


class TestDailyScorerValidClose:
    """Tests for _has_valid_close method."""

    def test_detects_valid_closes(self, mock_strategy_file, temp_cache_dir):
        """Should correctly identify valid Close values."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=mock_strategy_file,
            cache_dir=temp_cache_dir,
        )

        df = pd.DataFrame({
            "Close": [100.0, None, 101.0, np.nan, 102.0],
        })

        valid = scorer._has_valid_close(df)

        assert valid.tolist() == [True, False, True, False, True]
