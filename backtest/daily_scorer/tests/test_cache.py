"""
Tests for the score caching system.
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from backtest.daily_scorer.cache import ScoreCache
from backtest.daily_scorer.exceptions import CacheError


@pytest.fixture
def temp_cache_path():
    """Create a temporary cache file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_scores.csv")


@pytest.fixture
def cache(temp_cache_path):
    """Create a ScoreCache instance with temp path."""
    return ScoreCache(temp_cache_path)


class TestScoreCacheInit:
    """Tests for ScoreCache initialization."""

    def test_creates_empty_cache(self, temp_cache_path):
        """New cache should be empty."""
        cache = ScoreCache(temp_cache_path)
        assert len(cache) == 0
        assert cache.get_latest_score() is None

    def test_loads_existing_cache(self, temp_cache_path):
        """Should load existing cache file."""
        # Create initial cache
        cache1 = ScoreCache(temp_cache_path)
        dates = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        cache1.add_scores(dates, [0.5, 0.6], [0.75, 0.8])

        # Load same cache
        cache2 = ScoreCache(temp_cache_path)
        assert len(cache2) == 2

    def test_handles_missing_directory(self):
        """Should handle cache in non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "deep", "cache.csv")
            cache = ScoreCache(path)
            cache.add_scores([datetime(2024, 1, 1)], [0.5], [0.75])
            assert Path(path).exists()

    def test_raises_on_invalid_cache_file(self, temp_cache_path):
        """Should raise CacheError for invalid cache file."""
        # Write invalid CSV
        Path(temp_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(temp_cache_path, "w") as f:
            f.write("wrong_column,another_wrong\n1,2\n")

        with pytest.raises(CacheError, match="Failed to load cache"):
            ScoreCache(temp_cache_path)


class TestScoreCacheOperations:
    """Tests for cache operations."""

    def test_add_scores(self, cache):
        """Should add scores to cache."""
        dates = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        signal_raw = [0.5, -0.3]
        signal_0_1 = [0.75, 0.35]

        added = cache.add_scores(dates, signal_raw, signal_0_1)

        assert added == 2
        assert len(cache) == 2

    def test_add_scores_skips_duplicates(self, cache):
        """Should not add scores for already cached dates."""
        date = datetime(2024, 1, 1)

        # Add first time
        cache.add_scores([date], [0.5], [0.75])
        assert len(cache) == 1

        # Try to add again with different values
        added = cache.add_scores([date], [0.8], [0.9])
        assert added == 0
        assert len(cache) == 1

        # Original values should be preserved
        score = cache.get_score(date)
        assert score["signal_raw"] == 0.5

    def test_add_scores_validates_length(self, cache):
        """Should raise ValueError for mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            cache.add_scores([datetime(2024, 1, 1)], [0.5, 0.6], [0.75])

    def test_get_score(self, cache):
        """Should retrieve score for specific date."""
        date = datetime(2024, 1, 15)
        cache.add_scores([date], [0.5], [0.75])

        score = cache.get_score(date)

        assert score is not None
        assert score["signal_raw"] == 0.5
        assert score["signal_0_1"] == 0.75

    def test_get_score_not_found(self, cache):
        """Should return None for uncached date."""
        cache.add_scores([datetime(2024, 1, 1)], [0.5], [0.75])

        score = cache.get_score(datetime(2024, 1, 2))
        assert score is None

    def test_get_score_normalizes_time(self, cache):
        """Should find score regardless of time component."""
        cache.add_scores([datetime(2024, 1, 15)], [0.5], [0.75])

        # Query with different time
        score = cache.get_score(datetime(2024, 1, 15, 14, 30, 0))
        assert score is not None

    def test_get_latest_score(self, cache):
        """Should return most recent score."""
        dates = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 3),
            datetime(2024, 1, 2),
        ]
        cache.add_scores(dates, [0.1, 0.3, 0.2], [0.55, 0.65, 0.6])

        latest = cache.get_latest_score()

        assert latest is not None
        assert latest["date"] == datetime(2024, 1, 3)
        assert latest["signal_raw"] == 0.3

    def test_get_latest_score_empty_cache(self, cache):
        """Should return None for empty cache."""
        assert cache.get_latest_score() is None

    def test_get_scored_dates(self, cache):
        """Should return set of scored dates."""
        dates = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        cache.add_scores(dates, [0.5, 0.6], [0.75, 0.8])

        scored = cache.get_scored_dates()

        assert len(scored) == 2
        assert datetime(2024, 1, 1) in scored
        assert datetime(2024, 1, 2) in scored

    def test_get_scored_dates_empty(self, cache):
        """Should return empty set for empty cache."""
        assert cache.get_scored_dates() == set()

    def test_clear(self, cache, temp_cache_path):
        """Should clear all cached scores."""
        cache.add_scores([datetime(2024, 1, 1)], [0.5], [0.75])
        assert len(cache) == 1

        cache.clear()

        assert len(cache) == 0
        assert not Path(temp_cache_path).exists()

    def test_contains(self, cache):
        """Should support 'in' operator."""
        date = datetime(2024, 1, 1)
        cache.add_scores([date], [0.5], [0.75])

        assert date in cache
        assert datetime(2024, 1, 2) not in cache


class TestScoreCachePersistence:
    """Tests for cache persistence."""

    def test_persists_across_instances(self, temp_cache_path):
        """Scores should persist across cache instances."""
        # First instance
        cache1 = ScoreCache(temp_cache_path)
        cache1.add_scores([datetime(2024, 1, 1)], [0.5], [0.75])

        # Second instance
        cache2 = ScoreCache(temp_cache_path)
        assert len(cache2) == 1

        score = cache2.get_score(datetime(2024, 1, 1))
        assert score["signal_raw"] == 0.5

    def test_cache_file_format(self, temp_cache_path):
        """Cache file should be valid CSV with expected columns."""
        cache = ScoreCache(temp_cache_path)
        cache.add_scores([datetime(2024, 1, 1)], [0.5], [0.75])

        df = pd.read_csv(temp_cache_path)

        assert "date" in df.columns
        assert "signal_raw" in df.columns
        assert "signal_0_1" in df.columns
        assert "scored_at" in df.columns
        assert len(df) == 1
