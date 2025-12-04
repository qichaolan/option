"""
Integration tests for the DailyScorer module.

These tests verify end-to-end behavior across various scenarios:
- Happy path workflows
- Market data edge cases
- Cache behavior
- Multi-symbol handling
- Performance characteristics
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backtest.daily_scorer import DailyScorer, ScoreResult
from backtest.daily_scorer.cache import ScoreCache
from backtest.daily_scorer.exceptions import (
    CacheError,
    ConfigurationError,
    InsufficientDataError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def strategy_file():
    """Create a temporary strategy file for testing."""
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
      - indicator: rsi_14
        operator: ">"
        value: 70
        action: sell
        strength: 1.0
"""
        f.write(content)
        f.flush()
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def create_mock_stock_data(
    n_days: int = 300,
    start_date: str = "2023-01-01",
    nan_close_indices: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create mock stock data for testing.

    Args:
        n_days: Number of trading days to generate.
        start_date: Start date for the data.
        nan_close_indices: List of indices where Close should be NaN.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with OHLCV data.
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="B")  # Business days

    close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    high = close + np.abs(np.random.randn(n_days)) * 2
    low = close - np.abs(np.random.randn(n_days)) * 2
    open_price = close + np.random.randn(n_days) * 0.5

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.random.randint(1000000, 5000000, n_days),
    })

    # Set specified indices to NaN
    if nan_close_indices:
        for idx in nan_close_indices:
            if 0 <= idx < len(df):
                df.loc[idx, "Close"] = np.nan

    return df


# =============================================================================
# 1. Happy Path – Everything Works Normally
# =============================================================================


class TestHappyPath:
    """Test: Get daily score when market is closed (normal day)."""

    def test_get_latest_score_normal_day(self, strategy_file, cache_dir):
        """
        Mock data for the last year where the latest day has a valid Close price.
        No cache file exists yet.
        Call get_latest_score after refresh.

        Expect:
        - It downloads data
        - Builds indicators
        - Scores all valid days
        - Creates a cache file
        - Returns the latest date and a valid score 0–1.
        """
        # Create mock data with 300 days, all with valid Close
        mock_data = create_mock_stock_data(n_days=300)

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        # Verify no cache exists initially
        cache_path = Path(cache_dir) / "SPY_scores.csv"
        assert not cache_path.exists()

        # Mock the download and call refresh
        with patch.object(scorer, "_download_data", return_value=mock_data):
            result = scorer.refresh()

        # Verify result
        assert result is not None
        assert isinstance(result, ScoreResult)
        assert result.symbol == "SPY"
        assert 0.0 <= result.signal_0_1 <= 1.0
        assert -1.0 <= result.signal_raw <= 1.0

        # Verify cache was created
        assert cache_path.exists()

        # Verify cache contains scores for multiple days
        assert len(scorer.cache) > 0

        # Verify the returned date is the latest in the data
        expected_last_date = mock_data["Date"].iloc[-1].to_pydatetime()
        expected_last_date = datetime(
            expected_last_date.year,
            expected_last_date.month,
            expected_last_date.day
        )
        assert result.date == expected_last_date


# =============================================================================
# 2. Market Not Closed Yet (Missing Close)
# =============================================================================


class TestMarketNotClosed:
    """Test: Skip today when Close is missing (market still open)."""

    def test_skip_nan_close_latest_day(self, strategy_file, cache_dir):
        """
        Mock data so the latest row has Close = NaN, but previous days are valid.
        No cache file exists.

        Expect:
        - The last day (with NaN Close) is ignored.
        - The second-to-last day is used as the latest scored date.
        - The returned date is that second-to-last day, with its score.
        - Cache only contains rows with valid Close.
        """
        # Create mock data with NaN Close on the last day
        mock_data = create_mock_stock_data(n_days=300)
        mock_data.loc[mock_data.index[-1], "Close"] = np.nan

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        with patch.object(scorer, "_download_data", return_value=mock_data):
            result = scorer.refresh()

        # Verify result
        assert result is not None

        # The returned date should be the second-to-last day
        second_to_last = mock_data["Date"].iloc[-2].to_pydatetime()
        second_to_last = datetime(
            second_to_last.year,
            second_to_last.month,
            second_to_last.day
        )
        assert result.date == second_to_last

        # Cache should not contain the NaN day
        last_date = mock_data["Date"].iloc[-1].to_pydatetime()
        last_date = datetime(last_date.year, last_date.month, last_date.day)
        assert last_date not in scorer.cache


# =============================================================================
# 3. First Call (Cold Start)
# =============================================================================


class TestColdStart:
    """Test: First time call, no cache present."""

    def test_first_time_call_no_cache(self, strategy_file, cache_dir):
        """
        Ensure the cache file does not exist.
        Mock normal daily data for last year, all with valid Close.

        Expect:
        - It fetches data, builds indicators, scores everything.
        - It saves a new cache file with all scored dates.
        - It returns the most recent date and its score.
        """
        mock_data = create_mock_stock_data(n_days=300)

        # Verify cache doesn't exist
        cache_path = Path(cache_dir) / "SPY_scores.csv"
        assert not cache_path.exists()

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        with patch.object(scorer, "_download_data", return_value=mock_data):
            result = scorer.refresh()

        # Verify cache was created and populated
        assert cache_path.exists()
        assert len(scorer.cache) > 0

        # Verify result
        assert result is not None
        assert isinstance(result, ScoreResult)

        # Verify it returns the most recent date
        expected_last_date = mock_data["Date"].iloc[-1].to_pydatetime()
        expected_last_date = datetime(
            expected_last_date.year,
            expected_last_date.month,
            expected_last_date.day
        )
        assert result.date == expected_last_date


# =============================================================================
# 4. Second Call (Cache Reuse, No New Data)
# =============================================================================


class TestCacheReuse:
    """Test: Second call, nothing new to score."""

    def test_second_call_no_new_data(self, strategy_file, cache_dir):
        """
        Run the first call and keep the cache.
        On the second call, mock the same market data (no new days).

        Expect:
        - It loads the cache.
        - It finds that all dates are already scored.
        - It does NOT recompute scores.
        - It returns the same latest scored date and score.
        - Cache file contents are unchanged.
        """
        mock_data = create_mock_stock_data(n_days=300)

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        # First call
        with patch.object(scorer, "_download_data", return_value=mock_data):
            first_result = scorer.refresh()

        first_cache_size = len(scorer.cache)
        first_scores = scorer.get_all_scores().copy()

        # Second call with same data
        with patch.object(scorer, "_download_data", return_value=mock_data):
            second_result = scorer.refresh()

        # Verify same results
        assert second_result.date == first_result.date
        assert second_result.signal_raw == first_result.signal_raw
        assert second_result.signal_0_1 == first_result.signal_0_1

        # Verify cache size unchanged
        assert len(scorer.cache) == first_cache_size

        # Verify cache contents unchanged
        second_scores = scorer.get_all_scores()
        pd.testing.assert_frame_equal(
            first_scores.reset_index(drop=True),
            second_scores.reset_index(drop=True)
        )


# =============================================================================
# 5. Third Call (New Trading Days Added)
# =============================================================================


class TestNewTradingDays:
    """Test: Third call, new days have appeared since last run."""

    def test_new_days_appended_to_cache(self, strategy_file, cache_dir):
        """
        After second call, extend the mocked market data with 2–3 new closed days.

        Expect:
        - It loads the existing cache.
        - It identifies only the new dates as unscored.
        - It runs indicators + score only on those new dates.
        - It appends new scores to the cache.
        - It returns the newest date and its score.
        - Old scores in the cache remain unchanged.
        """
        # Initial data
        initial_data = create_mock_stock_data(n_days=300)

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        # First call with initial data
        with patch.object(scorer, "_download_data", return_value=initial_data):
            first_result = scorer.refresh()

        first_cache_size = len(scorer.cache)

        # Get scores for specific dates to verify they don't change
        sample_date = initial_data["Date"].iloc[150].to_pydatetime()
        sample_date = datetime(sample_date.year, sample_date.month, sample_date.day)
        original_score = scorer.cache.get_score(sample_date)

        # Extended data with 3 new days
        extended_data = create_mock_stock_data(n_days=303)

        # Second call with extended data
        with patch.object(scorer, "_download_data", return_value=extended_data):
            second_result = scorer.refresh()

        # Verify new days were added
        assert len(scorer.cache) == first_cache_size + 3

        # Verify the returned date is the newest
        expected_newest = extended_data["Date"].iloc[-1].to_pydatetime()
        expected_newest = datetime(
            expected_newest.year,
            expected_newest.month,
            expected_newest.day
        )
        assert second_result.date == expected_newest

        # Verify old scores remain unchanged
        current_score = scorer.cache.get_score(sample_date)
        assert current_score["signal_raw"] == original_score["signal_raw"]
        assert current_score["signal_0_1"] == original_score["signal_0_1"]


# =============================================================================
# 6. App Hasn't Run for a Few Days (Gap in Scores)
# =============================================================================


class TestGapInScores:
    """Test: No score entries for last 3 days, but market closed."""

    def test_fill_gap_in_scores(self, strategy_file, cache_dir):
        """
        Cache exists, but last scored date is, say, D-3.
        Mock market data that includes days D-2, D-1, D (all with valid Close).

        Expect:
        - It loads cache and sees last scored date = D-3.
        - It scores D-2, D-1, D (only those).
        - It appends them to cache.
        - It returns D as last scored date with its score.
        """
        # Create initial data for 300 days
        initial_data = create_mock_stock_data(n_days=297)

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        # First call - scores up to day 297
        with patch.object(scorer, "_download_data", return_value=initial_data):
            scorer.refresh()

        initial_cache_size = len(scorer.cache)
        last_scored_before = scorer.cache.get_latest_score()["date"]

        # Simulate 3 new days of data (days 298, 299, 300)
        extended_data = create_mock_stock_data(n_days=300)

        # Second call - should score only the 3 new days
        with patch.object(scorer, "_download_data", return_value=extended_data):
            result = scorer.refresh()

        # Verify 3 new scores were added
        assert len(scorer.cache) == initial_cache_size + 3

        # Verify the returned date is the newest (day 300)
        expected_newest = extended_data["Date"].iloc[-1].to_pydatetime()
        expected_newest = datetime(
            expected_newest.year,
            expected_newest.month,
            expected_newest.day
        )
        assert result.date == expected_newest

        # Verify last scored is now newer than before
        assert result.date > last_scored_before


# =============================================================================
# 7. Data Quality Cases
# =============================================================================


class TestDataQuality:
    """Tests for various data quality scenarios."""

    def test_all_nan_close_raises_error(self, strategy_file, cache_dir):
        """
        Mock data where every row has Close = NaN.

        Expect:
        - The app detects that there is no valid day to score.
        - It returns None or raises a clear error.
        """
        # Create data with all NaN Close
        mock_data = create_mock_stock_data(n_days=300)
        mock_data["Close"] = np.nan

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        with patch.object(scorer, "_download_data", return_value=mock_data):
            result = scorer.refresh()

        # Should return None when no valid data
        assert result is None

    def test_partial_nan_close_skipped(self, strategy_file, cache_dir):
        """
        Mock a year of data where some middle days have NaN Close.

        Expect:
        - Only days with valid Close are scored.
        - NaN rows are skipped completely.
        - Latest scored date is still the last valid Close day.
        """
        # Create data with some middle days having NaN Close
        nan_indices = [50, 51, 52, 100, 150, 200]
        mock_data = create_mock_stock_data(n_days=300, nan_close_indices=nan_indices)

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        with patch.object(scorer, "_download_data", return_value=mock_data):
            result = scorer.refresh()

        # Verify result exists
        assert result is not None

        # Verify the latest date is the last valid Close day (day 300, index 299)
        expected_last = mock_data["Date"].iloc[-1].to_pydatetime()
        expected_last = datetime(
            expected_last.year,
            expected_last.month,
            expected_last.day
        )
        assert result.date == expected_last

        # Verify NaN dates are not in cache
        for idx in nan_indices:
            nan_date = mock_data["Date"].iloc[idx].to_pydatetime()
            nan_date = datetime(nan_date.year, nan_date.month, nan_date.day)
            assert nan_date not in scorer.cache


# =============================================================================
# 8. Cache File Edge Cases
# =============================================================================


class TestCacheEdgeCases:
    """Tests for cache file edge cases."""

    def test_empty_cache_file(self, strategy_file, cache_dir):
        """
        Create an empty cache file (e.g., zero rows).

        Expect:
        - App behaves like first-time run.
        - It repopulates the cache with all valid dates.
        """
        # Create an empty cache file with just headers
        cache_path = Path(cache_dir) / "SPY_scores.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write("date,signal_raw,signal_0_1,scored_at\n")

        mock_data = create_mock_stock_data(n_days=300)

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        # Verify cache loads as empty
        assert len(scorer.cache) == 0

        with patch.object(scorer, "_download_data", return_value=mock_data):
            result = scorer.refresh()

        # Verify cache is now populated
        assert len(scorer.cache) > 0
        assert result is not None

    def test_corrupted_cache_file_raises_error(self, strategy_file, cache_dir):
        """
        Place a corrupted file at the cache path (e.g., invalid CSV).

        Expect:
        - App fails with a clear error message like "failed to read cache".
        - It does not silently pretend it worked.
        """
        # Create a corrupted cache file
        cache_path = Path(cache_dir) / "SPY_scores.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write("this is not valid CSV content\x00\x01\x02garbage")

        # Should raise CacheError when trying to initialize
        with pytest.raises(CacheError, match="Failed to load cache"):
            DailyScorer(
                symbol="SPY",
                strategy_files=strategy_file,
                cache_dir=cache_dir,
            )


# =============================================================================
# 9. Multiple Symbols Behavior
# =============================================================================


class TestMultipleSymbols:
    """Test: Separate caches per symbol."""

    def test_separate_caches_per_symbol(self, strategy_file, cache_dir):
        """
        Call scorer for "QQQ", cache goes to scores_QQQ.csv.
        Call scorer for "SPY", cache goes to scores_SPY.csv.

        Expect:
        - Each symbol has its own independent cache.
        - Their scores and latest dates are not mixed up.
        """
        mock_data_qqq = create_mock_stock_data(n_days=300, seed=42)
        mock_data_spy = create_mock_stock_data(n_days=300, seed=123)

        # Create scorers for different symbols
        scorer_qqq = DailyScorer(
            symbol="QQQ",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        scorer_spy = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        # Score QQQ
        with patch.object(scorer_qqq, "_download_data", return_value=mock_data_qqq):
            result_qqq = scorer_qqq.refresh()

        # Score SPY
        with patch.object(scorer_spy, "_download_data", return_value=mock_data_spy):
            result_spy = scorer_spy.refresh()

        # Verify separate cache files
        qqq_cache = Path(cache_dir) / "QQQ_scores.csv"
        spy_cache = Path(cache_dir) / "SPY_scores.csv"
        assert qqq_cache.exists()
        assert spy_cache.exists()

        # Verify results are for correct symbols
        assert result_qqq.symbol == "QQQ"
        assert result_spy.symbol == "SPY"

        # Verify caches have independent content
        assert len(scorer_qqq.cache) > 0
        assert len(scorer_spy.cache) > 0

        # Scores should be different (different seeds produce different data)
        # Note: With different random seeds, scores will differ
        assert result_qqq.signal_raw != result_spy.signal_raw or \
               result_qqq.signal_0_1 != result_spy.signal_0_1


# =============================================================================
# 10. Performance / Sanity
# =============================================================================


class TestPerformance:
    """Test: Efficiency with 1 year of data."""

    def test_no_recomputation_on_repeated_calls(self, strategy_file, cache_dir):
        """
        Use realistic 1-year daily data (≈ 250 rows).
        Ensure no unnecessary recomputation on repeated calls.
        """
        import time

        mock_data = create_mock_stock_data(n_days=260)  # ~1 year of trading days

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        # First call - should take time to compute all scores
        with patch.object(scorer, "_download_data", return_value=mock_data):
            start = time.time()
            first_result = scorer.refresh()
            first_duration = time.time() - start

        first_cache_size = len(scorer.cache)

        # Second call - should be faster (no new computation)
        with patch.object(scorer, "_download_data", return_value=mock_data):
            start = time.time()
            second_result = scorer.refresh()
            second_duration = time.time() - start

        # Verify cache didn't grow (no recomputation)
        assert len(scorer.cache) == first_cache_size

        # Verify same results
        assert second_result.date == first_result.date
        assert second_result.signal_raw == first_result.signal_raw

        # Note: We can't strictly assert second_duration < first_duration
        # because the scoring engine runs on all data for indicator calculation
        # but it shouldn't add new entries to cache


# =============================================================================
# 11. Overall "Happy Path" Summary - Full Integration Test
# =============================================================================


class TestFullHappyPathWorkflow:
    """
    Happy Path Full Flow:
    1. First run: No cache → fetch data → indicators → scores → save cache → return latest date + score.
    2. Second run (no new days): Cache found → no new dates → no recompute → return same latest date + score.
    3. Third run (new days appear): Cache found → find new dates → compute scores for new days only →
       append to cache → return newest date + score.
    """

    def test_complete_workflow(self, strategy_file, cache_dir):
        """Full integration test covering the complete happy path workflow."""

        # =====================================================================
        # STEP 1: First run - Cold start
        # =====================================================================
        initial_data = create_mock_stock_data(n_days=300)

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        # Verify no cache exists
        cache_path = Path(cache_dir) / "SPY_scores.csv"
        assert not cache_path.exists()

        with patch.object(scorer, "_download_data", return_value=initial_data):
            first_result = scorer.refresh()

        # Assertions for first run
        assert first_result is not None
        assert cache_path.exists()
        first_cache_size = len(scorer.cache)
        assert first_cache_size > 0

        expected_first_date = initial_data["Date"].iloc[-1].to_pydatetime()
        expected_first_date = datetime(
            expected_first_date.year,
            expected_first_date.month,
            expected_first_date.day
        )
        assert first_result.date == expected_first_date

        # Store original scores for verification
        original_scores = scorer.get_all_scores().copy()

        # =====================================================================
        # STEP 2: Second run - Same data, no new days
        # =====================================================================
        # Recreate scorer to simulate fresh app start
        scorer2 = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        # Cache should be loaded
        assert len(scorer2.cache) == first_cache_size

        with patch.object(scorer2, "_download_data", return_value=initial_data):
            second_result = scorer2.refresh()

        # Assertions for second run
        assert second_result.date == first_result.date
        assert second_result.signal_raw == first_result.signal_raw
        assert second_result.signal_0_1 == first_result.signal_0_1
        assert len(scorer2.cache) == first_cache_size

        # Verify cache contents unchanged
        current_scores = scorer2.get_all_scores()
        pd.testing.assert_frame_equal(
            original_scores.reset_index(drop=True),
            current_scores.reset_index(drop=True)
        )

        # =====================================================================
        # STEP 3: Third run - New days appear
        # =====================================================================
        extended_data = create_mock_stock_data(n_days=305)  # 5 new days

        # Recreate scorer to simulate fresh app start
        scorer3 = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        with patch.object(scorer3, "_download_data", return_value=extended_data):
            third_result = scorer3.refresh()

        # Assertions for third run
        expected_newest = extended_data["Date"].iloc[-1].to_pydatetime()
        expected_newest = datetime(
            expected_newest.year,
            expected_newest.month,
            expected_newest.day
        )
        assert third_result.date == expected_newest
        assert third_result.date > first_result.date

        # Verify cache grew by exactly 5 entries
        assert len(scorer3.cache) == first_cache_size + 5

        # Verify original scores are preserved
        sample_date = initial_data["Date"].iloc[150].to_pydatetime()
        sample_date = datetime(sample_date.year, sample_date.month, sample_date.day)

        original_sample = original_scores[
            original_scores["date"].dt.normalize() == pd.Timestamp(sample_date)
        ].iloc[0]

        current_sample = scorer3.cache.get_score(sample_date)

        assert current_sample["signal_raw"] == original_sample["signal_raw"]
        assert current_sample["signal_0_1"] == original_sample["signal_0_1"]


# =============================================================================
# Additional Edge Cases
# =============================================================================


class TestAdditionalEdgeCases:
    """Additional edge case tests."""

    def test_get_score_without_refresh_returns_none(self, strategy_file, cache_dir):
        """Without refresh, get_latest_score should return None for empty cache."""
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        result = scorer.get_latest_score()
        assert result is None

    def test_get_specific_date_score(self, strategy_file, cache_dir):
        """Test getting score for a specific date."""
        mock_data = create_mock_stock_data(n_days=300)

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        with patch.object(scorer, "_download_data", return_value=mock_data):
            scorer.refresh()

        # Get score for a specific date
        target_date = mock_data["Date"].iloc[150].to_pydatetime()
        result = scorer.get_score(target_date)

        assert result is not None
        assert result.symbol == "SPY"
        assert 0.0 <= result.signal_0_1 <= 1.0

    def test_clear_cache_and_refresh(self, strategy_file, cache_dir):
        """Test clearing cache and refreshing."""
        mock_data = create_mock_stock_data(n_days=300)

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        # First refresh
        with patch.object(scorer, "_download_data", return_value=mock_data):
            scorer.refresh()

        assert len(scorer.cache) > 0

        # Clear cache
        scorer.clear_cache()
        assert len(scorer.cache) == 0

        # Refresh again
        with patch.object(scorer, "_download_data", return_value=mock_data):
            result = scorer.refresh()

        assert result is not None
        assert len(scorer.cache) > 0

    def test_insufficient_data_raises_error(self, strategy_file, cache_dir):
        """Test that insufficient data raises appropriate error."""
        small_data = create_mock_stock_data(n_days=50)  # Too few days

        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        with patch.object(scorer, "_download_data", return_value=small_data):
            with pytest.raises(InsufficientDataError):
                scorer.refresh()

    def test_symbol_case_insensitive(self, strategy_file, cache_dir):
        """Test that symbol is case-insensitive."""
        mock_data = create_mock_stock_data(n_days=300)

        # Create scorer with lowercase symbol
        scorer = DailyScorer(
            symbol="spy",
            strategy_files=strategy_file,
            cache_dir=cache_dir,
        )

        assert scorer.symbol == "SPY"

        with patch.object(scorer, "_download_data", return_value=mock_data):
            result = scorer.refresh()

        assert result.symbol == "SPY"

        # Verify cache file is uppercase
        cache_path = Path(cache_dir) / "SPY_scores.csv"
        assert cache_path.exists()
