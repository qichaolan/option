"""
Tests for AI score service layer.
"""

import math
from datetime import datetime, date
from unittest.mock import MagicMock, patch

import pytest

from web.app.services import ai_score_service


class TestValidateScore:
    """Tests for _validate_score function."""

    def test_valid_scores(self):
        """Should accept valid scores."""
        assert ai_score_service._validate_score(0.5) == 0.5
        assert ai_score_service._validate_score(0.0) == 0.0
        assert ai_score_service._validate_score(1.0) == 1.0

    def test_clamps_out_of_range(self):
        """Should clamp scores outside 0-1 range."""
        assert ai_score_service._validate_score(1.5) == 1.0
        assert ai_score_service._validate_score(-0.5) == 0.0

    def test_rejects_nan(self):
        """Should reject NaN values."""
        with pytest.raises(ValueError, match="Invalid score value"):
            ai_score_service._validate_score(float("nan"))

    def test_rejects_infinity(self):
        """Should reject infinite values."""
        with pytest.raises(ValueError, match="Invalid score value"):
            ai_score_service._validate_score(float("inf"))
        with pytest.raises(ValueError, match="Invalid score value"):
            ai_score_service._validate_score(float("-inf"))


class TestBuildScoreResponse:
    """Tests for _build_score_response function."""

    def test_builds_response(self):
        """Should build correct response dict."""
        result = ai_score_service._build_score_response(
            symbol="spy",
            score_date=date(2024, 1, 15),
            signal_raw=0.5,
            signal_0_1=0.75,
        )

        assert result["symbol"] == "SPY"
        assert result["date"] == "2024-01-15"
        assert result["score_raw"] == 0.5
        assert result["score_0_1"] == 0.75
        assert result["ai_rating"] == "Buy"

    def test_clamps_out_of_range_score(self):
        """Should clamp out of range scores."""
        result = ai_score_service._build_score_response(
            symbol="SPY",
            score_date=date(2024, 1, 15),
            signal_raw=0.5,
            signal_0_1=1.5,  # Out of range
        )
        assert result["score_0_1"] == 1.0


class TestGetAIRating:
    """Tests for _get_ai_rating function."""

    def test_strong_buy_above_0_9(self):
        """Should return 'Strong Buy' for scores > 0.9."""
        assert ai_score_service._get_ai_rating(0.95) == "Strong Buy"
        assert ai_score_service._get_ai_rating(0.91) == "Strong Buy"

    def test_buy_between_0_7_and_0_9(self):
        """Should return 'Buy' for scores between 0.7 and 0.9."""
        assert ai_score_service._get_ai_rating(0.9) == "Buy"
        assert ai_score_service._get_ai_rating(0.8) == "Buy"
        assert ai_score_service._get_ai_rating(0.71) == "Buy"

    def test_hold_between_0_3_and_0_7(self):
        """Should return 'Hold' for scores between 0.3 and 0.7."""
        assert ai_score_service._get_ai_rating(0.7) == "Hold"
        assert ai_score_service._get_ai_rating(0.5) == "Hold"
        assert ai_score_service._get_ai_rating(0.3) == "Hold"

    def test_sell_between_0_1_and_0_3(self):
        """Should return 'Sell' for scores between 0.1 and 0.3."""
        assert ai_score_service._get_ai_rating(0.29) == "Sell"
        assert ai_score_service._get_ai_rating(0.2) == "Sell"
        assert ai_score_service._get_ai_rating(0.1) == "Sell"

    def test_must_sell_below_0_1(self):
        """Should return 'Must Sell' for scores < 0.1."""
        assert ai_score_service._get_ai_rating(0.09) == "Must Sell"
        assert ai_score_service._get_ai_rating(0.05) == "Must Sell"
        assert ai_score_service._get_ai_rating(0.0) == "Must Sell"

    def test_boundary_values(self):
        """Should handle boundary values correctly."""
        # 0.9 is Buy, not Strong Buy
        assert ai_score_service._get_ai_rating(0.9) == "Buy"
        # 0.7 is Hold, not Buy
        assert ai_score_service._get_ai_rating(0.7) == "Hold"
        # 0.3 is Hold, not Sell
        assert ai_score_service._get_ai_rating(0.3) == "Hold"
        # 0.1 is Sell, not Must Sell
        assert ai_score_service._get_ai_rating(0.1) == "Sell"


class TestIsMarketClosedToday:
    """Tests for _is_market_closed_today function."""

    @patch("web.app.services.ai_score_service.date")
    def test_returns_true_on_saturday(self, mock_date):
        """Should return True on Saturday."""
        mock_date.today.return_value = date(2024, 1, 6)  # Saturday
        assert ai_score_service._is_market_closed_today() is True

    @patch("web.app.services.ai_score_service.date")
    def test_returns_true_on_sunday(self, mock_date):
        """Should return True on Sunday."""
        mock_date.today.return_value = date(2024, 1, 7)  # Sunday
        assert ai_score_service._is_market_closed_today() is True

    @patch("web.app.services.ai_score_service.date")
    def test_returns_false_on_weekday(self, mock_date):
        """Should return False on weekdays."""
        mock_date.today.return_value = date(2024, 1, 8)  # Monday
        assert ai_score_service._is_market_closed_today() is False


class TestGetAIScore:
    """Tests for get_ai_score function."""

    def test_raises_for_empty_symbol(self):
        """Should raise ValueError for empty symbol."""
        with pytest.raises(ValueError, match="Symbol is required"):
            ai_score_service.get_ai_score("")

        with pytest.raises(ValueError, match="Symbol is required"):
            ai_score_service.get_ai_score("   ")

    @patch("web.app.services.ai_score_service.gcs_cache")
    @patch("web.app.services.ai_score_service.date")
    def test_returns_cached_score_for_today(self, mock_date, mock_cache):
        """Should return cached score if it's from today."""
        today = date(2024, 1, 15)
        mock_date.today.return_value = today

        mock_cache.get_latest_score.return_value = {
            "date": datetime(2024, 1, 15),
            "signal_raw": 0.5,
            "signal_0_1": 0.75,
        }

        result = ai_score_service.get_ai_score("SPY")

        assert result["symbol"] == "SPY"
        assert result["date"] == "2024-01-15"
        assert result["score_raw"] == 0.5
        assert result["score_0_1"] == 0.75
        assert result["ai_rating"] == "Buy"

    @patch("web.app.services.ai_score_service.gcs_cache")
    @patch("web.app.services.ai_score_service._is_market_closed_today")
    @patch("web.app.services.ai_score_service.date")
    def test_returns_cached_score_on_weekend(self, mock_date, mock_market, mock_cache):
        """Should return Friday's score on weekend."""
        today = date(2024, 1, 13)  # Saturday
        mock_date.today.return_value = today
        mock_market.return_value = True

        mock_cache.get_latest_score.return_value = {
            "date": datetime(2024, 1, 12),  # Friday
            "signal_raw": 0.5,
            "signal_0_1": 0.75,
        }

        result = ai_score_service.get_ai_score("SPY")

        assert result["symbol"] == "SPY"
        assert result["date"] == "2024-01-12"
        assert result["ai_rating"] == "Buy"

    @patch("web.app.services.ai_score_service.gcs_cache")
    @patch("web.app.services.ai_score_service.DailyScorer")
    @patch("web.app.services.ai_score_service.date")
    def test_refreshes_when_cache_stale(self, mock_date, MockScorer, mock_cache):
        """Should refresh when cache is stale."""
        today = date(2024, 1, 15)
        mock_date.today.return_value = today

        # Cache has old score
        mock_cache.get_latest_score.return_value = {
            "date": datetime(2024, 1, 10),  # 5 days old
            "signal_raw": 0.5,
            "signal_0_1": 0.75,
        }

        # Mock scorer refresh
        mock_scorer = MagicMock()
        mock_result = MagicMock()
        mock_result.date = datetime(2024, 1, 15)
        mock_result.signal_raw = 0.6
        mock_result.signal_0_1 = 0.8
        mock_scorer.refresh.return_value = mock_result
        MockScorer.return_value = mock_scorer

        mock_cache.add_score.return_value = True

        result = ai_score_service.get_ai_score("SPY")

        assert result["symbol"] == "SPY"
        assert result["score_raw"] == 0.6
        assert result["score_0_1"] == 0.8
        mock_scorer.refresh.assert_called_once()
        mock_cache.add_score.assert_called_once()

    @patch("web.app.services.ai_score_service.gcs_cache")
    @patch("web.app.services.ai_score_service.DailyScorer")
    def test_force_refresh_bypasses_cache(self, MockScorer, mock_cache):
        """Should bypass cache when force_refresh is True."""
        mock_scorer = MagicMock()
        mock_result = MagicMock()
        mock_result.date = datetime(2024, 1, 15)
        mock_result.signal_raw = 0.6
        mock_result.signal_0_1 = 0.8
        mock_scorer.refresh.return_value = mock_result
        MockScorer.return_value = mock_scorer

        mock_cache.add_score.return_value = True

        result = ai_score_service.get_ai_score("SPY", force_refresh=True)

        assert result["symbol"] == "SPY"
        mock_scorer.refresh.assert_called_once()
        # get_latest_score should not be called when force_refresh=True
        mock_cache.get_latest_score.assert_not_called()

    @patch("web.app.services.ai_score_service.gcs_cache")
    @patch("web.app.services.ai_score_service.DailyScorer")
    def test_falls_back_to_cache_on_refresh_none(self, MockScorer, mock_cache):
        """Should fall back to cache if refresh returns None."""
        mock_scorer = MagicMock()
        mock_scorer.refresh.return_value = None
        MockScorer.return_value = mock_scorer

        # When force_refresh=True, get_latest_score is only called as fallback
        mock_cache.get_latest_score.return_value = {
            "date": datetime(2024, 1, 10),
            "signal_raw": 0.5,
            "signal_0_1": 0.75,
        }

        result = ai_score_service.get_ai_score("SPY", force_refresh=True)

        assert result["symbol"] == "SPY"
        assert result["score_raw"] == 0.5

    @patch("web.app.services.ai_score_service.gcs_cache")
    @patch("web.app.services.ai_score_service.DailyScorer")
    def test_raises_when_no_data_available(self, MockScorer, mock_cache):
        """Should raise ValueError when no data available."""
        mock_scorer = MagicMock()
        mock_scorer.refresh.return_value = None
        MockScorer.return_value = mock_scorer

        mock_cache.get_latest_score.return_value = None

        with pytest.raises(ValueError, match="No score data available"):
            ai_score_service.get_ai_score("SPY", force_refresh=True)

    def test_uppercases_symbol(self):
        """Should uppercase the symbol."""
        with patch("web.app.services.ai_score_service.gcs_cache") as mock_cache:
            with patch("web.app.services.ai_score_service.date") as mock_date:
                mock_date.today.return_value = date(2024, 1, 15)
                mock_cache.get_latest_score.return_value = {
                    "date": datetime(2024, 1, 15),
                    "signal_raw": 0.5,
                    "signal_0_1": 0.75,
                }

                result = ai_score_service.get_ai_score("spy")

                assert result["symbol"] == "SPY"


class TestGetAIScoresBatch:
    """Tests for get_ai_scores_batch function."""

    @patch("web.app.services.ai_score_service.get_ai_score")
    def test_returns_scores_for_multiple_symbols(self, mock_get_score):
        """Should return scores for all symbols."""
        mock_get_score.side_effect = [
            {"symbol": "SPY", "date": "2024-01-15", "score_raw": 0.5, "score_0_1": 0.75, "ai_rating": "Buy"},
            {"symbol": "AAPL", "date": "2024-01-15", "score_raw": 0.6, "score_0_1": 0.8, "ai_rating": "Buy"},
        ]

        results = ai_score_service.get_ai_scores_batch(["SPY", "AAPL"])

        assert len(results) == 2
        assert results[0]["symbol"] == "SPY"
        assert results[1]["symbol"] == "AAPL"

    @patch("web.app.services.ai_score_service.get_ai_score")
    def test_handles_errors_gracefully(self, mock_get_score):
        """Should return error dict for failed symbols."""
        mock_get_score.side_effect = [
            {"symbol": "SPY", "date": "2024-01-15", "score_raw": 0.5, "score_0_1": 0.75, "ai_rating": "Buy"},
            ValueError("No data for INVALID"),
        ]

        results = ai_score_service.get_ai_scores_batch(["SPY", "INVALID"])

        assert len(results) == 2
        assert results[0]["symbol"] == "SPY"
        assert "error" in results[1]
        assert results[1]["symbol"] == "INVALID"

    @patch("web.app.services.ai_score_service.get_ai_score")
    def test_passes_force_refresh(self, mock_get_score):
        """Should pass force_refresh to individual calls."""
        mock_get_score.return_value = {
            "symbol": "SPY", "date": "2024-01-15",
            "score_raw": 0.5, "score_0_1": 0.75, "ai_rating": "Buy"
        }

        ai_score_service.get_ai_scores_batch(["SPY"], force_refresh=True)

        mock_get_score.assert_called_once_with("SPY", force_refresh=True)


class TestExtractDate:
    """Tests for _extract_date function."""

    def test_extracts_date_from_datetime(self):
        """Should extract date from datetime object."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = ai_score_service._extract_date(dt)
        assert result == date(2024, 1, 15)

    def test_returns_date_unchanged(self):
        """Should return date object unchanged."""
        d = date(2024, 1, 15)
        result = ai_score_service._extract_date(d)
        assert result == date(2024, 1, 15)


class TestGetAIScoreEdgeCases:
    """Additional edge case tests for get_ai_score."""

    @patch("web.app.services.ai_score_service.gcs_cache")
    @patch("web.app.services.ai_score_service.DailyScorer")
    def test_continues_when_cache_fails(self, MockScorer, mock_cache):
        """Should continue even when caching the new score fails."""
        mock_scorer = MagicMock()
        mock_result = MagicMock()
        mock_result.date = datetime(2024, 1, 15)
        mock_result.signal_raw = 0.6
        mock_result.signal_0_1 = 0.8
        mock_scorer.refresh.return_value = mock_result
        MockScorer.return_value = mock_scorer

        # Make add_score raise an exception
        mock_cache.add_score.side_effect = Exception("GCS write failed")

        result = ai_score_service.get_ai_score("SPY", force_refresh=True)

        # Should still return the result despite cache failure
        assert result["symbol"] == "SPY"
        assert result["score_raw"] == 0.6
        assert result["score_0_1"] == 0.8

    @patch("web.app.services.ai_score_service.gcs_cache")
    @patch("web.app.services.ai_score_service.DailyScorer")
    def test_raises_on_daily_scorer_error(self, MockScorer, mock_cache):
        """Should raise ValueError when DailyScorer raises DailyScorerError."""
        from backtest.daily_scorer.exceptions import DailyScorerError

        MockScorer.side_effect = DailyScorerError("Strategy file not found")

        with pytest.raises(ValueError, match="Error computing score for SPY"):
            ai_score_service.get_ai_score("SPY", force_refresh=True)

    @patch("web.app.services.ai_score_service.gcs_cache")
    @patch("web.app.services.ai_score_service.date")
    def test_returns_cached_with_date_object(self, mock_date, mock_cache):
        """Should handle cached score with date object (not datetime)."""
        today = date(2024, 1, 15)
        mock_date.today.return_value = today

        # Return a date object instead of datetime
        mock_cache.get_latest_score.return_value = {
            "date": date(2024, 1, 15),  # date, not datetime
            "signal_raw": 0.5,
            "signal_0_1": 0.75,
        }

        result = ai_score_service.get_ai_score("SPY")

        assert result["symbol"] == "SPY"
        assert result["date"] == "2024-01-15"
