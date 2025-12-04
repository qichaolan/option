"""
Tests for AI Score API endpoints.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# Create a minimal test app for testing the router
@pytest.fixture
def client():
    """Create test client for AI score API."""
    from fastapi import FastAPI
    from web.app.routes.ai_score import router

    app = FastAPI()
    app.include_router(router)

    return TestClient(app)


class TestGetAIScoreEndpoint:
    """Tests for GET /api/ai-score endpoint."""

    @patch("web.app.routes.ai_score.get_ai_score")
    def test_returns_score_for_valid_symbol(self, mock_get_score, client):
        """Should return score for valid symbol."""
        mock_get_score.return_value = {
            "symbol": "SPY",
            "date": "2024-01-15",
            "score_raw": 0.5,
            "score_0_1": 0.75,
            "ai_rating": "Buy",
        }

        response = client.get("/api/ai-score?symbol=SPY")

        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "SPY"
        assert data["date"] == "2024-01-15"
        assert data["score_raw"] == 0.5
        assert data["score_0_1"] == 0.75
        assert data["ai_rating"] == "Buy"

    @patch("web.app.routes.ai_score.get_ai_score")
    def test_passes_refresh_parameter(self, mock_get_score, client):
        """Should pass refresh parameter to service."""
        mock_get_score.return_value = {
            "symbol": "SPY",
            "date": "2024-01-15",
            "score_raw": 0.5,
            "score_0_1": 0.75,
            "ai_rating": "Buy",
        }

        client.get("/api/ai-score?symbol=SPY&refresh=true")

        mock_get_score.assert_called_once_with("SPY", force_refresh=True)

    def test_requires_symbol_parameter(self, client):
        """Should return 422 when symbol is missing."""
        response = client.get("/api/ai-score")

        assert response.status_code == 422

    @patch("web.app.routes.ai_score.get_ai_score")
    def test_returns_404_for_not_found(self, mock_get_score, client):
        """Should return 404 when symbol not found."""
        mock_get_score.side_effect = ValueError("No data not found for INVALID")

        response = client.get("/api/ai-score?symbol=INVALID")

        assert response.status_code == 404

    def test_returns_422_for_invalid_symbol_format(self, client):
        """Should return 422 for symbol with invalid characters (FastAPI validation)."""
        # Numbers are rejected by the regex pattern at the API level
        response = client.get("/api/ai-score?symbol=123")
        assert response.status_code == 422

    @patch("web.app.routes.ai_score.get_ai_score")
    def test_returns_400_for_invalid_symbol_from_service(self, mock_get_score, client):
        """Should return 400 for invalid symbol from service layer."""
        mock_get_score.side_effect = ValueError("Invalid symbol format")

        response = client.get("/api/ai-score?symbol=INVALID")

        assert response.status_code == 400

    @patch("web.app.routes.ai_score.get_ai_score")
    def test_returns_500_for_server_error(self, mock_get_score, client):
        """Should return 500 for unexpected errors."""
        mock_get_score.side_effect = RuntimeError("Database connection failed")

        response = client.get("/api/ai-score?symbol=SPY")

        assert response.status_code == 500
        assert "Server error" in response.json()["detail"]

    @patch("web.app.routes.ai_score.get_ai_score")
    def test_handles_all_ai_ratings(self, mock_get_score, client):
        """Should handle all AI rating values."""
        ratings = ["Strong Buy", "Buy", "Hold", "Sell", "Must Sell"]

        for rating in ratings:
            mock_get_score.return_value = {
                "symbol": "SPY",
                "date": "2024-01-15",
                "score_raw": 0.5,
                "score_0_1": 0.5,
                "ai_rating": rating,
            }

            response = client.get("/api/ai-score?symbol=SPY")

            assert response.status_code == 200
            assert response.json()["ai_rating"] == rating

    def test_symbol_max_length(self, client):
        """Should reject symbols longer than 10 characters."""
        response = client.get("/api/ai-score?symbol=VERYLONGSYMBOL")

        assert response.status_code == 422

    @patch("web.app.routes.ai_score.get_ai_score")
    def test_uppercases_symbol(self, mock_get_score, client):
        """Should accept lowercase symbols."""
        mock_get_score.return_value = {
            "symbol": "SPY",
            "date": "2024-01-15",
            "score_raw": 0.5,
            "score_0_1": 0.75,
            "ai_rating": "Buy",
        }

        response = client.get("/api/ai-score?symbol=spy")

        assert response.status_code == 200
        # Service should have been called (it handles uppercasing)
        mock_get_score.assert_called_once()
