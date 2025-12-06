"""
Tests for AI Explainer API routes.

Tests cover:
- POST /api/ai-explainer endpoint
- GET /api/ai-explainer/health endpoint
- DELETE /api/ai-explainer/cache endpoint
- Request validation
- Error handling
- Rate limiting
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


# =============================================================================
# Test POST /api/ai-explainer
# =============================================================================

class TestAiExplainerEndpoint:
    """Tests for POST /api/ai-explainer endpoint."""

    @patch("app.routes.ai_explainer.AI_EXPLAINER_ENABLED", False)
    def test_returns_disabled_when_feature_off(self, client):
        """Should return disabled response when feature flag is off."""
        response = client.post(
            "/api/ai-explainer",
            json={
                "pageId": "leaps_ranker",
                "contextType": "roi_simulator",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {"symbol": "SPY"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "disabled" in data["error"].lower()

    @patch("app.routes.ai_explainer.AI_EXPLAINER_ENABLED", True)
    @patch("app.routes.ai_explainer.ai_explainer_service")
    def test_returns_successful_explanation(self, mock_service, client):
        """Should return successful explanation."""
        mock_service.get_ai_explanation.return_value = {
            "success": True,
            "pageId": "leaps_ranker",
            "contextType": "roi_simulator",
            "content": {
                "summary": "Test summary",
                "key_insights": [
                    {"title": "Test", "description": "Test insight", "sentiment": "neutral"}
                ],
                "risks": [
                    {"risk": "Test risk", "severity": "medium"}
                ],
                "watch_items": [
                    {"item": "Test item", "trigger": "Test trigger"}
                ],
                "disclaimer": "Test disclaimer",
            },
            "cached": False,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        response = client.post(
            "/api/ai-explainer",
            json={
                "pageId": "leaps_ranker",
                "contextType": "roi_simulator",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {"symbol": "SPY", "underlying_price": 600},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content"]["summary"] == "Test summary"
        assert len(data["content"]["key_insights"]) == 1

    @patch("app.routes.ai_explainer.AI_EXPLAINER_ENABLED", True)
    @patch("app.routes.ai_explainer.USE_MOCK_RESPONSES", True)
    @patch("app.routes.ai_explainer.ai_explainer_service")
    def test_uses_mock_responses_when_enabled(self, mock_service, client):
        """Should use mock responses when mock mode enabled."""
        mock_service.get_mock_explanation.return_value = {
            "success": True,
            "pageId": "leaps_ranker",
            "contextType": "roi_simulator",
            "content": {
                "summary": "Mock summary",
                "key_insights": [],
                "risks": [],
                "watch_items": [],
                "disclaimer": "Mock disclaimer",
            },
            "cached": False,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        response = client.post(
            "/api/ai-explainer",
            json={
                "pageId": "leaps_ranker",
                "contextType": "roi_simulator",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {"symbol": "SPY"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_service.get_mock_explanation.assert_called_once()

    def test_validates_required_fields(self, client):
        """Should validate required fields."""
        # Missing pageId
        response = client.post(
            "/api/ai-explainer",
            json={
                "contextType": "roi_simulator",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {},
            },
        )
        assert response.status_code == 422

        # Missing contextType
        response = client.post(
            "/api/ai-explainer",
            json={
                "pageId": "leaps_ranker",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {},
            },
        )
        assert response.status_code == 422

        # Missing timestamp
        response = client.post(
            "/api/ai-explainer",
            json={
                "pageId": "leaps_ranker",
                "contextType": "roi_simulator",
                "metadata": {},
            },
        )
        assert response.status_code == 422

    def test_validates_page_id_whitelist(self, client):
        """Should validate pageId against whitelist."""
        # Invalid pageId not in whitelist
        response = client.post(
            "/api/ai-explainer",
            json={
                "pageId": "invalid_page",
                "contextType": "roi_simulator",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {},
            },
        )
        assert response.status_code == 422

    def test_validates_context_type_whitelist(self, client):
        """Should validate contextType against whitelist."""
        response = client.post(
            "/api/ai-explainer",
            json={
                "pageId": "leaps_ranker",
                "contextType": "unknown_context",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {},
            },
        )
        assert response.status_code == 422

    def test_validates_metadata_size(self, client):
        """Should validate metadata size limit."""
        # Create metadata larger than 10KB limit
        large_metadata = {"data": "x" * 15000}

        response = client.post(
            "/api/ai-explainer",
            json={
                "pageId": "leaps_ranker",
                "contextType": "roi_simulator",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": large_metadata,
            },
        )
        assert response.status_code == 422

    @patch("app.routes.ai_explainer.AI_EXPLAINER_ENABLED", True)
    @patch("app.routes.ai_explainer.ai_explainer_service")
    def test_handles_service_exception(self, mock_service, client):
        """Should handle service exceptions gracefully."""
        mock_service.get_ai_explanation.side_effect = Exception("Internal error")

        response = client.post(
            "/api/ai-explainer",
            json={
                "pageId": "leaps_ranker",
                "contextType": "roi_simulator",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {},
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "unavailable" in data["detail"].lower()

    @patch("app.routes.ai_explainer.AI_EXPLAINER_ENABLED", True)
    @patch("app.routes.ai_explainer.ai_explainer_service")
    def test_returns_cached_response(self, mock_service, client):
        """Should return cached response with cachedAt."""
        mock_service.get_ai_explanation.return_value = {
            "success": True,
            "pageId": "leaps_ranker",
            "contextType": "roi_simulator",
            "content": {
                "summary": "Cached summary",
                "key_insights": [],
                "risks": [],
                "watch_items": [],
                "disclaimer": "Disclaimer",
            },
            "cached": True,
            "cachedAt": "2024-01-01T00:00:00Z",
            "timestamp": "2024-01-01T12:00:00Z",
        }

        response = client.post(
            "/api/ai-explainer",
            json={
                "pageId": "leaps_ranker",
                "contextType": "roi_simulator",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["cached"] is True
        assert data["cachedAt"] == "2024-01-01T00:00:00Z"


# =============================================================================
# Test GET /api/ai-explainer/health
# =============================================================================

class TestAiExplainerHealth:
    """Tests for GET /api/ai-explainer/health endpoint."""

    @patch("app.routes.ai_explainer.AI_EXPLAINER_ENABLED", True)
    @patch("app.routes.ai_explainer.USE_MOCK_RESPONSES", False)
    @patch("app.routes.ai_explainer.ai_explainer_service")
    def test_returns_healthy_status(self, mock_service, client):
        """Should return healthy status when enabled."""
        mock_service.load_config.return_value = {
            "model": {"name": "gemini-2.0-flash-exp"},
            "cache": {"enabled": True},
            "rate_limits": {"hourly_max": 100, "daily_max": 500},
        }

        response = client.get("/api/ai-explainer/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["feature_enabled"] is True
        assert data["model"] == "gemini-2.0-flash-exp"

    @patch("app.routes.ai_explainer.AI_EXPLAINER_ENABLED", False)
    def test_returns_disabled_status(self, client):
        """Should return disabled status when feature off."""
        with patch("app.routes.ai_explainer.ai_explainer_service") as mock_service:
            mock_service.load_config.return_value = {"model": {}, "cache": {}, "rate_limits": {}}

            response = client.get("/api/ai-explainer/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "disabled"
            assert data["feature_enabled"] is False

    @patch("app.routes.ai_explainer.ai_explainer_service")
    def test_returns_degraded_on_config_error(self, mock_service, client):
        """Should return degraded status on config error."""
        mock_service.load_config.side_effect = Exception("Config load failed")

        response = client.get("/api/ai-explainer/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert "error" in data


# =============================================================================
# Test DELETE /api/ai-explainer/cache
# =============================================================================

class TestAiExplainerCacheClear:
    """Tests for DELETE /api/ai-explainer/cache endpoint."""

    @patch("app.routes.ai_explainer.ai_explainer_service")
    def test_clears_cache_successfully(self, mock_service, client):
        """Should clear cache and return count."""
        mock_service.clear_cache.return_value = 5

        response = client.delete("/api/ai-explainer/cache")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "5" in data["message"]

    @patch("app.routes.ai_explainer.ai_explainer_service")
    def test_handles_cache_clear_error(self, mock_service, client):
        """Should handle cache clear error."""
        mock_service.clear_cache.side_effect = Exception("Cache clear failed")

        response = client.delete("/api/ai-explainer/cache")

        assert response.status_code == 500
        data = response.json()
        assert "Failed to clear cache" in data["detail"]


# =============================================================================
# Test Client IP Extraction
# =============================================================================

class TestGetClientIp:
    """Tests for _get_client_ip function."""

    def test_extracts_from_x_forwarded_for(self, client):
        """Should extract IP from X-Forwarded-For header."""
        from app.routes.ai_explainer import _get_client_ip

        mock_request = MagicMock()
        mock_request.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}
        mock_request.client = MagicMock(host="127.0.0.1")

        ip = _get_client_ip(mock_request)
        assert ip == "1.2.3.4"

    def test_falls_back_to_client_host(self, client):
        """Should fall back to client.host."""
        from app.routes.ai_explainer import _get_client_ip

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client = MagicMock(host="192.168.1.1")

        ip = _get_client_ip(mock_request)
        assert ip == "192.168.1.1"

    def test_returns_unknown_when_no_client(self, client):
        """Should return 'unknown' when no client info."""
        from app.routes.ai_explainer import _get_client_ip

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client = None

        ip = _get_client_ip(mock_request)
        assert ip == "unknown"


# =============================================================================
# Test Request Model Validation
# =============================================================================

class TestAiExplainerRequestValidation:
    """Tests for AiExplainerRequest model validation."""

    def test_valid_request(self, client):
        """Should accept valid request."""
        from app.models import AiExplainerRequest

        request = AiExplainerRequest(
            pageId="leaps_ranker",
            contextType="roi_simulator",
            timestamp="2024-01-01T00:00:00Z",
            metadata={"symbol": "SPY"},
        )

        assert request.pageId == "leaps_ranker"
        assert request.contextType == "roi_simulator"

    def test_rejects_invalid_page_id(self):
        """Should reject invalid pageId (not in whitelist)."""
        from app.models import AiExplainerRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AiExplainerRequest(
                pageId="invalid_page",
                contextType="roi_simulator",
                timestamp="2024-01-01T00:00:00Z",
                metadata={},
            )

    def test_rejects_invalid_context_type(self):
        """Should reject invalid contextType (not in whitelist)."""
        from app.models import AiExplainerRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AiExplainerRequest(
                pageId="leaps_ranker",
                contextType="invalid_context",
                timestamp="2024-01-01T00:00:00Z",
                metadata={},
            )

    def test_requires_timestamp(self):
        """Should require timestamp field."""
        from app.models import AiExplainerRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AiExplainerRequest(
                pageId="leaps_ranker",
                contextType="roi_simulator",
                metadata={},
            )

    def test_rejects_oversized_metadata(self):
        """Should reject metadata larger than 10KB."""
        from app.models import AiExplainerRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AiExplainerRequest(
                pageId="leaps_ranker",
                contextType="roi_simulator",
                timestamp="2024-01-01T00:00:00Z",
                metadata={"data": "x" * 15000},
            )


# =============================================================================
# Test Response Model
# =============================================================================

class TestAiExplainerResponseModel:
    """Tests for AiExplainerResponse model."""

    def test_success_response(self):
        """Should create success response."""
        from app.models import AiExplainerResponse, AiExplainerContent

        content = AiExplainerContent(
            summary="Test summary",
            key_insights=[],
            risks=[],
            watch_items=[],
            disclaimer="Test disclaimer",
        )

        response = AiExplainerResponse(
            success=True,
            pageId="leaps_ranker",
            contextType="roi_simulator",
            content=content,
            cached=False,
            timestamp="2024-01-01T00:00:00Z",
        )

        assert response.success is True
        assert response.content.summary == "Test summary"

    def test_error_response(self):
        """Should create error response."""
        from app.models import AiExplainerResponse

        response = AiExplainerResponse(
            success=False,
            pageId="leaps_ranker",
            contextType="roi_simulator",
            content=None,
            cached=False,
            error="Rate limit exceeded",
            timestamp="2024-01-01T00:00:00Z",
        )

        assert response.success is False
        assert response.error == "Rate limit exceeded"
        assert response.content is None
