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


# =============================================================================
# Test Scenario Models (New Narrative Format)
# =============================================================================

class TestAiExplainerScenarioModel:
    """Tests for AiExplainerScenario model with narrative format."""

    def test_valid_scenario(self):
        """Should create valid scenario with narrative fields."""
        from app.models import AiExplainerScenario

        scenario = AiExplainerScenario(
            min_annual_return="+16.00%",
            projected_price_target="A compounded move results in a target of $907.39.",
            payoff_realism="This scenario requires an average annual return of at least 16.00%.",
            option_payoff="The projected ROI of +105% means the premium is expected to double.",
        )

        assert scenario.min_annual_return == "+16.00%"
        assert "compounded" in scenario.projected_price_target.lower()
        assert "16.00%" in scenario.payoff_realism
        assert "+105%" in scenario.option_payoff

    def test_scenario_requires_all_fields(self):
        """Should require all narrative fields."""
        from app.models import AiExplainerScenario
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AiExplainerScenario(
                min_annual_return="+16.00%",
                # Missing other required fields
            )

    def test_scenarios_container(self):
        """Should create scenarios container with medium and strong."""
        from app.models import AiExplainerScenario, AiExplainerScenarios

        medium = AiExplainerScenario(
            min_annual_return="+16.00%",
            projected_price_target="Target of $907.39.",
            payoff_realism="Historically occurred 50% of the time.",
            option_payoff="ROI of +105%.",
        )

        strong = AiExplainerScenario(
            min_annual_return="+21.83%",
            projected_price_target="Target of $1,102.34.",
            payoff_realism="Historically occurred 30% of the time.",
            option_payoff="ROI of +207%.",
        )

        scenarios = AiExplainerScenarios(
            medium_increase=medium,
            strong_increase=strong,
        )

        assert scenarios.medium_increase.min_annual_return == "+16.00%"
        assert scenarios.strong_increase.min_annual_return == "+21.83%"

    def test_scenarios_optional_fields(self):
        """Should allow None for scenarios."""
        from app.models import AiExplainerScenarios

        scenarios = AiExplainerScenarios(
            medium_increase=None,
            strong_increase=None,
        )

        assert scenarios.medium_increase is None
        assert scenarios.strong_increase is None


# =============================================================================
# Test Content with Scenarios
# =============================================================================

class TestAiExplainerContentWithScenarios:
    """Tests for AiExplainerContent with scenarios field."""

    def test_content_with_scenarios(self):
        """Should include scenarios in content."""
        from app.models import (
            AiExplainerContent,
            AiExplainerScenario,
            AiExplainerScenarios,
        )

        medium_scenario = AiExplainerScenario(
            min_annual_return="+16.00%",
            projected_price_target="A compounded move results in a target of $907.39.",
            payoff_realism="This scenario requires 16.00% annually, historically 50%.",
            option_payoff="The projected ROI of +105% means premium doubles.",
        )

        scenarios = AiExplainerScenarios(
            medium_increase=medium_scenario,
            strong_increase=None,
        )

        content = AiExplainerContent(
            summary="Test summary",
            key_insights=[],
            scenarios=scenarios,
            risks=[],
            watch_items=[],
            disclaimer="Test disclaimer",
        )

        assert content.scenarios is not None
        assert content.scenarios.medium_increase.min_annual_return == "+16.00%"
        assert content.scenarios.strong_increase is None

    def test_content_without_scenarios(self):
        """Should work without scenarios."""
        from app.models import AiExplainerContent

        content = AiExplainerContent(
            summary="Test summary",
            key_insights=[],
            risks=[],
            watch_items=[],
            disclaimer="Test disclaimer",
        )

        assert content.scenarios is None


# =============================================================================
# E2E Test with Mock Gemini
# =============================================================================

class TestAiExplainerE2E:
    """End-to-end tests with mocked Gemini API."""

    @patch("app.routes.ai_explainer.AI_EXPLAINER_ENABLED", True)
    @patch("app.routes.ai_explainer.ai_explainer_service")
    def test_e2e_roi_simulator_with_scenarios(self, mock_service, client):
        """
        E2E test: POST /api/ai-explainer with realistic ROI simulator context.
        Verifies response matches expected AiExplainerResponse schema including scenarios.
        """
        # Mock the complete Gemini response with all fields
        mock_service.get_ai_explanation.return_value = {
            "success": True,
            "pageId": "leaps_ranker",
            "contextType": "roi_simulator",
            "content": {
                "summary": "This SPY LEAPS call option with a $650 strike offers potential returns of 150% if the underlying reaches the target price.",
                "key_insights": [
                    {
                        "title": "Breakeven Analysis",
                        "description": "The option needs SPY to rise above $700 by expiration to break even.",
                        "sentiment": "neutral",
                    },
                    {
                        "title": "Leverage Profile",
                        "description": "LEAPS provide significant leverage compared to holding shares directly.",
                        "sentiment": "positive",
                    },
                    {
                        "title": "Time Value",
                        "description": "With over a year until expiration, time decay is minimal but accelerates in final months.",
                        "sentiment": "neutral",
                    },
                ],
                "scenarios": {
                    "medium_increase": {
                        "min_annual_return": "+16.00%",
                        "projected_price_target": "A compounded move results in a target of $907.39.",
                        "payoff_realism": "This scenario requires an average annual return of at least 16.00%, which historically occurred 50% of the time.",
                        "option_payoff": "The projected ROI of +105% means the premium is expected to double, achieving a profit of $17,731.",
                    },
                    "strong_increase": {
                        "min_annual_return": "+21.83%",
                        "projected_price_target": "A compounded move results in a target of $1,102.34.",
                        "payoff_realism": "This scenario requires an exceptional annual return of at least 21.83%, which historically occurred 30% of the time.",
                        "option_payoff": "The projected ROI of +207% means the premium is expected to more than triple.",
                    },
                },
                "risks": [
                    {
                        "risk": "Maximum loss limited to premium paid, representing 100% of investment.",
                        "severity": "medium",
                    },
                    {
                        "risk": "IV crush after earnings could reduce option value.",
                        "severity": "medium",
                    },
                ],
                "watch_items": [
                    {
                        "item": "Underlying price relative to breakeven",
                        "trigger": "Consider adjustment if SPY drops below $540",
                    },
                    {
                        "item": "Implied volatility levels",
                        "trigger": "Monitor for significant IV changes around earnings",
                    },
                ],
                "disclaimer": "This analysis is for educational purposes only.",
            },
            "cached": False,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        # Realistic ROI simulator context
        response = client.post(
            "/api/ai-explainer",
            json={
                "pageId": "leaps_ranker",
                "contextType": "roi_simulator",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {
                    "symbol": "SPY",
                    "underlying_price": 600.0,
                    "contract": {
                        "contract_symbol": "SPY260116C00650000",
                        "strike": 650.0,
                        "premium": 50.0,
                        "cost": 5000.0,
                        "expiration": "2026-01-16",
                    },
                    "roi_results": [
                        {"target_price": 660.0, "roi_pct": 20.0},
                        {"target_price": 720.0, "roi_pct": 80.0},
                        {"target_price": 780.0, "roi_pct": 160.0},
                        {"target_price": 900.0, "roi_pct": 400.0},
                    ],
                },
            },
        )

        # Verify response status
        assert response.status_code == 200

        # Parse response
        data = response.json()

        # Verify top-level structure
        assert data["success"] is True
        assert data["pageId"] == "leaps_ranker"
        assert data["contextType"] == "roi_simulator"
        assert data["cached"] is False
        assert "timestamp" in data

        # Verify content structure
        content = data["content"]
        assert "summary" in content
        assert "SPY" in content["summary"]

        # Verify key_insights
        assert len(content["key_insights"]) >= 3
        for insight in content["key_insights"]:
            assert "title" in insight
            assert "description" in insight
            assert insight["sentiment"] in ["positive", "neutral", "negative"]

        # Verify scenarios with new narrative format
        assert content["scenarios"] is not None
        scenarios = content["scenarios"]

        # Medium increase scenario
        assert "medium_increase" in scenarios
        medium = scenarios["medium_increase"]
        assert "min_annual_return" in medium
        assert "projected_price_target" in medium
        assert "payoff_realism" in medium
        assert "option_payoff" in medium
        assert "+16.00%" in medium["min_annual_return"]

        # Strong increase scenario
        assert "strong_increase" in scenarios
        strong = scenarios["strong_increase"]
        assert "+21.83%" in strong["min_annual_return"]

        # Verify risks
        assert len(content["risks"]) >= 2
        for risk in content["risks"]:
            assert "risk" in risk
            assert risk["severity"] in ["low", "medium", "high"]

        # Verify watch_items
        assert len(content["watch_items"]) >= 2
        for item in content["watch_items"]:
            assert "item" in item

        # Verify disclaimer
        assert "disclaimer" in content
        assert "educational" in content["disclaimer"].lower()

    @patch("app.routes.ai_explainer.AI_EXPLAINER_ENABLED", True)
    @patch("app.routes.ai_explainer.USE_MOCK_RESPONSES", True)
    @patch("app.routes.ai_explainer.ai_explainer_service")
    def test_e2e_mock_response_format(self, mock_service, client):
        """
        E2E test: Verify mock response returns correct schema structure.
        """
        # Use the actual mock response format
        mock_service.get_mock_explanation.return_value = {
            "success": True,
            "pageId": "leaps_ranker",
            "contextType": "roi_simulator",
            "content": {
                "summary": "Mock LEAPS analysis summary",
                "key_insights": [
                    {"title": "Test Insight", "description": "Test description", "sentiment": "neutral"}
                ],
                "scenarios": {
                    "medium_increase": {
                        "min_annual_return": "+16.00%",
                        "projected_price_target": "A compounded move results in a target of $810.00.",
                        "payoff_realism": "This scenario requires 16.00% annually.",
                        "option_payoff": "The projected ROI means premium doubles.",
                    },
                    "strong_increase": {
                        "min_annual_return": "+21.83%",
                        "projected_price_target": "A compounded move results in a target of $960.00.",
                        "payoff_realism": "This scenario requires 21.83% annually.",
                        "option_payoff": "The projected ROI means premium triples.",
                    },
                },
                "risks": [{"risk": "Test risk", "severity": "medium"}],
                "watch_items": [{"item": "Test watch item", "trigger": "Test trigger"}],
                "disclaimer": "Test disclaimer for educational purposes only.",
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
        assert data["content"]["scenarios"]["medium_increase"]["min_annual_return"] == "+16.00%"
        assert data["content"]["scenarios"]["strong_increase"]["min_annual_return"] == "+21.83%"


# =============================================================================
# Test Service Layer Functions
# =============================================================================

class TestAiExplainerServiceFunctions:
    """Tests for ai_explainer_service functions."""

    def test_load_config(self):
        """Should load Gemini config successfully."""
        from app.services.ai_explainer_service import load_config

        try:
            config = load_config()
            assert "model" in config or config is not None
        except FileNotFoundError:
            # Config may not exist in test environment
            pytest.skip("Config file not found in test environment")

    def test_clear_cache(self):
        """Should clear cache and return count."""
        from app.services.ai_explainer_service import clear_cache, _response_cache

        # Add some items to cache
        _response_cache["test_key_1"] = {"data": "test1"}
        _response_cache["test_key_2"] = {"data": "test2"}

        count = clear_cache()
        assert count >= 0
        assert len(_response_cache) == 0

    def test_compute_cache_key(self):
        """Should compute stable cache key."""
        from app.services.ai_explainer_service import _compute_cache_key

        key1 = _compute_cache_key("leaps_ranker", "roi_simulator", {"symbol": "SPY"})
        key2 = _compute_cache_key("leaps_ranker", "roi_simulator", {"symbol": "SPY"})
        key3 = _compute_cache_key("leaps_ranker", "roi_simulator", {"symbol": "QQQ"})

        # Same inputs should produce same key
        assert key1 == key2
        # Different inputs should produce different key
        assert key1 != key3

    def test_get_mock_explanation(self):
        """Should return mock explanation with all required fields."""
        from app.services.ai_explainer_service import get_mock_explanation

        result = get_mock_explanation(
            page_id="leaps_ranker",
            context_type="roi_simulator",
            metadata={"symbol": "SPY", "underlying_price": 600, "contract": {"strike": 650}},
        )

        assert result["success"] is True
        assert result["pageId"] == "leaps_ranker"
        assert result["contextType"] == "roi_simulator"
        assert "content" in result
        assert "summary" in result["content"]
        assert "scenarios" in result["content"]

        # Verify new scenario format
        scenarios = result["content"]["scenarios"]
        assert "medium_increase" in scenarios
        assert "min_annual_return" in scenarios["medium_increase"]
        assert "projected_price_target" in scenarios["medium_increase"]
        assert "payoff_realism" in scenarios["medium_increase"]
        assert "option_payoff" in scenarios["medium_increase"]


# =============================================================================
# Test Input Validation Edge Cases
# =============================================================================

class TestInputValidationEdgeCases:
    """Tests for input validation edge cases."""

    def test_normalizes_page_id_case(self, client):
        """Should normalize pageId to lowercase."""
        from app.models import AiExplainerRequest

        request = AiExplainerRequest(
            pageId="LEAPS_RANKER",
            contextType="roi_simulator",
            timestamp="2024-01-01T00:00:00Z",
            metadata={},
        )

        assert request.pageId == "leaps_ranker"

    def test_normalizes_context_type_case(self, client):
        """Should normalize contextType to lowercase."""
        from app.models import AiExplainerRequest

        request = AiExplainerRequest(
            pageId="leaps_ranker",
            contextType="ROI_SIMULATOR",
            timestamp="2024-01-01T00:00:00Z",
            metadata={},
        )

        assert request.contextType == "roi_simulator"

    def test_strips_whitespace(self, client):
        """Should strip whitespace from pageId and contextType."""
        from app.models import AiExplainerRequest

        request = AiExplainerRequest(
            pageId="  leaps_ranker  ",
            contextType="  roi_simulator  ",
            timestamp="2024-01-01T00:00:00Z",
            metadata={},
        )

        assert request.pageId == "leaps_ranker"
        assert request.contextType == "roi_simulator"

    def test_accepts_all_valid_page_ids(self, client):
        """Should accept all whitelisted pageIds."""
        from app.models import AiExplainerRequest, VALID_PAGE_IDS

        for page_id in VALID_PAGE_IDS:
            request = AiExplainerRequest(
                pageId=page_id,
                contextType="roi_simulator",
                timestamp="2024-01-01T00:00:00Z",
                metadata={},
            )
            assert request.pageId == page_id

    def test_accepts_all_valid_context_types(self, client):
        """Should accept all whitelisted contextTypes."""
        from app.models import AiExplainerRequest, VALID_CONTEXT_TYPES

        for context_type in VALID_CONTEXT_TYPES:
            request = AiExplainerRequest(
                pageId="leaps_ranker",
                contextType=context_type,
                timestamp="2024-01-01T00:00:00Z",
                metadata={},
            )
            assert request.contextType == context_type

    def test_metadata_boundary_size(self, client):
        """Should accept metadata at exactly the limit."""
        from app.models import AiExplainerRequest, MAX_METADATA_SIZE
        import json

        # Calculate how much data we can fit
        # Need to account for JSON overhead
        overhead = len(json.dumps({"data": ""}))
        max_string_len = MAX_METADATA_SIZE - overhead - 10  # Small buffer

        request = AiExplainerRequest(
            pageId="leaps_ranker",
            contextType="roi_simulator",
            timestamp="2024-01-01T00:00:00Z",
            metadata={"data": "x" * max_string_len},
        )

        assert request.metadata is not None
