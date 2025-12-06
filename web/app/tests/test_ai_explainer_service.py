"""
Tests for AI Explainer service layer.

Tests cover:
- Configuration loading
- Prompt management
- Caching layer
- Rate limiting
- Metadata sanitization
- Gemini client
- Response parsing
- Main service function
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Import directly to avoid triggering other service imports that may have missing deps
import app.services.ai_explainer_service as ai_explainer_service


# =============================================================================
# Test Configuration
# =============================================================================

class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_config_from_file(self, tmp_path):
        """Should load config from YAML file."""
        config_dir = tmp_path / "ai" / "config"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "gemini.yaml"
        config_file.write_text("""
model:
  name: gemini-2.0-flash-exp
  temperature: 0.3
cache:
  enabled: true
  ttl_seconds: 86400
""")

        with patch.object(ai_explainer_service, "_get_config_path", return_value=config_file):
            config = ai_explainer_service.load_config()

        assert config["model"]["name"] == "gemini-2.0-flash-exp"
        assert config["model"]["temperature"] == 0.3
        assert config["cache"]["enabled"] is True

    def test_raises_when_config_not_found(self):
        """Should raise FileNotFoundError when config not found."""
        with patch.object(ai_explainer_service, "_get_config_path", side_effect=FileNotFoundError("Config not found")):
            with pytest.raises(FileNotFoundError):
                ai_explainer_service.load_config()


# =============================================================================
# Test Prompt Management
# =============================================================================

class TestLoadSystemPrompt:
    """Tests for load_system_prompt function."""

    def test_loads_specific_prompt(self, tmp_path):
        """Should load page__context specific prompt."""
        prompts_dir = tmp_path / "ai" / "prompts"
        prompts_dir.mkdir(parents=True)
        specific_prompt = prompts_dir / "leaps_ranker__roi_simulator.md"
        specific_prompt.write_text("# LEAPS ROI Simulator Prompt")

        with patch.object(ai_explainer_service, "_get_prompts_dir", return_value=prompts_dir):
            prompt = ai_explainer_service.load_system_prompt("leaps_ranker", "roi_simulator")

        assert "LEAPS ROI Simulator Prompt" in prompt

    def test_falls_back_to_page_prompt(self, tmp_path):
        """Should fall back to page-level prompt."""
        prompts_dir = tmp_path / "ai" / "prompts"
        prompts_dir.mkdir(parents=True)
        page_prompt = prompts_dir / "leaps_ranker.md"
        page_prompt.write_text("# LEAPS Ranker Page Prompt")

        with patch.object(ai_explainer_service, "_get_prompts_dir", return_value=prompts_dir):
            prompt = ai_explainer_service.load_system_prompt("leaps_ranker", "unknown_context")

        assert "LEAPS Ranker Page Prompt" in prompt

    def test_falls_back_to_default_prompt(self, tmp_path):
        """Should fall back to default prompt."""
        prompts_dir = tmp_path / "ai" / "prompts"
        prompts_dir.mkdir(parents=True)
        default_prompt = prompts_dir / "default_explainer.md"
        default_prompt.write_text("# Default Explainer Prompt")

        with patch.object(ai_explainer_service, "_get_prompts_dir", return_value=prompts_dir):
            prompt = ai_explainer_service.load_system_prompt("unknown_page", "unknown_context")

        assert "Default Explainer Prompt" in prompt

    def test_raises_when_no_prompt_found(self, tmp_path):
        """Should raise FileNotFoundError when no prompt found."""
        prompts_dir = tmp_path / "ai" / "prompts"
        prompts_dir.mkdir(parents=True)

        with patch.object(ai_explainer_service, "_get_prompts_dir", return_value=prompts_dir):
            with pytest.raises(FileNotFoundError, match="No prompt found"):
                ai_explainer_service.load_system_prompt("unknown", "unknown")


# =============================================================================
# Test Caching Layer
# =============================================================================

class TestComputeCacheKey:
    """Tests for _compute_cache_key function."""

    def test_produces_stable_key(self):
        """Should produce stable key for same inputs."""
        key1 = ai_explainer_service._compute_cache_key("page", "context", {"a": 1})
        key2 = ai_explainer_service._compute_cache_key("page", "context", {"a": 1})
        assert key1 == key2

    def test_produces_different_keys_for_different_inputs(self):
        """Should produce different keys for different inputs."""
        key1 = ai_explainer_service._compute_cache_key("page1", "context", {"a": 1})
        key2 = ai_explainer_service._compute_cache_key("page2", "context", {"a": 1})
        assert key1 != key2

    def test_key_is_32_chars(self):
        """Should produce 32-character key."""
        key = ai_explainer_service._compute_cache_key("page", "context", {"a": 1})
        assert len(key) == 32


class TestCacheOperations:
    """Tests for cache get/set operations."""

    def setup_method(self):
        """Clear cache before each test."""
        ai_explainer_service._response_cache.clear()

    def test_get_cached_response_returns_none_when_empty(self):
        """Should return None when cache is empty."""
        result = ai_explainer_service.get_cached_response("nonexistent_key")
        assert result is None

    def test_set_and_get_cached_response(self):
        """Should store and retrieve cached response."""
        response = {"content": {"summary": "Test"}}
        ai_explainer_service.set_cached_response("test_key", response)

        cached = ai_explainer_service.get_cached_response("test_key")
        assert cached is not None
        assert cached["content"]["summary"] == "Test"
        assert "cached_at" in cached

    def test_expired_cache_returns_none(self):
        """Should return None for expired cache entries."""
        # Set cache entry with old timestamp
        ai_explainer_service._response_cache["old_key"] = {
            "content": {"summary": "Old"},
            "cached_at": (datetime.utcnow() - timedelta(hours=25)).isoformat(),
        }

        result = ai_explainer_service.get_cached_response("old_key", ttl_seconds=86400)
        assert result is None
        # Entry should be deleted
        assert "old_key" not in ai_explainer_service._response_cache

    def test_clear_cache(self):
        """Should clear all cached responses."""
        ai_explainer_service.set_cached_response("key1", {"a": 1})
        ai_explainer_service.set_cached_response("key2", {"b": 2})

        count = ai_explainer_service.clear_cache()

        assert count == 2
        assert len(ai_explainer_service._response_cache) == 0


# =============================================================================
# Test Rate Limiting
# =============================================================================

class TestRateLimiting:
    """Tests for rate limiting functions."""

    def setup_method(self):
        """Clear rate limit tracker before each test."""
        ai_explainer_service._rate_limit_tracker.clear()

    def test_allows_first_request(self):
        """Should allow first request from user."""
        allowed, error = ai_explainer_service.check_rate_limit("user1")
        assert allowed is True
        assert error is None

    def test_allows_requests_under_hourly_limit(self):
        """Should allow requests under hourly limit."""
        # Record 50 requests (under 100 hourly limit)
        for _ in range(50):
            ai_explainer_service.record_request("user1")

        allowed, error = ai_explainer_service.check_rate_limit("user1", hourly_max=100)
        assert allowed is True

    def test_blocks_requests_over_hourly_limit(self):
        """Should block requests over hourly limit."""
        # Record 100 requests
        for _ in range(100):
            ai_explainer_service.record_request("user1")

        allowed, error = ai_explainer_service.check_rate_limit("user1", hourly_max=100)
        assert allowed is False
        assert "Hourly rate limit exceeded" in error

    def test_blocks_requests_over_daily_limit(self):
        """Should block requests over daily limit."""
        # Record 500 requests
        for _ in range(500):
            ai_explainer_service.record_request("user1")

        # Set hourly_max high so we hit daily limit first
        allowed, error = ai_explainer_service.check_rate_limit("user1", hourly_max=1000, daily_max=500)
        assert allowed is False
        assert "Daily rate limit exceeded" in error

    def test_record_request(self):
        """Should record request timestamp."""
        ai_explainer_service.record_request("user1")

        assert "user1" in ai_explainer_service._rate_limit_tracker
        assert len(ai_explainer_service._rate_limit_tracker["user1"]["requests"]) == 1


class TestGetUserKey:
    """Tests for _get_user_key function."""

    def test_extracts_client_ip(self):
        """Should extract client IP from request info."""
        request_info = {"client_ip": "192.168.1.1"}
        key = ai_explainer_service._get_user_key(request_info)
        assert key == "192.168.1.1"

    def test_returns_anonymous_when_no_ip(self):
        """Should return 'anonymous' when no IP provided."""
        key = ai_explainer_service._get_user_key({})
        assert key == "anonymous"

        key = ai_explainer_service._get_user_key(None)
        assert key == "anonymous"


# =============================================================================
# Test Metadata Sanitization
# =============================================================================

class TestSanitizeMetadata:
    """Tests for sanitize_metadata function."""

    def test_redacts_pii_fields(self):
        """Should redact PII fields."""
        metadata = {
            "symbol": "SPY",
            "email": "user@example.com",
            "username": "testuser",
            "password": "secret123",
        }

        result = ai_explainer_service.sanitize_metadata(metadata)

        assert result["symbol"] == "SPY"
        assert result["email"] == "[REDACTED]"
        assert result["username"] == "[REDACTED]"
        assert result["password"] == "[REDACTED]"

    def test_preserves_safe_fields(self):
        """Should preserve non-PII fields."""
        metadata = {
            "symbol": "SPY",
            "underlying_price": 600.0,
            "strike": 650.0,
        }

        result = ai_explainer_service.sanitize_metadata(metadata)

        assert result["symbol"] == "SPY"
        assert result["underlying_price"] == 600.0
        assert result["strike"] == 650.0

    def test_handles_nested_dicts(self):
        """Should handle nested dictionaries."""
        metadata = {
            "contract": {
                "symbol": "SPY",
                "user_id": "12345",
            }
        }

        result = ai_explainer_service.sanitize_metadata(metadata)

        assert result["contract"]["symbol"] == "SPY"
        assert result["contract"]["user_id"] == "[REDACTED]"

    def test_handles_lists(self):
        """Should handle lists of dicts."""
        metadata = {
            "users": [
                {"name": "John", "score": 10},
                {"name": "Jane", "score": 20},
            ]
        }

        result = ai_explainer_service.sanitize_metadata(metadata)

        assert result["users"][0]["name"] == "[REDACTED]"
        assert result["users"][0]["score"] == 10


# =============================================================================
# Test Gemini Client
# =============================================================================

class TestGeminiClient:
    """Tests for GeminiClient class."""

    def test_init_with_config(self):
        """Should initialize with provided config."""
        config = {"model": {"name": "test-model"}}
        client = ai_explainer_service.GeminiClient(config)
        assert client.config == config

    def test_init_loads_config_when_none_provided(self):
        """Should load config when none provided."""
        with patch.object(ai_explainer_service, "load_config") as mock_load:
            mock_load.return_value = {"model": {"name": "default"}}
            client = ai_explainer_service.GeminiClient()
            mock_load.assert_called_once()
            assert client.config["model"]["name"] == "default"

    def test_model_is_lazy_loaded(self):
        """Should not initialize model until _get_model is called."""
        config = {"model": {"name": "test-model"}}
        client = ai_explainer_service.GeminiClient(config)
        # Model should not be loaded yet
        assert client._model is None

    def test_raises_import_error_when_genai_not_installed(self):
        """Should raise ImportError when google-generativeai not installed."""
        # This test verifies error handling when the module isn't available
        # Since we're in a test environment without google-generativeai installed,
        # we can test this directly
        try:
            import google.generativeai
            pytest.skip("google-generativeai is installed, skipping import error test")
        except ImportError:
            config = {"model": {"name": "test"}}
            client = ai_explainer_service.GeminiClient(config)
            with pytest.raises(ImportError, match="google-generativeai"):
                client._get_model()


# =============================================================================
# Test Response Parsing
# =============================================================================

class TestParseGeminiResponse:
    """Tests for parse_gemini_response function."""

    def test_parses_valid_json(self):
        """Should parse valid JSON response."""
        response = '{"summary": "Test", "key_insights": []}'
        result = ai_explainer_service.parse_gemini_response(response)

        assert result["summary"] == "Test"
        assert result["key_insights"] == []

    def test_handles_markdown_code_block(self):
        """Should handle markdown code block wrapping."""
        response = '```json\n{"summary": "Test"}\n```'
        result = ai_explainer_service.parse_gemini_response(response)

        assert result["summary"] == "Test"

    def test_handles_plain_code_block(self):
        """Should handle plain code block wrapping."""
        response = '```\n{"summary": "Test"}\n```'
        result = ai_explainer_service.parse_gemini_response(response)

        assert result["summary"] == "Test"

    def test_extracts_json_from_text(self):
        """Should extract JSON from surrounding text."""
        response = 'Here is my analysis:\n{"summary": "Test"}\nEnd of response.'
        result = ai_explainer_service.parse_gemini_response(response)

        assert result["summary"] == "Test"

    def test_raises_on_invalid_json(self):
        """Should raise ValueError on invalid JSON."""
        response = "This is not JSON at all"
        with pytest.raises(ValueError, match="Invalid JSON"):
            ai_explainer_service.parse_gemini_response(response)


# =============================================================================
# Test Main Service Function
# =============================================================================

class TestGetAIExplanation:
    """Tests for get_ai_explanation function."""

    def setup_method(self):
        """Clear caches before each test."""
        ai_explainer_service._response_cache.clear()
        ai_explainer_service._rate_limit_tracker.clear()

    @patch.object(ai_explainer_service, "AI_EXPLAINER_ENABLED", False)
    def test_returns_disabled_when_feature_off(self):
        """Should return disabled response when feature flag is off."""
        result = ai_explainer_service.get_ai_explanation(
            page_id="test",
            context_type="test",
            metadata={},
            timestamp="2024-01-01T00:00:00Z",
        )

        assert result["success"] is False
        assert "disabled" in result["error"].lower()

    @patch.object(ai_explainer_service, "load_config")
    @patch.object(ai_explainer_service, "check_rate_limit")
    def test_returns_error_when_rate_limited(self, mock_rate_limit, mock_config):
        """Should return error when rate limited."""
        mock_config.return_value = {"cache": {"enabled": True}, "rate_limits": {}}
        mock_rate_limit.return_value = (False, "Rate limit exceeded")

        result = ai_explainer_service.get_ai_explanation(
            page_id="test",
            context_type="test",
            metadata={},
            timestamp="2024-01-01T00:00:00Z",
            request_info={"client_ip": "1.2.3.4"},
        )

        assert result["success"] is False
        assert "Rate limit" in result["error"]

    @patch.object(ai_explainer_service, "load_config")
    @patch.object(ai_explainer_service, "get_cached_response")
    @patch.object(ai_explainer_service, "check_rate_limit")
    def test_returns_cached_response(self, mock_rate_limit, mock_cache, mock_config):
        """Should return cached response when available."""
        mock_config.return_value = {"cache": {"enabled": True, "ttl_seconds": 86400}, "rate_limits": {}}
        mock_rate_limit.return_value = (True, None)
        mock_cache.return_value = {
            "content": {"summary": "Cached response"},
            "cached_at": "2024-01-01T00:00:00Z",
        }

        result = ai_explainer_service.get_ai_explanation(
            page_id="test",
            context_type="test",
            metadata={},
            timestamp="2024-01-01T00:00:00Z",
        )

        assert result["success"] is True
        assert result["cached"] is True
        assert result["content"]["summary"] == "Cached response"

    @patch.object(ai_explainer_service, "load_config")
    @patch.object(ai_explainer_service, "load_system_prompt")
    @patch.object(ai_explainer_service, "check_rate_limit")
    def test_returns_error_when_prompt_not_found(self, mock_rate_limit, mock_prompt, mock_config):
        """Should return error when prompt not found."""
        mock_config.return_value = {"cache": {"enabled": False}, "rate_limits": {}}
        mock_rate_limit.return_value = (True, None)
        mock_prompt.side_effect = FileNotFoundError("No prompt found")

        result = ai_explainer_service.get_ai_explanation(
            page_id="test",
            context_type="test",
            metadata={},
            timestamp="2024-01-01T00:00:00Z",
        )

        assert result["success"] is False
        assert "configuration error" in result["error"].lower()

    @patch.object(ai_explainer_service, "load_config")
    @patch.object(ai_explainer_service, "load_system_prompt")
    @patch.object(ai_explainer_service, "GeminiClient")
    @patch.object(ai_explainer_service, "check_rate_limit")
    @patch.object(ai_explainer_service, "record_request")
    def test_successful_explanation(self, mock_record, mock_rate_limit, MockClient, mock_prompt, mock_config):
        """Should return successful explanation."""
        mock_config.return_value = {"cache": {"enabled": False}, "rate_limits": {}}
        mock_rate_limit.return_value = (True, None)
        mock_prompt.return_value = "System prompt"

        mock_client = MagicMock()
        mock_client.generate.return_value = json.dumps({
            "summary": "Test summary",
            "key_insights": [],
            "risks": [],
            "watch_items": [],
        })
        MockClient.return_value = mock_client

        result = ai_explainer_service.get_ai_explanation(
            page_id="leaps_ranker",
            context_type="roi_simulator",
            metadata={"symbol": "SPY"},
            timestamp="2024-01-01T00:00:00Z",
        )

        assert result["success"] is True
        assert result["cached"] is False
        assert result["content"]["summary"] == "Test summary"
        mock_record.assert_called_once()

    @patch.object(ai_explainer_service, "load_config")
    @patch.object(ai_explainer_service, "load_system_prompt")
    @patch.object(ai_explainer_service, "GeminiClient")
    @patch.object(ai_explainer_service, "check_rate_limit")
    def test_handles_timeout(self, mock_rate_limit, MockClient, mock_prompt, mock_config):
        """Should handle timeout gracefully."""
        mock_config.return_value = {"cache": {"enabled": False}, "rate_limits": {}}
        mock_rate_limit.return_value = (True, None)
        mock_prompt.return_value = "System prompt"

        mock_client = MagicMock()
        mock_client.generate.side_effect = TimeoutError("Request timed out")
        MockClient.return_value = mock_client

        result = ai_explainer_service.get_ai_explanation(
            page_id="test",
            context_type="test",
            metadata={},
            timestamp="2024-01-01T00:00:00Z",
        )

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @patch.object(ai_explainer_service, "load_config")
    @patch.object(ai_explainer_service, "load_system_prompt")
    @patch.object(ai_explainer_service, "GeminiClient")
    @patch.object(ai_explainer_service, "check_rate_limit")
    def test_handles_parse_error(self, mock_rate_limit, MockClient, mock_prompt, mock_config):
        """Should handle parse error gracefully."""
        mock_config.return_value = {"cache": {"enabled": False}, "rate_limits": {}}
        mock_rate_limit.return_value = (True, None)
        mock_prompt.return_value = "System prompt"

        mock_client = MagicMock()
        mock_client.generate.return_value = "Invalid JSON response"
        MockClient.return_value = mock_client

        result = ai_explainer_service.get_ai_explanation(
            page_id="test",
            context_type="test",
            metadata={},
            timestamp="2024-01-01T00:00:00Z",
        )

        assert result["success"] is False
        assert "parse" in result["error"].lower()


# =============================================================================
# Test Mock Explanation
# =============================================================================

class TestGetMockExplanation:
    """Tests for get_mock_explanation function."""

    def test_returns_mock_response(self):
        """Should return mock response with expected structure."""
        result = ai_explainer_service.get_mock_explanation(
            page_id="leaps_ranker",
            context_type="roi_simulator",
            metadata={"symbol": "SPY", "underlying_price": 600},
        )

        assert result["success"] is True
        assert result["pageId"] == "leaps_ranker"
        assert result["contextType"] == "roi_simulator"
        assert "summary" in result["content"]
        assert "key_insights" in result["content"]
        assert "risks" in result["content"]
        assert "watch_items" in result["content"]
        assert "disclaimer" in result["content"]

    def test_uses_metadata_values(self):
        """Should use metadata values in response."""
        result = ai_explainer_service.get_mock_explanation(
            page_id="test",
            context_type="test",
            metadata={"symbol": "AAPL", "underlying_price": 150},
        )

        assert "AAPL" in result["content"]["summary"]


class TestCreditSpreadMock:
    """Tests for credit spread mock explanation."""

    def test_returns_credit_spread_mock_for_spread_simulator(self):
        """Should return credit spread mock for spread_simulator context."""
        result = ai_explainer_service.get_mock_explanation(
            page_id="credit_spread_screener",
            context_type="spread_simulator",
            metadata={
                "symbol": "SPY",
                "spread_type": "PCS",
                "short_strike": 580,
                "long_strike": 575,
                "net_credit": 1.25,
                "underlying_price": 600,
            },
        )

        assert result["success"] is True
        assert result["pageId"] == "credit_spread_screener"
        assert "strategy_name" in result["content"]
        assert "trade_mechanics" in result["content"]
        assert "key_metrics" in result["content"]
        assert "visualization" in result["content"]
        assert "strategy_analysis" in result["content"]
        assert "risk_management" in result["content"]

    def test_credit_spread_mock_has_correct_structure(self):
        """Should have correct credit spread structure."""
        result = ai_explainer_service.get_mock_explanation(
            page_id="credit_spread_screener",
            context_type="spread_simulator",
            metadata={"symbol": "QQQ", "spread_type": "CCS"},
        )

        # Trade mechanics
        tm = result["content"]["trade_mechanics"]
        assert "structure" in tm
        assert "credit_received" in tm
        assert "margin_requirement" in tm
        assert "breakeven" in tm

        # Key metrics
        km = result["content"]["key_metrics"]
        assert "max_profit" in km
        assert "max_loss" in km
        assert "risk_reward_ratio" in km

        # Strategy analysis
        sa = result["content"]["strategy_analysis"]
        assert "bullish_outcome" in sa
        assert "neutral_outcome" in sa
        assert "bearish_outcome" in sa


class TestIronCondorMock:
    """Tests for iron condor mock explanation."""

    def test_returns_iron_condor_mock_for_iron_condor_page(self):
        """Should return iron condor mock for iron_condor_screener page."""
        result = ai_explainer_service.get_mock_explanation(
            page_id="iron_condor_screener",
            context_type="spread_simulator",
            metadata={
                "symbol": "SPY",
                "short_put_strike": 570,
                "long_put_strike": 565,
                "short_call_strike": 630,
                "long_call_strike": 635,
                "net_credit": 2.50,
                "underlying_price": 600,
            },
        )

        assert result["success"] is True
        assert result["pageId"] == "iron_condor_screener"
        assert "strategy_name" in result["content"]
        assert "Iron Condor" in result["content"]["strategy_name"]

    def test_iron_condor_mock_has_correct_structure(self):
        """Should have correct iron condor structure with 4 outcomes."""
        result = ai_explainer_service.get_mock_explanation(
            page_id="iron_condor_screener",
            context_type="spread_simulator",
            metadata={
                "symbol": "QQQ",
                "short_put_strike": 470,
                "long_put_strike": 465,
                "short_call_strike": 530,
                "long_call_strike": 535,
            },
        )

        # Trade mechanics with breakevens (plural)
        tm = result["content"]["trade_mechanics"]
        assert "structure" in tm
        assert "credit_received" in tm
        assert "margin_requirement" in tm
        assert "breakevens" in tm  # Iron Condor has breakevens (plural)

        # Key metrics
        km = result["content"]["key_metrics"]
        assert "max_profit" in km
        assert "max_loss" in km
        assert "risk_reward_ratio" in km

        # Visualization with two loss zones
        viz = result["content"]["visualization"]
        assert "profit_zone" in viz
        assert "lower_loss_zone" in viz  # Iron Condor specific
        assert "upper_loss_zone" in viz  # Iron Condor specific
        assert "transition_zones" in viz

        # Strategy analysis with 4 outcomes
        sa = result["content"]["strategy_analysis"]
        assert "bullish_outcome" in sa
        assert "neutral_outcome" in sa
        assert "bearish_outcome" in sa
        assert "extreme_move_outcome" in sa  # Iron Condor specific

        # Risk management
        rm = result["content"]["risk_management"]
        assert "early_exit_trigger" in rm
        assert "adjustment_options" in rm
        assert "worst_case" in rm

    def test_iron_condor_mock_calculates_metrics_correctly(self):
        """Should calculate iron condor metrics correctly."""
        result = ai_explainer_service.get_mock_explanation(
            page_id="iron_condor_screener",
            context_type="spread_simulator",
            metadata={
                "symbol": "SPY",
                "short_put_strike": 570,
                "long_put_strike": 565,  # 5 point width
                "short_call_strike": 630,
                "long_call_strike": 635,  # 5 point width
                "net_credit": 2.50,
                "underlying_price": 600,
            },
        )

        content = result["content"]

        # Max profit = net_credit * 100 = $250
        assert "$250" in content["key_metrics"]["max_profit"]["value"]

        # Max loss = (5 - 2.50) * 100 = $250
        assert "$250" in content["key_metrics"]["max_loss"]["value"]

        # Profit zone should be between short strikes
        assert "570" in content["visualization"]["profit_zone"]
        assert "630" in content["visualization"]["profit_zone"]

    def test_iron_condor_mock_uses_symbol_from_metadata(self):
        """Should use symbol from metadata."""
        result = ai_explainer_service.get_mock_explanation(
            page_id="iron_condor_screener",
            context_type="spread_simulator",
            metadata={"symbol": "AAPL"},
        )

        assert "AAPL" in result["content"]["summary"]
        assert "Iron Condor on AAPL" in result["content"]["strategy_name"]

    def test_iron_condor_mock_has_key_insights(self):
        """Should have iron condor specific key insights."""
        result = ai_explainer_service.get_mock_explanation(
            page_id="iron_condor_screener",
            context_type="spread_simulator",
            metadata={"symbol": "SPY"},
        )

        insights = result["content"]["key_insights"]
        assert len(insights) >= 3
        titles = [i["title"] for i in insights]
        assert "Profit Zone Width" in titles
        assert "Time Decay Advantage" in titles

    def test_iron_condor_mock_has_risks(self):
        """Should have iron condor specific risks."""
        result = ai_explainer_service.get_mock_explanation(
            page_id="iron_condor_screener",
            context_type="spread_simulator",
            metadata={"symbol": "SPY"},
        )

        risks = result["content"]["risks"]
        assert len(risks) >= 2
        # Check that risks mention both sides
        risk_texts = " ".join([r["risk"] for r in risks])
        assert "assignment" in risk_texts.lower() or "loss" in risk_texts.lower()
