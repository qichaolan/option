"""
AI Explainer Service Layer.

Orchestrates Gemini AI calls to provide explanations for simulation results.
Includes caching, rate limiting, and prompt management.
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Feature flag - can be disabled via environment
AI_EXPLAINER_ENABLED = os.environ.get("AI_EXPLAINER_ENABLED", "true").lower() == "true"

# GCP Project ID for Vertex AI
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")


def _get_config_path() -> Path:
    """Get path to Gemini config file."""
    possible_paths = [
        Path("/app/ai/config/gemini.yaml"),  # Docker
        Path(__file__).parent.parent.parent / "ai" / "config" / "gemini.yaml",  # Local
        Path.cwd() / "ai" / "config" / "gemini.yaml",  # CWD
    ]
    for p in possible_paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"Gemini config not found. Tried: {possible_paths}")


def _get_prompts_dir() -> Path:
    """Get path to prompts directory."""
    possible_paths = [
        Path("/app/ai/prompts"),  # Docker
        Path(__file__).parent.parent.parent / "ai" / "prompts",  # Local
        Path.cwd() / "ai" / "prompts",  # CWD
    ]
    for p in possible_paths:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(f"Prompts directory not found. Tried: {possible_paths}")


def load_config() -> Dict[str, Any]:
    """
    Load Gemini configuration from YAML file.

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
    """
    config_path = _get_config_path()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Loaded Gemini config from {config_path}")
    return config


# =============================================================================
# Prompt Management
# =============================================================================


def load_system_prompt(page_id: str, context_type: str) -> str:
    """
    Load system prompt for given page and context.

    Tries specific prompt first, falls back to default.

    Args:
        page_id: Page identifier (e.g., "leaps_ranker")
        context_type: Context type (e.g., "roi_simulator")

    Returns:
        System prompt content

    Raises:
        FileNotFoundError: If no prompt found
    """
    prompts_dir = _get_prompts_dir()

    # Try specific prompt first: pageId__contextType.md
    specific_prompt = prompts_dir / f"{page_id}__{context_type}.md"
    if specific_prompt.exists():
        with open(specific_prompt, "r") as f:
            prompt = f.read()
        logger.debug(f"Loaded specific prompt: {specific_prompt.name}")
        return prompt

    # Try page-level prompt: pageId.md
    page_prompt = prompts_dir / f"{page_id}.md"
    if page_prompt.exists():
        with open(page_prompt, "r") as f:
            prompt = f.read()
        logger.debug(f"Loaded page prompt: {page_prompt.name}")
        return prompt

    # Fall back to default
    default_prompt = prompts_dir / "default_explainer.md"
    if default_prompt.exists():
        with open(default_prompt, "r") as f:
            prompt = f.read()
        logger.debug("Loaded default prompt")
        return prompt

    raise FileNotFoundError(
        f"No prompt found for page_id={page_id}, context_type={context_type}. "
        f"Checked: {specific_prompt}, {page_prompt}, {default_prompt}"
    )


# =============================================================================
# Caching Layer
# =============================================================================

# In-memory cache for AI responses
_response_cache: Dict[str, Dict[str, Any]] = {}

# Rate limiting tracking: {user_key: {"hourly": [(timestamp, count)], "daily": [(timestamp, count)]}}
_rate_limit_tracker: Dict[str, Dict[str, list]] = {}


def _compute_cache_key(page_id: str, context_type: str, metadata: dict) -> str:
    """
    Compute a stable cache key from request parameters.

    Args:
        page_id: Page identifier
        context_type: Context type
        metadata: Request metadata

    Returns:
        SHA256 hash as cache key
    """
    # Create deterministic JSON representation
    key_data = {
        "page_id": page_id,
        "context_type": context_type,
        "metadata": metadata,
    }
    key_json = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_json.encode()).hexdigest()[:32]


def get_cached_response(cache_key: str, ttl_seconds: int = 86400) -> Optional[Dict[str, Any]]:
    """
    Get cached response if exists and not expired.

    Args:
        cache_key: Cache key
        ttl_seconds: Time-to-live in seconds

    Returns:
        Cached response or None
    """
    if cache_key not in _response_cache:
        return None

    cached = _response_cache[cache_key]
    cached_time = datetime.fromisoformat(cached["cached_at"])
    if datetime.utcnow() - cached_time > timedelta(seconds=ttl_seconds):
        # Expired
        del _response_cache[cache_key]
        return None

    logger.debug(f"Cache hit for key: {cache_key[:8]}...")
    return cached


def set_cached_response(cache_key: str, response: Dict[str, Any]) -> None:
    """
    Store response in cache.

    Args:
        cache_key: Cache key
        response: Response to cache
    """
    _response_cache[cache_key] = {
        **response,
        "cached_at": datetime.utcnow().isoformat(),
    }
    logger.debug(f"Cached response for key: {cache_key[:8]}...")


def clear_cache() -> int:
    """Clear all cached responses. Returns count of cleared items."""
    count = len(_response_cache)
    _response_cache.clear()
    return count


# =============================================================================
# Rate Limiting
# =============================================================================


def _get_user_key(request_info: Optional[Dict[str, Any]] = None) -> str:
    """Get a key identifying the user for rate limiting."""
    if request_info and "client_ip" in request_info:
        return request_info["client_ip"]
    return "anonymous"


def check_rate_limit(
    user_key: str,
    hourly_max: int = 100,
    daily_max: int = 500,
) -> Tuple[bool, Optional[str]]:
    """
    Check if user has exceeded rate limits.

    Args:
        user_key: User identifier
        hourly_max: Maximum requests per hour
        daily_max: Maximum requests per day

    Returns:
        Tuple of (is_allowed, error_message)
    """
    now = datetime.utcnow()
    hour_ago = now - timedelta(hours=1)
    day_ago = now - timedelta(days=1)

    if user_key not in _rate_limit_tracker:
        _rate_limit_tracker[user_key] = {"requests": []}

    # Clean old entries
    _rate_limit_tracker[user_key]["requests"] = [
        ts for ts in _rate_limit_tracker[user_key]["requests"]
        if ts > day_ago
    ]

    requests = _rate_limit_tracker[user_key]["requests"]

    # Check hourly limit
    hourly_count = sum(1 for ts in requests if ts > hour_ago)
    if hourly_count >= hourly_max:
        return False, f"Hourly rate limit exceeded ({hourly_max}/hour). Please try again later."

    # Check daily limit
    daily_count = len(requests)
    if daily_count >= daily_max:
        return False, f"Daily rate limit exceeded ({daily_max}/day). Please try again tomorrow."

    return True, None


def record_request(user_key: str) -> None:
    """Record a request for rate limiting."""
    if user_key not in _rate_limit_tracker:
        _rate_limit_tracker[user_key] = {"requests": []}
    _rate_limit_tracker[user_key]["requests"].append(datetime.utcnow())


# =============================================================================
# Metadata Sanitization
# =============================================================================


def sanitize_metadata(metadata: dict) -> dict:
    """
    Sanitize metadata to remove PII and sensitive data.

    Args:
        metadata: Raw metadata from request

    Returns:
        Sanitized metadata safe to send to Gemini
    """
    # Fields to redact (case-insensitive)
    pii_fields = {
        "email", "username", "user_id", "userid", "password",
        "ssn", "social_security", "phone", "address", "name",
        "account", "account_number", "api_key", "token",
    }

    def _sanitize_dict(d: dict) -> dict:
        result = {}
        for key, value in d.items():
            key_lower = key.lower()
            if any(pii in key_lower for pii in pii_fields):
                result[key] = "[REDACTED]"
            elif isinstance(value, dict):
                result[key] = _sanitize_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    _sanitize_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    return _sanitize_dict(metadata)


# =============================================================================
# Gemini Client
# =============================================================================


class GeminiClient:
    """Client for interacting with GCP Vertex AI Gemini."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Gemini client.

        Args:
            config: Optional configuration override
        """
        self.config = config or load_config()
        self._client = None
        self._model = None

    def _get_model(self):
        """Lazy-load the Gemini model."""
        if self._model is not None:
            return self._model

        try:
            import google.generativeai as genai

            # Configure with API key from environment
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set"
                )

            genai.configure(api_key=api_key)

            model_config = self.config.get("model", {})
            model_name = model_config.get("name", "gemini-2.0-flash-exp")

            self._model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": model_config.get("temperature", 0.3),
                    "max_output_tokens": model_config.get("max_output_tokens", 2048),
                    "top_p": model_config.get("top_p", 0.8),
                    "top_k": model_config.get("top_k", 40),
                },
            )
            logger.info(f"Initialized Gemini model: {model_name}")
            return self._model

        except ImportError:
            logger.error("google-generativeai package not installed")
            raise ImportError(
                "google-generativeai package required. Install with: pip install google-generativeai"
            )

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """
        Generate response from Gemini.

        Args:
            system_prompt: System prompt to set context
            user_content: User content to analyze
            timeout_seconds: Optional timeout override

        Returns:
            Generated response text

        Raises:
            TimeoutError: If request times out
            RuntimeError: If Gemini API fails
        """
        model = self._get_model()
        request_config = self.config.get("request", {})
        timeout = timeout_seconds or request_config.get("timeout_seconds", 30)
        max_retries = request_config.get("max_retries", 2)
        retry_delay = request_config.get("retry_delay_seconds", 1)

        # Combine system prompt and user content
        full_prompt = f"{system_prompt}\n\n---\n\n## User Request\n\n{user_content}"

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()

                # Generate response
                response = model.generate_content(
                    full_prompt,
                    request_options={"timeout": timeout},
                )

                elapsed = time.time() - start_time
                logger.info(f"Gemini response generated in {elapsed:.2f}s")

                if not response.text:
                    raise RuntimeError("Empty response from Gemini")

                return response.text

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                # Check for timeout
                if "timeout" in error_msg or "deadline" in error_msg:
                    if attempt < max_retries:
                        logger.warning(f"Gemini timeout, retrying ({attempt + 1}/{max_retries})")
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise TimeoutError(f"Gemini request timed out after {timeout}s")

                # Check for rate limiting
                if "quota" in error_msg or "rate" in error_msg:
                    logger.error(f"Gemini rate limit: {e}")
                    raise RuntimeError("AI service temporarily unavailable. Please try again later.")

                # Other errors
                if attempt < max_retries:
                    logger.warning(f"Gemini error, retrying ({attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue

                logger.error(f"Gemini API error: {e}")
                raise RuntimeError(f"AI service error: {e}")

        raise RuntimeError(f"Gemini failed after {max_retries + 1} attempts: {last_error}")


# =============================================================================
# Response Parsing
# =============================================================================


def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """
    Parse Gemini response into structured format.

    Args:
        response_text: Raw response from Gemini

    Returns:
        Parsed response dictionary

    Raises:
        ValueError: If response cannot be parsed
    """
    # Try to extract JSON from response
    # Handle potential markdown code blocks
    text = response_text.strip()

    # Remove markdown code block if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        parsed = json.loads(text)
        return parsed
    except json.JSONDecodeError as e:
        # Try to find JSON object in the text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return parsed
            except json.JSONDecodeError:
                pass

        logger.error(f"Failed to parse Gemini response: {e}")
        logger.debug(f"Response text: {text[:500]}...")
        raise ValueError(f"Invalid JSON response from AI: {e}")


# =============================================================================
# Main Service Function
# =============================================================================


def get_ai_explanation(
    page_id: str,
    context_type: str,
    metadata: dict,
    timestamp: str,
    request_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get AI explanation for simulation context.

    This is the main entry point for the AI Explainer service.

    Args:
        page_id: Page identifier (e.g., "leaps_ranker")
        context_type: Context type (e.g., "roi_simulator")
        metadata: Domain-specific data for analysis
        timestamp: Client timestamp
        request_info: Optional request context (for rate limiting)

    Returns:
        Dictionary with explanation content

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If AI service fails
    """
    # Check feature flag
    if not AI_EXPLAINER_ENABLED:
        return {
            "success": False,
            "pageId": page_id,
            "contextType": context_type,
            "content": None,
            "cached": False,
            "error": "AI Explainer is currently disabled",
            "timestamp": datetime.utcnow().isoformat(),
        }

    # Load config
    config = load_config()

    # Check rate limits
    user_key = _get_user_key(request_info)
    rate_limits = config.get("rate_limits", {})
    is_allowed, error_msg = check_rate_limit(
        user_key,
        hourly_max=rate_limits.get("hourly_max", 100),
        daily_max=rate_limits.get("daily_max", 500),
    )
    if not is_allowed:
        return {
            "success": False,
            "pageId": page_id,
            "contextType": context_type,
            "content": None,
            "cached": False,
            "error": error_msg,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # Sanitize metadata
    sanitized_metadata = sanitize_metadata(metadata)

    # Check cache
    cache_config = config.get("cache", {})
    if cache_config.get("enabled", True):
        cache_key = _compute_cache_key(page_id, context_type, sanitized_metadata)
        cached = get_cached_response(
            cache_key,
            ttl_seconds=cache_config.get("ttl_seconds", 86400),
        )
        if cached:
            return {
                "success": True,
                "pageId": page_id,
                "contextType": context_type,
                "content": cached.get("content"),
                "cached": True,
                "cachedAt": cached.get("cached_at"),
                "timestamp": datetime.utcnow().isoformat(),
            }

    # Load system prompt
    try:
        system_prompt = load_system_prompt(page_id, context_type)
    except FileNotFoundError as e:
        logger.error(f"Prompt not found: {e}")
        return {
            "success": False,
            "pageId": page_id,
            "contextType": context_type,
            "content": None,
            "cached": False,
            "error": "AI configuration error. Please contact support.",
            "timestamp": datetime.utcnow().isoformat(),
        }

    # Build user content
    user_content = f"""
## Analysis Context

- **Page**: {page_id}
- **Context Type**: {context_type}
- **Client Timestamp**: {timestamp}

## Simulation Data

```json
{json.dumps(sanitized_metadata, indent=2, default=str)}
```

Please analyze this simulation data and provide your explanation in the required JSON format.
"""

    # Call Gemini
    try:
        client = GeminiClient(config)
        response_text = client.generate(system_prompt, user_content)

        # Parse response
        parsed_response = parse_gemini_response(response_text)

        # Validate required fields
        if "summary" not in parsed_response:
            raise ValueError("Response missing required 'summary' field")

        # Record successful request for rate limiting
        record_request(user_key)

        # Prepare content
        content = {
            "summary": parsed_response.get("summary", ""),
            "key_insights": parsed_response.get("key_insights", []),
            "risks": parsed_response.get("risks", []),
            "watch_items": parsed_response.get("watch_items", []),
            "disclaimer": parsed_response.get(
                "disclaimer",
                "This analysis is for educational purposes only and should not be considered financial advice."
            ),
        }

        # Cache the response
        if cache_config.get("enabled", True):
            set_cached_response(cache_key, {"content": content})

        return {
            "success": True,
            "pageId": page_id,
            "contextType": context_type,
            "content": content,
            "cached": False,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except TimeoutError as e:
        logger.error(f"Gemini timeout: {e}")
        return {
            "success": False,
            "pageId": page_id,
            "contextType": context_type,
            "content": None,
            "cached": False,
            "error": "AI analysis timed out. Please try again.",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except ValueError as e:
        logger.error(f"Response parsing error: {e}")
        return {
            "success": False,
            "pageId": page_id,
            "contextType": context_type,
            "content": None,
            "cached": False,
            "error": "Failed to parse AI response. Please try again.",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.exception(f"AI Explainer error: {e}")
        return {
            "success": False,
            "pageId": page_id,
            "contextType": context_type,
            "content": None,
            "cached": False,
            "error": "AI service temporarily unavailable. Please try again later.",
            "timestamp": datetime.utcnow().isoformat(),
        }


# =============================================================================
# Utility Functions for Testing
# =============================================================================


def get_mock_explanation(
    page_id: str,
    context_type: str,
    metadata: dict,
) -> Dict[str, Any]:
    """
    Get a mock explanation for testing purposes.

    Args:
        page_id: Page identifier
        context_type: Context type
        metadata: Simulation metadata

    Returns:
        Mock explanation response
    """
    # Extract key values from metadata for mock response
    symbol = metadata.get("symbol", "SPY")
    underlying = metadata.get("underlying_price", 600)
    strike = metadata.get("contract", {}).get("strike", 650)
    roi_at_target = metadata.get("roi_results", [{}])[0].get("roi_pct", 150) if metadata.get("roi_results") else 150

    return {
        "success": True,
        "pageId": page_id,
        "contextType": context_type,
        "content": {
            "summary": f"This {symbol} LEAPS call option with a ${strike} strike offers potential returns of {roi_at_target:.0f}% if the underlying reaches the target price. The position requires careful monitoring of time decay and implied volatility changes.",
            "key_insights": [
                {
                    "title": "Breakeven Analysis",
                    "description": f"The option needs {symbol} to rise above ${strike + 50:.2f} by expiration to break even, accounting for the premium paid.",
                    "sentiment": "neutral",
                },
                {
                    "title": "Leverage Profile",
                    "description": "LEAPS provide significant leverage compared to holding shares directly, amplifying both gains and losses.",
                    "sentiment": "positive",
                },
                {
                    "title": "Time Value Consideration",
                    "description": "With over a year until expiration, time decay (theta) is minimal now but will accelerate in the final 3-6 months.",
                    "sentiment": "neutral",
                },
            ],
            "risks": [
                {
                    "risk": "Maximum loss limited to premium paid, but represents 100% of the investment if the option expires worthless.",
                    "severity": "medium",
                },
                {
                    "risk": "Implied volatility crush after earnings or market events could reduce option value even if the underlying moves favorably.",
                    "severity": "medium",
                },
            ],
            "watch_items": [
                {
                    "item": "Underlying price relative to breakeven",
                    "trigger": f"Consider adjustment if {symbol} drops below ${underlying * 0.9:.2f}",
                },
                {
                    "item": "Implied volatility levels",
                    "trigger": "Monitor for significant IV changes around earnings dates",
                },
            ],
            "disclaimer": "This analysis is for educational purposes only and should not be considered financial advice. LEAPS options involve significant risk including potential loss of entire premium. Always do your own research and consult with a qualified financial advisor.",
        },
        "cached": False,
        "timestamp": datetime.utcnow().isoformat(),
    }
