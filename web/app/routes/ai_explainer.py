"""AI Explainer API routes."""

import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models import (
    AiExplainerRequest,
    AiExplainerResponse,
    AiExplainerContent,
)
from app.services import ai_explainer_service

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/api", tags=["ai-explainer"])

# Feature flag check
AI_EXPLAINER_ENABLED = os.environ.get("AI_EXPLAINER_ENABLED", "true").lower() == "true"

# Use mock responses for testing (set via environment)
USE_MOCK_RESPONSES = os.environ.get("AI_EXPLAINER_USE_MOCK", "false").lower() == "true"


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    # Check X-Forwarded-For header (for proxies like Cloud Run)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()

    # Fall back to direct client IP
    if request.client:
        return request.client.host

    return "unknown"


@router.post("/ai-explainer", response_model=AiExplainerResponse)
@limiter.limit("30/minute")
async def get_ai_explanation(request: Request, explainer_request: AiExplainerRequest):
    """
    Get AI-powered explanation for simulation results.

    This endpoint accepts simulation context and returns a structured
    AI-generated explanation with key insights, risks, and watch items.

    Args:
        request: FastAPI request object
        explainer_request: AI Explainer request with pageId, contextType, and metadata

    Returns:
        AiExplainerResponse with structured explanation content

    Raises:
        HTTPException: On validation errors or service failures
    """
    # Check feature flag
    if not AI_EXPLAINER_ENABLED:
        return AiExplainerResponse(
            success=False,
            pageId=explainer_request.pageId,
            contextType=explainer_request.contextType,
            content=None,
            cached=False,
            error="AI Explainer is currently disabled",
            timestamp=datetime.utcnow().isoformat(),
        )

    try:
        # Build request info for rate limiting
        request_info = {
            "client_ip": _get_client_ip(request),
        }

        # Use mock response if configured (for testing)
        if USE_MOCK_RESPONSES:
            logger.info("Using mock AI response (AI_EXPLAINER_USE_MOCK=true)")
            result = ai_explainer_service.get_mock_explanation(
                page_id=explainer_request.pageId,
                context_type=explainer_request.contextType,
                metadata=explainer_request.metadata,
            )
        else:
            # Call the AI Explainer service
            result = ai_explainer_service.get_ai_explanation(
                page_id=explainer_request.pageId,
                context_type=explainer_request.contextType,
                metadata=explainer_request.metadata,
                timestamp=explainer_request.timestamp,
                request_info=request_info,
            )

        # Convert to response model
        content = None
        if result.get("content"):
            content = AiExplainerContent(**result["content"])

        return AiExplainerResponse(
            success=result.get("success", False),
            pageId=result.get("pageId", explainer_request.pageId),
            contextType=result.get("contextType", explainer_request.contextType),
            content=content,
            cached=result.get("cached", False),
            cachedAt=result.get("cachedAt"),
            error=result.get("error"),
            timestamp=result.get("timestamp", datetime.utcnow().isoformat()),
        )

    except ValidationError as e:
        logger.warning(f"Validation error in AI Explainer: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid request format. Please check your input.",
        )

    except Exception as e:
        # Log the error but don't expose details to client
        logger.exception(f"AI Explainer error: {e}")
        raise HTTPException(
            status_code=500,
            detail="AI service temporarily unavailable. Please try again later.",
        )


@router.get("/ai-explainer/health")
async def ai_explainer_health():
    """
    Health check endpoint for AI Explainer service.

    Returns status of the AI Explainer feature and configuration.
    """
    try:
        # Try to load config
        config = ai_explainer_service.load_config()

        return {
            "status": "healthy" if AI_EXPLAINER_ENABLED else "disabled",
            "feature_enabled": AI_EXPLAINER_ENABLED,
            "mock_mode": USE_MOCK_RESPONSES,
            "model": config.get("model", {}).get("name", "unknown"),
            "cache_enabled": config.get("cache", {}).get("enabled", True),
            "rate_limits": {
                "hourly": config.get("rate_limits", {}).get("hourly_max", 100),
                "daily": config.get("rate_limits", {}).get("daily_max", 500),
            },
        }

    except Exception as e:
        logger.warning(f"AI Explainer health check warning: {e}")
        return {
            "status": "degraded",
            "feature_enabled": AI_EXPLAINER_ENABLED,
            "mock_mode": USE_MOCK_RESPONSES,
            "error": str(e),
        }


@router.delete("/ai-explainer/cache")
@limiter.limit("5/minute")
async def clear_ai_cache(request: Request):
    """
    Clear the AI Explainer response cache.

    This is an administrative endpoint for cache management.
    Rate limited to prevent abuse.
    """
    try:
        count = ai_explainer_service.clear_cache()
        return {
            "success": True,
            "message": f"Cleared {count} cached responses",
        }
    except Exception as e:
        logger.error(f"Error clearing AI cache: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to clear cache",
        )
