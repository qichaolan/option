"""
FastAPI routes for AI Score API.

Provides endpoints for retrieving AI-generated stock scores.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.services.ai_score_service import get_ai_score

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

# Thread pool for running sync operations
_executor = ThreadPoolExecutor(max_workers=4)

router = APIRouter(prefix="/api", tags=["ai-score"])


class AIScoreResponse(BaseModel):
    """Response model for AI score endpoint."""

    symbol: str = Field(..., description="Stock ticker symbol")
    date: str = Field(..., description="Score date in YYYY-MM-DD format")
    score_raw: float = Field(..., description="Raw signal value from strategy")
    score_0_1: float = Field(..., ge=0.0, le=1.0, description="Normalized score between 0 and 1")
    ai_rating: str = Field(..., description="AI rating: Strong Buy, Buy, Hold, Sell, or Must Sell")


class AIScoreErrorResponse(BaseModel):
    """Error response model."""

    detail: str


@router.get(
    "/ai-score",
    response_model=AIScoreResponse,
    responses={
        400: {"model": AIScoreErrorResponse, "description": "Invalid request"},
        404: {"model": AIScoreErrorResponse, "description": "Symbol not found or no data"},
        429: {"model": AIScoreErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": AIScoreErrorResponse, "description": "Server error"},
    },
    summary="Get AI Score for a symbol",
    description="""
    Returns the AI-generated score for a stock symbol.

    The score includes:
    - **score_raw**: Raw signal value from the strategy
    - **score_0_1**: Normalized score between 0 and 1
    - **ai_rating**: Human-readable rating (Strong Buy, Buy, Hold, Sell, Must Sell)

    The score is cached in GCS and refreshed when stale.
    """,
)
@limiter.limit("30/minute")
async def get_ai_score_endpoint(
    request: Request,
    symbol: str = Query(
        ...,
        description="Stock ticker symbol (e.g., SPY, AAPL)",
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z]+$",
    ),
    refresh: bool = Query(
        False,
        description="Force refresh the score from source data",
    ),
) -> AIScoreResponse:
    """
    Get AI score for a symbol.

    Args:
        request: FastAPI request object (for rate limiting)
        symbol: Stock ticker symbol
        refresh: If True, force refresh from source data

    Returns:
        AI score data

    Raises:
        HTTPException: On error
    """
    logger.info(f"AI score request: symbol={symbol}, refresh={refresh}")

    try:
        # Run sync function in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: get_ai_score(symbol, force_refresh=refresh)
        )
        return AIScoreResponse(**result)

    except ValueError as e:
        error_msg = str(e)
        logger.warning(f"ValueError for {symbol}: {error_msg}")
        if "not found" in error_msg.lower() or "no data" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg)
        if "invalid" in error_msg.lower():
            raise HTTPException(status_code=400, detail=error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    except Exception as e:
        logger.exception(f"Unexpected error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
