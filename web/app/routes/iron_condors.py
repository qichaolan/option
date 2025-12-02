"""
Iron Condor API Endpoints

Provides endpoints to:
1. List and rank Iron Condor candidates (GET /api/iron-condors)
2. Get payoff/ROI curve for a specific candidate (GET /api/iron-condors/{id}/payoff)

Uses the iron_condor.py engine module for calculations.

Units & Conventions:
- POP (Probability of Profit): 0.0 to 1.0 (0% to 100%)
- ROC (Return on Capital): 0.0 to 1.0+ (e.g., 0.25 = 25%)
- Credit: per-share (e.g., $2.50)
- Payoff/MaxProfit/MaxLoss: per-contract in dollars (100 shares)
- Breakevens: absolute price levels
"""

import datetime
import hashlib
import logging
import math
import re
import sys
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from credit_spread_screener import ScreenerConfig, run_screener
from iron_condor import (
    CreditSpread,
    IronCondor,
    rank_iron_condors,
    payoff_roi_curve,
)

logger = logging.getLogger(__name__)

# Initialize rate limiter (shared with main app)
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/api", tags=["iron_condors"])

# =============================================================================
# CONFIGURATION
# =============================================================================

# Supported tickers for Iron Condors
SUPPORTED_TICKERS = frozenset(["SPY", "QQQ"])

# UUID hex pattern for validation (32 lowercase hex chars)
UUID_HEX_PATTERN = re.compile(r"^[a-f0-9]{32}$")

# Payoff curve limits
MAX_PAYOFF_POINTS = 50  # Cap on number of points in payoff curve

# =============================================================================
# LRU CACHE FOR CONDOR OBJECTS
# =============================================================================

# LRU cache using OrderedDict for efficient eviction
# Key: UUID string, Value: IronCondor object
IRON_CONDOR_CACHE: OrderedDict[str, IronCondor] = OrderedDict()

# Maximum cache size to prevent memory issues
MAX_CONDOR_CACHE_SIZE = 500


def _cache_put(condor_id: str, condor: IronCondor) -> None:
    """Add condor to cache with LRU eviction."""
    # If key exists, move to end (most recently used)
    if condor_id in IRON_CONDOR_CACHE:
        IRON_CONDOR_CACHE.move_to_end(condor_id)
        IRON_CONDOR_CACHE[condor_id] = condor
        return

    # Evict oldest entries if at capacity
    while len(IRON_CONDOR_CACHE) >= MAX_CONDOR_CACHE_SIZE:
        evicted_id, _ = IRON_CONDOR_CACHE.popitem(last=False)
        logger.debug(f"Evicted condor {evicted_id} from cache")

    IRON_CONDOR_CACHE[condor_id] = condor


def _cache_get(condor_id: str) -> Optional[IronCondor]:
    """Get condor from cache, updating access order."""
    if condor_id in IRON_CONDOR_CACHE:
        IRON_CONDOR_CACHE.move_to_end(condor_id)
        return IRON_CONDOR_CACHE[condor_id]
    return None


# =============================================================================
# QUERY RESULT CACHE (Time-based)
# =============================================================================

@dataclass
class CachedQueryResult:
    """Cached result of an Iron Condor query."""
    candidates: list  # List of (condor_id, IronCondorSummary) tuples
    underlying_price: float
    timestamp: float  # Unix timestamp when cached


# Query result cache: key is hash of (symbol, dte_min, dte_max, min_roc, min_pop)
QUERY_CACHE: OrderedDict[str, CachedQueryResult] = OrderedDict()
MAX_QUERY_CACHE_SIZE = 20
QUERY_CACHE_TTL_SECONDS = 60  # Results valid for 60 seconds


def _make_query_cache_key(symbol: str, dte_min: int, dte_max: int,
                          min_roc: float, min_pop: float, limit: int) -> str:
    """Create a hash key for query parameters."""
    key_str = f"{symbol}:{dte_min}:{dte_max}:{min_roc:.4f}:{min_pop:.4f}:{limit}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def _get_cached_query(cache_key: str) -> Optional[CachedQueryResult]:
    """Get cached query result if still valid."""
    if cache_key not in QUERY_CACHE:
        return None

    result = QUERY_CACHE[cache_key]

    # Check if expired
    if time.time() - result.timestamp > QUERY_CACHE_TTL_SECONDS:
        del QUERY_CACHE[cache_key]
        return None

    # Move to end (most recently used)
    QUERY_CACHE.move_to_end(cache_key)
    return result


def _put_cached_query(cache_key: str, result: CachedQueryResult) -> None:
    """Store query result in cache."""
    # Evict oldest if at capacity
    while len(QUERY_CACHE) >= MAX_QUERY_CACHE_SIZE:
        QUERY_CACHE.popitem(last=False)

    QUERY_CACHE[cache_key] = result


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class IronCondorSummary(BaseModel):
    """Summary of an Iron Condor candidate for list endpoint."""

    id: str = Field(..., description="Unique ID for payoff lookup")
    symbol: str = Field(..., description="Underlying symbol")
    expiration: str = Field(..., description="Expiration date (YYYY-MM-DD)")
    dte: int = Field(..., ge=0, description="Days to expiration")
    short_put: float = Field(..., description="Short put strike")
    long_put: float = Field(..., description="Long put strike")
    short_call: float = Field(..., description="Short call strike")
    long_call: float = Field(..., description="Long call strike")
    total_credit: float = Field(..., ge=0, description="Total credit per share ($)")
    max_profit: float = Field(..., ge=0, description="Max profit per contract ($)")
    max_loss: float = Field(..., ge=0, description="Max loss per contract ($)")
    risk_reward_ratio: float = Field(
        ..., ge=0, description="Risk/reward ratio (max_profit / max_loss)"
    )
    combined_pop: float = Field(
        ..., ge=0, le=1, description="Combined probability of profit (0-1)"
    )
    combined_score: float = Field(
        ..., ge=0, le=1, description="AI score for ranking (0-1)"
    )
    breakeven_low: float = Field(..., description="Lower breakeven price")
    breakeven_high: float = Field(..., description="Upper breakeven price")


class IronCondorListResponse(BaseModel):
    """Response for list Iron Condors endpoint."""

    symbol: str
    underlying_price: float
    total_candidates: int
    candidates: list[IronCondorSummary]
    timestamp: str
    cached: bool = Field(default=False, description="Whether result was from cache")


class PayoffPoint(BaseModel):
    """Single point on payoff/ROI curve."""

    move_pct: float = Field(..., description="Price move percentage (e.g., -0.05 = -5%)")
    price: float = Field(..., description="Underlying price at expiration")
    payoff: float = Field(..., description="Payoff in $ per contract (100 shares)")
    roi: float = Field(..., description="ROI as decimal (payoff / max_loss)")


class IronCondorPayoffResponse(BaseModel):
    """Response for payoff endpoint."""

    condor_id: str
    symbol: str
    current_price: float
    expiration: str
    dte: int
    short_put: float
    long_put: float
    short_call: float
    long_call: float
    total_credit: float = Field(..., description="Credit per share ($)")
    max_profit: float = Field(..., description="Max profit per contract ($)")
    max_loss: float = Field(..., description="Max loss per contract ($)")
    breakeven_low: float
    breakeven_high: float
    risk_reward_ratio: float
    points: list[PayoffPoint]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _compute_risk_reward_ratio(max_profit: float, max_loss: float) -> float:
    """
    Compute risk/reward ratio.

    Formula: max_profit / max_loss

    For Iron Condors, max_profit = total_credit * 100,
    so this equals ROC (Return on Capital).

    Returns 0 if max_loss <= 0 (edge case: free trade or invalid data).
    """
    if max_loss <= 0:
        return 0.0
    return round(max_profit / max_loss, 4)


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert value to float, handling NaN/inf."""
    if val is None:
        return default
    try:
        f = float(val)
        if not np.isfinite(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(val, default: int = 0) -> int:
    """Safely convert value to int, handling NaN/inf."""
    if val is None:
        return default
    try:
        f = float(val)
        if not np.isfinite(f):
            return default
        return int(f)
    except (ValueError, TypeError):
        return default


def _condor_to_summary(condor: IronCondor, condor_id: str) -> IronCondorSummary:
    """Convert IronCondor object to API summary model."""
    return IronCondorSummary(
        id=condor_id,
        symbol=condor.underlying,
        expiration=condor.expiration,
        dte=condor.days_to_expiration,
        short_put=condor.short_put_strike,
        long_put=condor.long_put_strike,
        short_call=condor.short_call_strike,
        long_call=condor.long_call_strike,
        total_credit=round(condor.total_credit, 2),
        max_profit=round(condor.max_profit_dollars, 2),
        max_loss=round(condor.max_loss_dollars, 2),
        risk_reward_ratio=_compute_risk_reward_ratio(
            condor.max_profit_dollars, condor.max_loss_dollars
        ),
        combined_pop=round(condor.pop, 4),
        combined_score=round(condor.total_score, 4),
        breakeven_low=round(condor.breakeven_low, 2),
        breakeven_high=round(condor.breakeven_high, 2),
    )


def _row_to_credit_spread(row: dict, symbol: str) -> CreditSpread:
    """
    Convert a DataFrame row (as dict) to CreditSpread object.

    Args:
        row: Dictionary with keys from credit spread DataFrame
        symbol: Underlying symbol

    Returns:
        CreditSpread object for Iron Condor construction
    """
    return CreditSpread(
        underlying=symbol,
        expiration=row["expiration"],
        spread_type=row["type"],  # Column is "type" not "spread_type"
        short_strike=_safe_float(row.get("short_strike"), 0.0),
        long_strike=_safe_float(row.get("long_strike"), 0.0),
        credit=_safe_float(row.get("credit"), 0.0),
        short_delta=abs(_safe_float(row.get("short_delta"), 0.15)),
        bid_ask_spread=_safe_float(row.get("bid_ask_spread"), 0.10),
        volume=_safe_int(row.get("volume"), 100),
        open_interest=_safe_int(row.get("open_interest"), 500),
    )


def _get_underlying_price(df, symbol: str) -> float:
    """
    Get underlying price from DataFrame or fetch from yfinance.

    Args:
        df: Credit spread DataFrame
        symbol: Underlying symbol

    Returns:
        Current underlying price, or 0.0 if unavailable
    """
    # Try to get from screener data first (faster)
    if "underlying_price" in df.columns and not df["underlying_price"].isna().all():
        price = df["underlying_price"].iloc[0]
        safe_price = _safe_float(price, 0.0)
        if safe_price > 0:
            return safe_price

    # Fall back to yfinance
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.get("lastPrice", 0.0)
        if not price:
            price = ticker.info.get("regularMarketPrice", 0.0)
        return float(price) if price else 0.0
    except Exception as e:
        logger.warning(f"Could not fetch underlying price for {symbol}: {e}")
        return 0.0


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("/iron-condors", response_model=IronCondorListResponse)
@limiter.limit("15/minute")
async def list_iron_condors(
    request: Request,
    symbol: str = Query(default="QQQ", description="Underlying symbol (SPY, QQQ)"),
    dte_min: int = Query(default=14, ge=1, le=180, description="Minimum DTE"),
    dte_max: int = Query(default=45, ge=1, le=180, description="Maximum DTE"),
    min_roc: float = Query(
        default=0.15, ge=0.0, le=1.0, description="Minimum ROC (0.15 = 15%)"
    ),
    min_pop: float = Query(
        default=0.50, ge=0.0, le=1.0, description="Minimum POP (0.50 = 50%)"
    ),
    limit: int = Query(default=20, ge=1, le=50, description="Max candidates to return"),
):
    """
    List and rank Iron Condor candidates.

    Fetches credit spreads using the existing screener, then combines them
    into Iron Condors and ranks by total_score.

    Returns:
        List of Iron Condor candidates sorted by combined_score descending.
    """
    # Validate symbol
    symbol = symbol.strip().upper()
    if symbol not in SUPPORTED_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Symbol must be one of: {', '.join(sorted(SUPPORTED_TICKERS))}",
        )

    # Validate DTE range
    if dte_min > dte_max:
        raise HTTPException(
            status_code=400,
            detail="dte_min must be less than or equal to dte_max",
        )

    # Check query cache first
    cache_key = _make_query_cache_key(symbol, dte_min, dte_max, min_roc, min_pop, limit)
    cached_result = _get_cached_query(cache_key)

    if cached_result:
        logger.debug(f"Returning cached result for {symbol}")
        # Rebuild response from cache (condors are still in IRON_CONDOR_CACHE)
        return IronCondorListResponse(
            symbol=symbol,
            underlying_price=cached_result.underlying_price,
            total_candidates=len(cached_result.candidates),
            candidates=[summary for _, summary in cached_result.candidates],
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            cached=True,
        )

    try:
        # Use credit spread screener to fetch PCS and CCS
        # For Iron Condors, use symmetrical delta range (0.10–0.35)
        # This selects OTM options with ~10-35% delta on both sides
        config = ScreenerConfig(
            tickers=[symbol],
            min_dte=dte_min,
            max_dte=dte_max,
            min_delta=0.10,  # Short leg delta floor (both puts and calls)
            max_delta=0.35,  # Short leg delta ceiling (symmetrical for IC)
            max_width=10,
            min_roc=min_roc,
        )

        logger.info(f"Fetching credit spreads for {symbol} (DTE: {dte_min}-{dte_max})")
        df = run_screener(config)

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if df is None or df.empty:
            return IronCondorListResponse(
                symbol=symbol,
                underlying_price=0.0,
                total_candidates=0,
                candidates=[],
                timestamp=now_iso,
            )

        # Get underlying price
        underlying_price = _get_underlying_price(df, symbol)

        # Separate PCS and CCS
        pcs_df = df[df["type"] == "PCS"]
        ccs_df = df[df["type"] == "CCS"]

        if pcs_df.empty or ccs_df.empty:
            logger.info(f"Not enough spreads for {symbol}: PCS={len(pcs_df)}, CCS={len(ccs_df)}")
            return IronCondorListResponse(
                symbol=symbol,
                underlying_price=underlying_price,
                total_candidates=0,
                candidates=[],
                timestamp=now_iso,
            )

        # Convert DataFrames to list of dicts for faster iteration
        pcs_records = pcs_df.to_dict("records")
        ccs_records = ccs_df.to_dict("records")

        # Group spreads by expiration to build valid Iron Condors
        pcs_by_exp: dict[str, list[dict]] = {}
        ccs_by_exp: dict[str, list[dict]] = {}
        dte_by_exp: dict[str, int] = {}

        for row in pcs_records:
            exp = row["expiration"]
            pcs_by_exp.setdefault(exp, []).append(row)
            if exp not in dte_by_exp:
                dte_by_exp[exp] = int(row.get("dte", 30))

        for row in ccs_records:
            exp = row["expiration"]
            ccs_by_exp.setdefault(exp, []).append(row)

        # Build Iron Condors for each expiration that has both PCS and CCS
        all_candidates: list[IronCondor] = []

        for expiration in pcs_by_exp:
            if expiration not in ccs_by_exp:
                continue

            exp_pcs_rows = pcs_by_exp[expiration]
            exp_ccs_rows = ccs_by_exp[expiration]
            dte = dte_by_exp.get(expiration, 30)

            # Convert to CreditSpread objects
            put_spreads = [_row_to_credit_spread(row, symbol) for row in exp_pcs_rows]
            call_spreads = [_row_to_credit_spread(row, symbol) for row in exp_ccs_rows]

            logger.debug(
                f"Building Iron Condors for {expiration}: "
                f"{len(put_spreads)} PCS x {len(call_spreads)} CCS"
            )

            # Build and rank Iron Condors for this expiration
            # Request more than limit to allow for cross-expiration ranking
            exp_condors = rank_iron_condors(
                put_spreads=put_spreads,
                call_spreads=call_spreads,
                underlying_price=underlying_price,
                days_to_expiration=dte,
                top_n=limit * 2,  # Get extra for cross-expiration ranking
            )

            all_candidates.extend(exp_condors)

        # Sort all candidates by score and filter by min_pop
        all_candidates.sort(key=lambda c: c.total_score, reverse=True)
        filtered_condors = [c for c in all_candidates if c.pop >= min_pop]

        # Take top N after filtering
        final_condors = filtered_condors[:limit]

        # Store in LRU cache and build response
        candidates_for_cache: list[tuple[str, IronCondorSummary]] = []
        response_candidates: list[IronCondorSummary] = []

        for condor in final_condors:
            condor_id = uuid.uuid4().hex
            _cache_put(condor_id, condor)
            summary = _condor_to_summary(condor, condor_id)
            candidates_for_cache.append((condor_id, summary))
            response_candidates.append(summary)

        # Cache the query result
        _put_cached_query(
            cache_key,
            CachedQueryResult(
                candidates=candidates_for_cache,
                underlying_price=underlying_price,
                timestamp=time.time(),
            )
        )

        logger.info(f"Returning {len(response_candidates)} Iron Condor candidates for {symbol}")

        return IronCondorListResponse(
            symbol=symbol,
            underlying_price=underlying_price,
            total_candidates=len(response_candidates),
            candidates=response_candidates,
            timestamp=now_iso,
        )

    except ValueError as e:
        logger.warning(f"Validation error for {symbol}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error fetching Iron Condors for {symbol}")
        raise HTTPException(
            status_code=500,
            detail="Unable to fetch Iron Condor data. Please try again later.",
        )


@router.get("/iron-condors/{condor_id}/payoff", response_model=IronCondorPayoffResponse)
@limiter.limit("30/minute")
async def get_iron_condor_payoff(
    request: Request,
    condor_id: str,
    move_low_pct: float = Query(
        default=-0.08, ge=-0.30, le=0.0, description="Lower move bound (e.g., -0.08 = -8%)"
    ),
    move_high_pct: float = Query(
        default=0.08, ge=0.0, le=0.30, description="Upper move bound (e.g., 0.08 = +8%)"
    ),
    step_pct: float = Query(
        default=0.01, ge=0.005, le=0.05, description="Step size (e.g., 0.01 = 1%)"
    ),
):
    """
    Get payoff/ROI curve for a specific Iron Condor.

    Looks up the condor from cache by ID and computes payoff at expiration
    for the specified price move range.

    Payoff formula per contract (100 shares):
        PCS payoff = credit_PCS*100 - max(0, K_short_put - S_T)*100 + max(0, K_long_put - S_T)*100
        CCS payoff = credit_CCS*100 - max(0, S_T - K_short_call)*100 + max(0, S_T - K_long_call)*100
        Total = PCS payoff + CCS payoff

    ROI = payoff / max_loss_dollars

    Returns:
        Payoff curve with summary metrics.
    """
    # Validate condor_id format (should be 32 lowercase hex chars)
    if not UUID_HEX_PATTERN.match(condor_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid condor ID format.",
        )

    # Validate move range
    if move_low_pct >= move_high_pct:
        raise HTTPException(
            status_code=400,
            detail="move_low_pct must be less than move_high_pct.",
        )

    # Calculate expected number of points and cap if necessary
    expected_points = int((move_high_pct - move_low_pct) / step_pct) + 1
    if expected_points > MAX_PAYOFF_POINTS:
        # Adjust step_pct to limit points
        step_pct = (move_high_pct - move_low_pct) / (MAX_PAYOFF_POINTS - 1)
        logger.debug(f"Adjusted step_pct to {step_pct:.4f} to cap points at {MAX_PAYOFF_POINTS}")

    # Lookup condor from LRU cache
    condor = _cache_get(condor_id)

    if condor is None:
        raise HTTPException(
            status_code=404,
            detail="Iron Condor not found. The scan results may have expired. Please run a new scan.",
        )

    # Generate payoff curve using the engine module
    # This computes payoff at expiration for each price scenario
    curve = payoff_roi_curve(
        condor=condor,
        move_low_pct=move_low_pct,
        move_high_pct=move_high_pct,
        step_pct=step_pct,
    )

    # Convert to Pydantic models
    points = [
        PayoffPoint(
            move_pct=round(p["move_pct"], 4),
            price=round(p["price"], 2),
            payoff=round(p["payoff"], 2),
            roi=round(p["roi"], 4),
        )
        for p in curve
    ]

    return IronCondorPayoffResponse(
        condor_id=condor_id,
        symbol=condor.underlying,
        current_price=round(condor.underlying_price, 2),
        expiration=condor.expiration,
        dte=condor.days_to_expiration,
        short_put=condor.short_put_strike,
        long_put=condor.long_put_strike,
        short_call=condor.short_call_strike,
        long_call=condor.long_call_strike,
        total_credit=round(condor.total_credit, 2),
        max_profit=round(condor.max_profit_dollars, 2),
        max_loss=round(condor.max_loss_dollars, 2),
        breakeven_low=round(condor.breakeven_low, 2),
        breakeven_high=round(condor.breakeven_high, 2),
        risk_reward_ratio=_compute_risk_reward_ratio(
            condor.max_profit_dollars, condor.max_loss_dollars
        ),
        points=points,
    )
