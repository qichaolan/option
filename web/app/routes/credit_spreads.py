"""Credit Spreads Screener API routes."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

logger = logging.getLogger(__name__)

# Add parent directory to path to import credit_spread_screener
current_dir = Path(__file__).parent.parent.parent  # web/app/routes -> web
repo_root = current_dir.parent  # web -> repo root (for local dev)

for path in [repo_root, Path("/app"), Path.cwd()]:
    if (path / "credit_spread_screener.py").exists():
        sys.path.insert(0, str(path))
        break

from credit_spread_screener import ScreenerConfig, run_screener

from app.models import (
    CreditSpreadRequest,
    CreditSpreadResponse,
    CreditSpreadResult,
)

router = APIRouter(prefix="/api", tags=["credit_spreads"])

# Supported tickers for credit spreads
SUPPORTED_TICKERS = ["SPY", "QQQ"]


@router.get("/credit-spreads/tickers")
@limiter.limit("30/minute")
async def get_credit_spread_tickers(request: Request):
    """Get list of supported tickers for credit spreads."""
    return [{"symbol": t, "name": t} for t in SUPPORTED_TICKERS]


@router.post("/credit-spreads", response_model=CreditSpreadResponse)
@limiter.limit("5/minute")
async def screen_credit_spreads(request: Request, spread_request: CreditSpreadRequest):
    """
    Screen for credit spread opportunities.

    Args:
        spread_request: CreditSpreadRequest with screening parameters

    Returns:
        CreditSpreadResponse with ranked spread candidates
    """
    symbol = spread_request.symbol.upper()

    if symbol not in SUPPORTED_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported ticker: {symbol}. Supported: {SUPPORTED_TICKERS}",
        )

    # Validate DTE range
    if spread_request.min_dte >= spread_request.max_dte:
        raise HTTPException(
            status_code=400,
            detail="min_dte must be less than max_dte",
        )

    # Validate delta range
    if spread_request.min_delta >= spread_request.max_delta:
        raise HTTPException(
            status_code=400,
            detail="min_delta must be less than max_delta",
        )

    try:
        # Create screener config
        config = ScreenerConfig(
            tickers=[symbol],
            min_dte=spread_request.min_dte,
            max_dte=spread_request.max_dte,
            min_delta=spread_request.min_delta,
            max_delta=spread_request.max_delta,
            max_width=spread_request.max_width,
            min_roc=spread_request.min_roc,
            min_ivp=0,  # Don't filter by IVP in API, let frontend handle display
        )

        # Run screener
        df = run_screener(config)

        if df.empty:
            return CreditSpreadResponse(
                symbol=symbol,
                underlying_price=0.0,
                ivp=0.0,
                spread_type_filter=spread_request.spread_type or "ALL",
                total_pcs=0,
                total_ccs=0,
                spreads=[],
                timestamp=datetime.utcnow().isoformat(),
            )

        # Filter by spread type if specified
        spread_type_filter = spread_request.spread_type or "ALL"
        if spread_type_filter != "ALL":
            df = df[df["type"] == spread_type_filter]

        # Get underlying price and IVP from first result
        underlying_price = float(df["underlying_price"].iloc[0]) if len(df) > 0 else 0.0
        ivp = float(df["ivp"].iloc[0]) if len(df) > 0 else 0.0

        # Count by type
        total_pcs = len(df[df["type"] == "PCS"])
        total_ccs = len(df[df["type"] == "CCS"])

        # Convert to response model using vectorized to_dict (much faster than iterrows)
        # Rename 'type' column to 'spread_type' for the model
        df_out = df.rename(columns={"type": "spread_type"})

        # Select and order columns to match CreditSpreadResult
        result_columns = [
            "symbol", "spread_type", "expiration", "dte", "short_strike",
            "long_strike", "width", "credit", "max_loss", "roc", "short_delta",
            "delta_estimated", "prob_profit", "iv", "ivp", "underlying_price",
            "break_even", "break_even_distance_pct", "liquidity_score",
            "slippage_score", "total_score"
        ]
        available_cols = [c for c in result_columns if c in df_out.columns]
        df_out = df_out[available_cols]

        # Convert to list of dicts and then to Pydantic models
        records = df_out.to_dict(orient="records")
        spreads = [CreditSpreadResult(**record) for record in records]

        return CreditSpreadResponse(
            symbol=symbol,
            underlying_price=round(underlying_price, 2),
            ivp=round(ivp, 1),
            spread_type_filter=spread_type_filter,
            total_pcs=total_pcs,
            total_ccs=total_ccs,
            spreads=spreads,
            timestamp=datetime.utcnow().isoformat(),
        )

    except ValueError as e:
        logger.warning(f"Validation error for {symbol}: {e}")
        # Only expose safe validation messages
        error_msg = str(e)
        if "No options" in error_msg or "No spreads" in error_msg:
            raise HTTPException(status_code=400, detail="No options data available for this symbol and date range.")
        raise HTTPException(status_code=400, detail="Invalid request parameters. Please check your input.")
    except Exception as e:
        # Log full exception but return generic message to client
        logger.exception(f"Unexpected error screening credit spreads for {symbol}")
        raise HTTPException(
            status_code=500,
            detail="Unable to fetch options data. Please try again later.",
        )
