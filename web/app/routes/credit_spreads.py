"""Credit Spreads Screener API routes."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
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
    CreditSpreadSimulatorRequest,
    CreditSpreadSimulatorResponse,
    CreditSpreadSimulatorSummary,
    CreditSpreadSimulatorPoint,
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

        # Get underlying price and IVP from first result (before filtering)
        # Use safe conversion to handle potential NaN values
        underlying_price = 0.0
        ivp = 0.0
        if len(df) > 0:
            raw_price = df["underlying_price"].iloc[0]
            raw_ivp = df["ivp"].iloc[0]
            if raw_price is not None and np.isfinite(raw_price):
                underlying_price = float(raw_price)
            if raw_ivp is not None and np.isfinite(raw_ivp):
                ivp = float(raw_ivp)

        # Count by type BEFORE filtering (so frontend can show available counts)
        total_pcs = len(df[df["type"] == "PCS"])
        total_ccs = len(df[df["type"] == "CCS"])

        # Filter by spread type if specified
        spread_type_filter = spread_request.spread_type or "ALL"
        logger.info(f"Spread type filter: {spread_type_filter}, total spreads: {len(df)} (PCS={total_pcs}, CCS={total_ccs})")
        if spread_type_filter != "ALL":
            df = df[df["type"] == spread_type_filter]
            logger.info(f"After type filter: {len(df)} spreads remaining")

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

        # Handle NaN/inf in required float fields to prevent Pydantic validation errors
        required_float_fields = [
            "short_strike", "long_strike", "width", "credit", "max_loss", "roc",
            "short_delta", "prob_profit", "iv", "ivp", "underlying_price",
            "break_even", "break_even_distance_pct", "liquidity_score",
            "slippage_score", "total_score"
        ]
        for col in required_float_fields:
            if col in df_out.columns:
                # Use numpy for reliable NaN/inf handling
                df_out[col] = df_out[col].replace([np.inf, -np.inf], np.nan)
                df_out[col] = df_out[col].fillna(0.0)
                # Double-check: convert any remaining non-finite values
                df_out[col] = df_out[col].apply(lambda x: 0.0 if not np.isfinite(x) else x)

        # Convert to list of dicts and then to Pydantic models
        records = df_out.to_dict(orient="records")
        logger.info(f"Converting {len(records)} spreads to response model")
        spreads = []
        for record in records:
            try:
                spreads.append(CreditSpreadResult(**record))
            except Exception as e:
                # Log the problematic record and skip it
                logger.warning(f"Skipping invalid spread record: {e}")
                logger.debug(f"Record data: {record}")

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


@router.post("/credit-spreads/simulate", response_model=CreditSpreadSimulatorResponse)
@limiter.limit("30/minute")
async def simulate_credit_spread(request: Request, sim_request: CreditSpreadSimulatorRequest):
    """
    Simulate P/L at expiration for a credit spread across different price moves.

    Calculates P/L for every 1% move between -5% and +5% from current price.

    For PCS (Put Credit Spread - Bullish):
    - Max gain = net_credit * 100
    - Max loss = (short_strike - long_strike - net_credit) * 100
    - If S_T >= short_strike: P/L = +max_gain
    - If long_strike < S_T < short_strike: P/L = net_credit*100 - (short_strike - S_T)*100
    - If S_T <= long_strike: P/L = -max_loss

    For CCS (Call Credit Spread - Bearish):
    - Max gain = net_credit * 100
    - Max loss = (long_strike - short_strike - net_credit) * 100
    - If S_T <= short_strike: P/L = +max_gain
    - If short_strike < S_T < long_strike: P/L = net_credit*100 - (S_T - short_strike)*100
    - If S_T >= long_strike: P/L = -max_loss
    """
    symbol = sim_request.symbol.upper()
    spread_type = sim_request.spread_type
    short_strike = sim_request.short_strike
    long_strike = sim_request.long_strike
    net_credit = sim_request.net_credit
    underlying_now = sim_request.underlying_price_now

    # Validate strike relationship based on spread type
    if spread_type == "PCS":
        # For put credit spread: short_strike > long_strike
        if short_strike <= long_strike:
            raise HTTPException(
                status_code=400,
                detail="For PCS, short_strike must be greater than long_strike",
            )
        width = short_strike - long_strike
        max_gain = net_credit * 100
        max_loss = (width - net_credit) * 100
        breakeven_price = short_strike - net_credit
    else:  # CCS
        # For call credit spread: short_strike < long_strike
        if short_strike >= long_strike:
            raise HTTPException(
                status_code=400,
                detail="For CCS, short_strike must be less than long_strike",
            )
        width = long_strike - short_strike
        max_gain = net_credit * 100
        max_loss = (width - net_credit) * 100
        breakeven_price = short_strike + net_credit

    # Calculate breakeven as percentage move from current price
    breakeven_pct = ((breakeven_price - underlying_now) / underlying_now) * 100

    # Generate simulation points for -5% to +5% in 1% increments
    points = []
    for pct_move in range(-5, 6):  # -5, -4, ..., 0, ..., 4, 5
        price_at_expiry = underlying_now * (1 + pct_move / 100.0)

        # Calculate P/L based on spread type
        if spread_type == "PCS":
            if price_at_expiry >= short_strike:
                pl = max_gain
            elif price_at_expiry <= long_strike:
                pl = -max_loss
            else:
                # Between strikes: partial loss
                pl = net_credit * 100 - (short_strike - price_at_expiry) * 100
        else:  # CCS
            if price_at_expiry <= short_strike:
                pl = max_gain
            elif price_at_expiry >= long_strike:
                pl = -max_loss
            else:
                # Between strikes: partial loss
                pl = net_credit * 100 - (price_at_expiry - short_strike) * 100

        points.append(CreditSpreadSimulatorPoint(
            pct_move=float(pct_move),
            underlying_price=round(price_at_expiry, 2),
            pl_per_spread=round(pl, 2),
        ))

    summary = CreditSpreadSimulatorSummary(
        max_gain=round(max_gain, 2),
        max_loss=round(max_loss, 2),
        breakeven_price=round(breakeven_price, 2),
        breakeven_pct=round(breakeven_pct, 2),
    )

    return CreditSpreadSimulatorResponse(
        symbol=symbol,
        spread_type=spread_type,
        expiration=sim_request.expiration,
        short_strike=short_strike,
        long_strike=long_strike,
        net_credit=net_credit,
        underlying_price_now=underlying_now,
        summary=summary,
        points=points,
    )
