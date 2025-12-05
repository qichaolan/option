"""LEAPS ranking API routes."""

import logging
import os
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

# Add parent directory to path to import leaps_ranker
# Works for both local dev (from web/) and Docker (from /app)
current_dir = Path(__file__).parent.parent.parent  # web/app/routes -> web
repo_root = current_dir.parent  # web -> repo root (for local dev)

# Try repo root first (local dev), then current working directory (Docker)
for path in [repo_root, Path("/app"), Path.cwd()]:
    if (path / "leaps_ranker.py").exists():
        sys.path.insert(0, str(path))
        break

from leaps_ranker import rank_leaps, load_config

from app.models import (
    LEAPSRequest,
    LEAPSResponse,
    LEAPSContract,
    ROISimulatorRequest,
    ROISimulatorResponse,
    ROISimulatorResult,
    TickerInfo,
)

router = APIRouter(prefix="/api", tags=["leaps"])


def get_config_path() -> Path:
    """Find config file in various locations."""
    possible_paths = [
        Path("/app/config/leaps_ranker.yaml"),  # Docker
        Path.cwd() / "config" / "leaps_ranker.yaml",  # CWD
        Path(__file__).parent.parent.parent.parent / "config" / "leaps_ranker.yaml",  # Local dev
    ]
    for p in possible_paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"Config file not found. Tried: {possible_paths}")

# Supported tickers with default annual target percentages
# Based on Median Annual Total Returns from 2010-2024
SUPPORTED_TICKERS = {
    "SPY": {"name": "S&P 500 ETF", "default_target_pct": 0.16},
    "QQQ": {"name": "Nasdaq 100 ETF", "default_target_pct": 0.1956},
    "IWM": {"name": "Russell 2000 ETF", "default_target_pct": 0.14},
    "GOOG": {"name": "Alphabet Inc.", "default_target_pct": 0.30},
    "NVDA": {"name": "NVIDIA Corp.", "default_target_pct": 0.30},
    "MSFT": {"name": "Microsoft Corp.", "default_target_pct": 0.25},
}


@router.get("/tickers", response_model=List[TickerInfo])
@limiter.limit("30/minute")
async def get_supported_tickers(request: Request):
    """Get list of supported tickers."""
    return [
        TickerInfo(
            symbol=symbol,
            name=info["name"],
            default_target_pct=info["default_target_pct"],
        )
        for symbol, info in SUPPORTED_TICKERS.items()
    ]


@router.post("/leaps", response_model=LEAPSResponse)
@limiter.limit("10/minute")
async def get_leaps_ranking(request: Request, leaps_request: LEAPSRequest):
    """
    Get ranked LEAPS options for a given ticker.

    Args:
        leaps_request: LEAPSRequest with symbol, target_pct, mode, and top_n

    Returns:
        LEAPSResponse with ranked contracts
    """
    symbol = leaps_request.symbol.upper()

    if symbol not in SUPPORTED_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported ticker: {symbol}. Supported: {list(SUPPORTED_TICKERS.keys())}",
        )

    try:
        # Load config
        config = load_config(str(get_config_path()))

        # Run the ranker
        df = rank_leaps(
            symbol=symbol,
            provider="cboe",
            mode=leaps_request.mode,
            target_pct=leaps_request.target_pct,
            min_dte=365,
            longest_only=True,
            top_n=leaps_request.top_n,
            config=config,
        )

        # Convert DataFrame to list of contracts using vectorized operations
        # Select columns that match LEAPSContract model
        contract_columns = [
            "contract_symbol", "expiration", "strike", "target_price",
            "premium", "cost", "payoff_target", "roi_target",
            "ease_score", "roi_score", "score", "implied_volatility", "open_interest"
        ]
        available_cols = [c for c in contract_columns if c in df.columns]
        df_out = df[available_cols].copy()

        # Required float fields - replace NaN/inf with 0 to prevent Pydantic validation errors
        required_float_fields = [
            "strike", "target_price", "premium", "cost", "payoff_target",
            "roi_target", "ease_score", "roi_score", "score"
        ]
        for col in required_float_fields:
            if col in df_out.columns:
                # Use numpy for reliable NaN/inf handling
                df_out[col] = df_out[col].replace([np.inf, -np.inf], np.nan)
                df_out[col] = df_out[col].fillna(0.0)
                # Double-check: convert any remaining non-finite values
                df_out[col] = df_out[col].apply(lambda x: 0.0 if not np.isfinite(x) else x)

        # Filter out rows with missing required string fields
        if "contract_symbol" in df_out.columns:
            df_out = df_out[df_out["contract_symbol"].notna() & (df_out["contract_symbol"] != "")]
        if "expiration" in df_out.columns:
            df_out = df_out[df_out["expiration"].notna() & (df_out["expiration"] != "")]

        # Convert to list of dicts
        records = df_out.to_dict(orient="records")

        # Clean up optional fields in each record (handle NaN -> None)
        for record in records:
            # Handle implied_volatility: NaN or 0 -> None
            if "implied_volatility" in record:
                val = record["implied_volatility"]
                if val is None or (isinstance(val, float) and (not np.isfinite(val) or val == 0)):
                    record["implied_volatility"] = None

            # Handle open_interest: NaN or 0 -> None, otherwise convert to int
            if "open_interest" in record:
                val = record["open_interest"]
                if val is None or (isinstance(val, float) and (not np.isfinite(val) or val == 0)):
                    record["open_interest"] = None
                elif val is not None:
                    record["open_interest"] = int(val)

        # Convert to Pydantic models
        contracts = []
        for record in records:
            try:
                contracts.append(LEAPSContract(**record))
            except Exception as e:
                # Log the problematic record and skip it
                logger.warning(f"Skipping invalid contract record: {e}")
                logger.debug(f"Record data: {record}")

        # Get underlying and target prices from dataframe
        # Use the actual underlying price from the data, not reverse-calculated
        # Note: compute_metrics sets "current_underlying_price" column
        underlying_price = 0.0
        target_price = 0.0
        if not df.empty:
            if "current_underlying_price" in df.columns:
                underlying_price = float(df["current_underlying_price"].iloc[0])
            if "target_price" in df.columns:
                target_price = float(df["target_price"].iloc[0])

        # Calculate effective target_pct from actual prices (compounded, not annual)
        effective_target_pct = (target_price / underlying_price - 1) if underlying_price > 0 else leaps_request.target_pct

        return LEAPSResponse(
            symbol=symbol,
            underlying_price=round(underlying_price, 2),
            target_price=round(target_price, 2),
            target_pct=effective_target_pct,
            mode=leaps_request.mode,
            contracts=contracts,
            timestamp=datetime.utcnow().isoformat(),
        )

    except FileNotFoundError as e:
        logger.error(f"Configuration error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Service configuration error. Please try again later.")
    except ValueError as e:
        logger.warning(f"Validation error for {symbol}: {e}")
        raise HTTPException(status_code=400, detail="Invalid request parameters. Please check your input.")
    except Exception as e:
        logger.exception(f"Unexpected error fetching LEAPS for {symbol}")
        raise HTTPException(status_code=500, detail="Unable to fetch options data. Please try again later.")


@router.post("/roi-simulator", response_model=ROISimulatorResponse)
@limiter.limit("30/minute")
async def simulate_roi(request: Request, sim_request: ROISimulatorRequest):
    """
    Simulate ROI for different target prices.

    Args:
        sim_request: ROISimulatorRequest with strike, premium, underlying, and targets

    Returns:
        ROISimulatorResponse with simulation results
    """
    cost = sim_request.premium * sim_request.contract_size
    results = []

    for target in sim_request.target_prices:
        intrinsic = max(target - sim_request.strike, 0)
        payoff = intrinsic * sim_request.contract_size
        profit = payoff - cost
        roi_pct = (profit / cost) * 100 if cost > 0 else 0
        price_change_pct = ((target - sim_request.underlying_price) / sim_request.underlying_price) * 100

        results.append(
            ROISimulatorResult(
                target_price=round(target, 2),
                price_change_pct=round(price_change_pct, 2),
                intrinsic_value=round(intrinsic, 2),
                payoff=round(payoff, 2),
                profit=round(profit, 2),
                roi_pct=round(roi_pct, 2),
            )
        )

    return ROISimulatorResponse(
        strike=sim_request.strike,
        premium=sim_request.premium,
        cost=round(cost, 2),
        underlying_price=sim_request.underlying_price,
        results=results,
    )
