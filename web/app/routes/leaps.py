"""LEAPS ranking API routes."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

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

# Supported tickers with default target percentages
SUPPORTED_TICKERS = {
    "SPY": {"name": "S&P 500 ETF", "default_target_pct": 0.32},
    "QQQ": {"name": "Nasdaq 100 ETF", "default_target_pct": 0.45},
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

        # Handle optional fields - convert NaN to None
        if "implied_volatility" in df_out.columns:
            df_out["implied_volatility"] = df_out["implied_volatility"].where(
                df_out["implied_volatility"].notna() & (df_out["implied_volatility"] != 0), None
            )
        if "open_interest" in df_out.columns:
            df_out["open_interest"] = df_out["open_interest"].where(
                df_out["open_interest"].notna() & (df_out["open_interest"] != 0), None
            )
            # Convert to int where not None
            df_out.loc[df_out["open_interest"].notna(), "open_interest"] = (
                df_out.loc[df_out["open_interest"].notna(), "open_interest"].astype(int)
            )

        # Convert to list of dicts and then to Pydantic models
        records = df_out.to_dict(orient="records")
        contracts = [LEAPSContract(**record) for record in records]

        # Get underlying and target prices from first contract
        underlying_price = 0.0
        target_price = 0.0
        if contracts:
            target_price = contracts[0].target_price
            underlying_price = target_price / (1 + leaps_request.target_pct)

        return LEAPSResponse(
            symbol=symbol,
            underlying_price=round(underlying_price, 2),
            target_price=round(target_price, 2),
            target_pct=leaps_request.target_pct,
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
