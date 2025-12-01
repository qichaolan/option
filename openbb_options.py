#!/usr/bin/env python3
"""
Bull Call Spread Scanner & Analyzer

Scans all bull call spread combinations for a given ticker and expiry,
evaluates risk/reward and scenario performance, and ranks spreads by
configurable criteria.

Features:
- Pull clean option chains from OpenBB
- Generate all valid bull call spread candidates
- Compute payoff metrics (max profit, max loss, breakeven)
- Scenario analysis (underlying down X%, flat, up Y%)
- Probability & expected value calculations
- Scoring function to rank spreads
- Comprehensive visualizations
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, date

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Default scenario percentages for P/L analysis
# -3%, 0% (flat), +1%, +2%, +3%, +4%, +5%, +7%, +10%
DEFAULT_SCENARIOS = [-0.03, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]

# -----------------------------------------------------------------------------
# Scoring Weight Profiles (weights must sum to 1.0)
# Components: ease, profit, scenario, convexity, liquidity, slippage
# -----------------------------------------------------------------------------
SCORING_PROFILES = {
    'balanced': {
        'ease': 0.35, 'profit': 0.25, 'scenario': 0.20,
        'convexity': 0.05, 'liquidity': 0.10, 'slippage': 0.05
    },  # Recommended default - balanced across all factors
    'high_probability': {
        'ease': 0.40, 'profit': 0.20, 'scenario': 0.25,
        'convexity': 0.05, 'liquidity': 0.05, 'slippage': 0.05
    },  # Favor easy breakeven + near-term scenario performance (+2%/+3%/+5%)
    'convexity': {
        'ease': 0.15, 'profit': 0.30, 'scenario': 0.10,
        'convexity': 0.35, 'liquidity': 0.05, 'slippage': 0.05
    },  # Favor tail payoff (+7%/+10%), accept low probability
    'conservative': {
        'ease': 0.45, 'profit': 0.15, 'scenario': 0.15,
        'convexity': 0.05, 'liquidity': 0.12, 'slippage': 0.08
    },  # Favor easy breakeven + execution quality
    'aggressive': {
        'ease': 0.20, 'profit': 0.40, 'scenario': 0.20,
        'convexity': 0.10, 'liquidity': 0.07, 'slippage': 0.03
    },  # Favor profit potential
}

# Scenario weights for composite scenario score
# Only near-term moves (+2%, +3%, +5%) used for scenario_score
# Tail moves (+7%, +10%) are used separately for convexity_score
SCENARIO_WEIGHTS = {
    'up_2': 0.40,   # +2% move (most likely)
    'up_3': 0.35,   # +3% move
    'up_5': 0.25,   # +5% move
}

# Validate scenario weights sum to 1.0
assert abs(sum(SCENARIO_WEIGHTS.values()) - 1.0) < 1e-6, \
    f"SCENARIO_WEIGHTS must sum to 1.0, got {sum(SCENARIO_WEIGHTS.values())}"

# -----------------------------------------------------------------------------
# Scoring Caps & Penalties
# -----------------------------------------------------------------------------
# Reward-to-Risk cap (prevents extreme R:R from dominating)
RR_CAP = 10.0

# Scenario score cap (prevents extreme scenario values from dominating)
SCENARIO_CAP = 5.0

# Convexity score cap (prevents extreme tail payoffs from dominating)
CONVEXITY_CAP = 5.0

# Ease score smooth penalty parameters
EASE_PENALTY_K = 20.0  # Decay rate for penalty beyond max_be_pct

# -----------------------------------------------------------------------------
# Liquidity & Slippage Parameters
# -----------------------------------------------------------------------------
# Liquidity target: score = 1.0 when min(OI, volume) >= this value
LIQUIDITY_TARGET = 10

# Slippage penalty factor: higher = more penalty for wide bid-ask spreads
SLIPPAGE_PENALTY_FACTOR = 5.0

# Maximum allowed bid-ask spread percentage (for optional hard filter)
MAX_BID_ASK_SPREAD_PCT = 0.10  # 10%

# -----------------------------------------------------------------------------
# Default Values
# -----------------------------------------------------------------------------
# Default implied volatility when IV is unavailable
DEFAULT_IV = 0.25

# Minimum values to prevent division by zero
MIN_BREAKEVEN_DISTANCE = 0.01
MIN_DAYS_TO_EXPIRY = 1 / 365.0


# =============================================================================
# SCORE NORMALIZATION BOUNDS
# =============================================================================
# Min/max bounds for linear normalization of each score type
# These define the expected range of raw scores before normalization

SCORE_BOUNDS = {
    'profit': {'min': 0.0, 'max': 5.0},      # R:R of 0 to 5 maps to [0, 1]
    'scenario': {'min': 0.0, 'max': 2.0},    # Scenario raw 0-2 maps to [0, 1]
    'convexity': {'min': 0.0, 'max': 2.0},   # Convexity raw 0-2 maps to [0, 1]
    'liquidity': {'min': 0.0, 'max': 1.0},   # Ratio 0-1 maps to [0, 1]
    'slippage': {'min': 0.0, 'max': 0.10},   # 0-10% slippage maps to [0, 1]
}


# =============================================================================
# SCORING HELPER FUNCTIONS
# =============================================================================

def normalize_linear(raw: float, min_val: float, max_val: float) -> float:
    """
    Linear normalization to [0, 1] range.

    Maps raw values from [min_val, max_val] to [0, 1] linearly.
    Values outside the range are clamped.

    Args:
        raw: Raw score value
        min_val: Minimum expected value (maps to 0)
        max_val: Maximum expected value (maps to 1)

    Returns:
        Normalized score in range [0, 1]
    """
    if max_val <= min_val:
        return 0.5
    clamped = max(min_val, min(raw, max_val))
    return (clamped - min_val) / (max_val - min_val)


def normalize_score(raw_score: float) -> float:
    """
    Legacy normalize function using x / (x + 1) formula.

    DEPRECATED: Use normalize_linear() for better normalization.

    Args:
        raw_score: Raw score value (should be >= 0)

    Returns:
        Normalized score in range [0, 1)
    """
    if raw_score <= 0:
        return 0.0
    return raw_score / (raw_score + 1)


def compute_ease_score(breakeven_distance: float) -> float:
    """
    Compute the 'ease of breakeven' score.

    Uses a piecewise linear mapping based on breakeven distance:

    | BE Distance | Ease Score |
    |-------------|------------|
    | ≤0% (ITM)   | 1.00       |
    | 0–1%        | 0.99       |
    | 1–2%        | 0.90       |
    | 2–3%        | 0.80       |
    | 3–4%        | 0.50       |
    | 4–5%        | 0.25       |
    | 5–6%        | 0.10       |
    | 6–8%        | 0.03       |
    | ≥8%         | 0.00       |

    Args:
        breakeven_distance: float
            (breakeven / current_price) - 1, e.g.:
              -0.03 → breakeven 3% below spot (excellent)
              +0.02 → need 2% move
              +0.05 → need 5% move (hard)

    Returns:
        float: ease_score in [0, 1]
    """
    # Breakeven BELOW current price → perfect score
    if breakeven_distance <= 0:
        return 1.0

    # Convert to percentage for easier mapping
    bd_pct = breakeven_distance * 100  # e.g., 0.02 → 2.0

    # Piecewise linear interpolation based on the mapping table
    # Points: (0, 1.0), (1, 0.99), (2, 0.90), (3, 0.80), (4, 0.50), (5, 0.25), (6, 0.10), (8, 0.03)
    breakpoints = [0, 1, 2, 3, 4, 5, 6, 8]
    scores = [1.00, 0.99, 0.90, 0.80, 0.50, 0.25, 0.10, 0.03]

    # Beyond 8% → 0
    if bd_pct >= 8:
        return 0.0

    # Find the segment and interpolate
    for i in range(len(breakpoints) - 1):
        if bd_pct <= breakpoints[i + 1]:
            # Linear interpolation between breakpoints[i] and breakpoints[i+1]
            t = (bd_pct - breakpoints[i]) / (breakpoints[i + 1] - breakpoints[i])
            ease_score = scores[i] + t * (scores[i + 1] - scores[i])
            return float(ease_score)

    return 0.0


def compute_profit_score(max_profit: float, max_loss: float) -> float:
    """
    Compute the profit attractiveness score component.

    Based on reward-to-risk ratio (capital efficiency).
    Higher R:R = higher score, capped at RR_CAP to prevent extreme values.

    Formula:
        RR_raw = min(max_profit / max_loss, RR_CAP)
        profit_score = log1p(RR_raw) / log1p(RR_CAP)

    Args:
        max_profit: Maximum profit per share
        max_loss: Maximum loss per share (debit)

    Returns:
        Normalized profit score in range (0, 1)
    """
    if max_loss <= 0 or max_profit <= 0:
        return 0.0

    rr_raw = max_profit / max_loss
    rr_raw = min(rr_raw, RR_CAP)  # Cap extreme R:R values

    # Log scaling: log1p(RR_raw) / log1p(RR_max)
    return math.log1p(rr_raw) / math.log1p(RR_CAP)


def compute_scenario_score(
    pl_up_2: float,
    pl_up_3: float,
    pl_up_5: float,
    max_loss: float,
    cap: float = None,
) -> float:
    """
    Compute the scenario-based profit score for near-term moves.

    Measures performance on realistic upside moves (+2%, +3%, +5%).
    Uses R-multiples (P/L / max_loss) with weighted sum.

    Formula:
        r2 = pl_up_2 / max_loss
        r3 = pl_up_3 / max_loss
        r5 = pl_up_5 / max_loss
        scenario_raw = 0.40*r2 + 0.35*r3 + 0.25*r5
        scenario_raw = clamp(scenario_raw, 0, cap)
        scenario_score = normalize(scenario_raw)

    Args:
        pl_up_2: P/L at +2% underlying move
        pl_up_3: P/L at +3% underlying move
        pl_up_5: P/L at +5% underlying move
        max_loss: Maximum loss per share (for R-multiple calculation)
        cap: Maximum raw score before normalization (default: SCENARIO_CAP)

    Returns:
        Normalized scenario score in range [0, 1)
    """
    if cap is None:
        cap = SCENARIO_CAP

    # Guard for invalid max_loss
    if max_loss is None or max_loss <= 0:
        return 0.0

    # Convert to R-multiples
    r2 = float(pl_up_2) / float(max_loss)
    r3 = float(pl_up_3) / float(max_loss)
    r5 = float(pl_up_5) / float(max_loss)

    # Weighted sum using SCENARIO_WEIGHTS
    scenario_raw = (
        SCENARIO_WEIGHTS['up_2'] * r2 +
        SCENARIO_WEIGHTS['up_3'] * r3 +
        SCENARIO_WEIGHTS['up_5'] * r5
    )

    # Clamp to [0, cap]
    scenario_raw = max(0.0, min(scenario_raw, cap))

    # Linear normalization
    bounds = SCORE_BOUNDS['scenario']
    return normalize_linear(scenario_raw, bounds['min'], bounds['max'])


def compute_convexity_score(
    pl_up_2: float,
    pl_up_3: float,
    pl_up_7: float,
    pl_up_10: float,
    max_loss: float,
    cap: float = None,
) -> float:
    """
    Compute the convexity score for tail payoff (big moves).

    Measures extra payoff in the tail moves (+7%, +10%) relative to
    near-term moves (+2%, +3%). Higher convexity = more "lottery" upside.

    Formula:
        r2  = pl_up_2  / max_loss
        r3  = pl_up_3  / max_loss
        r7  = pl_up_7  / max_loss
        r10 = pl_up_10 / max_loss
        convexity_raw = 0.5*(r7 + r10) - 0.5*(r2 + r3)
        convexity_raw = clamp(convexity_raw, 0, cap)
        convexity_score = normalize(convexity_raw)

    Args:
        pl_up_2: P/L at +2% underlying move
        pl_up_3: P/L at +3% underlying move
        pl_up_7: P/L at +7% underlying move
        pl_up_10: P/L at +10% underlying move
        max_loss: Maximum loss per share (for R-multiple calculation)
        cap: Maximum raw score before normalization (default: CONVEXITY_CAP)

    Returns:
        Normalized convexity score in range [0, 1)
    """
    if cap is None:
        cap = CONVEXITY_CAP

    # Guard for invalid max_loss
    if max_loss is None or max_loss <= 0:
        return 0.0

    # Convert to R-multiples
    r2 = float(pl_up_2) / float(max_loss)
    r3 = float(pl_up_3) / float(max_loss)
    r7 = float(pl_up_7) / float(max_loss)
    r10 = float(pl_up_10) / float(max_loss)

    # Compute convexity: extra tail payoff vs near-term payoff
    # Positive = tail moves pay more than near-term (good for lottery spreads)
    convexity_raw = 0.5 * (r7 + r10) - 0.5 * (r2 + r3)

    # Clamp to [0, cap] - only reward positive convexity
    convexity_raw = max(0.0, min(convexity_raw, cap))

    # Linear normalization
    bounds = SCORE_BOUNDS['convexity']
    return normalize_linear(convexity_raw, bounds['min'], bounds['max'])


def compute_liquidity_score(
    long_oi: float,
    short_oi: float,
    long_volume: float,
    short_volume: float,
    liquidity_threshold: float = None,
) -> float:
    """
    Compute the liquidity score component (binary).

    Measures whether the spread is actually tradable based on
    open interest and volume of both legs.

    Formula:
        min_metric = min(long_oi, short_oi, long_vol, short_vol)
        liquidity_score = 1.0 if min_metric >= threshold else 0.0

    Args:
        long_oi: Open interest of long leg
        short_oi: Open interest of short leg
        long_volume: Volume of long leg
        short_volume: Volume of short leg
        liquidity_threshold: Minimum threshold for liquidity (default: LIQUIDITY_TARGET)

    Returns:
        Liquidity score: 1.0 if liquid, 0.0 if not
    """
    if liquidity_threshold is None:
        liquidity_threshold = LIQUIDITY_TARGET

    if liquidity_threshold <= 0:
        return 1.0  # No liquidity requirement

    # Binary: pass or fail based on minimum metric
    min_metric = min(long_oi, short_oi, long_volume, short_volume)
    return 1.0 if min_metric >= liquidity_threshold else 0.0


def compute_slippage_score(
    long_bid: float,
    long_ask: float,
    short_bid: float,
    short_ask: float,
) -> float:
    """
    Compute the slippage score component.

    Measures execution cost due to bid-ask spread.
    High slippage = low score.

    Formula:
        mid_long = (bid_long + ask_long) / 2
        mid_short = (bid_short + ask_short) / 2
        slippage_long = (ask_long - bid_long) / mid_long
        slippage_short = (ask_short - bid_short) / mid_short
        slippage_raw = max(slippage_long, slippage_short)
        z = (slippage_raw - mean) / std
        slippage_score = 1 - sigmoid(z)  # Invert: lower slippage = higher score

    Args:
        long_bid: Bid price of long leg
        long_ask: Ask price of long leg
        short_bid: Bid price of short leg
        short_ask: Ask price of short leg

    Returns:
        Slippage score in range (0, 1)
        Higher = tight spread (good), Lower = wide spread (bad)
    """
    # Calculate mid prices
    mid_long = (long_bid + long_ask) / 2
    mid_short = (short_bid + short_ask) / 2

    # Handle edge cases
    if mid_long <= 0 or mid_short <= 0:
        return 0.5  # Default to medium score if prices invalid

    # Calculate percentage slippage for each leg
    slippage_long = (long_ask - long_bid) / mid_long
    slippage_short = (short_ask - short_bid) / mid_short

    # Use the worse (higher) slippage
    slippage_raw = max(slippage_long, slippage_short)

    # Linear normalization (inverted: lower slippage = higher score)
    bounds = SCORE_BOUNDS['slippage']
    normalized = normalize_linear(slippage_raw, bounds['min'], bounds['max'])

    # Invert: we want LOW slippage to give HIGH score
    return 1.0 - normalized


def compute_composite_score(
    ease_score: float,
    profit_score: float,
    scenario_score: float,
    convexity_score: float,
    liquidity_score: float,
    slippage_score: float,
    weights: Dict[str, float],
) -> float:
    """
    Compute final composite score from all component scores.

    All component scores are normalized to [0, 1], and weights sum to 1.0,
    ensuring the final score is also in [0, 1].

    Formula:
        score = w_ease*ease + w_profit*profit + w_scenario*scenario +
                w_convexity*convexity + w_liquidity*liquidity + w_slippage*slippage

    Args:
        ease_score: Normalized ease of breakeven score [0, 1]
        profit_score: Normalized profit attractiveness score [0, 1]
        scenario_score: Normalized scenario performance score [0, 1] (near-term moves)
        convexity_score: Normalized convexity/tail payoff score [0, 1] (big moves)
        liquidity_score: Normalized liquidity score [0, 1]
        slippage_score: Normalized slippage/execution score [0, 1]
        weights: Dict with keys 'ease', 'profit', 'scenario', 'convexity',
                 'liquidity', 'slippage' for component weights

    Returns:
        Final composite score in [0, 1]
    """
    return (
        weights.get('ease', 0.30) * ease_score +
        weights.get('profit', 0.30) * profit_score +
        weights.get('scenario', 0.15) * scenario_score +
        weights.get('convexity', 0.15) * convexity_score +
        weights.get('liquidity', 0.05) * liquidity_score +
        weights.get('slippage', 0.05) * slippage_score
    )


@dataclass
class ScenarioResult:
    """Result of a scenario analysis."""
    scenario_name: str
    underlying_price: float
    underlying_move_pct: float
    pnl_per_share: float
    pnl_per_contract: float
    pnl_pct_of_max_loss: float
    return_pct: float  # payoff / debit (percentage return on risk)


@dataclass
class SpreadCandidate:
    """A bull call spread candidate with all metrics."""
    # Basic spread info
    long_strike: float
    short_strike: float
    spread_width: float

    # Premium info
    long_premium: float
    short_premium: float
    net_debit: float

    # Payoff metrics
    max_profit: float
    max_loss: float
    breakeven: float
    reward_to_risk: float

    # Greeks (if available)
    long_delta: Optional[float] = None
    short_delta: Optional[float] = None
    net_delta: Optional[float] = None
    long_iv: Optional[float] = None
    short_iv: Optional[float] = None

    # Probability metrics
    prob_profit: Optional[float] = None
    prob_max_profit: Optional[float] = None
    expected_value: Optional[float] = None

    # Scenario results
    scenarios: List[ScenarioResult] = field(default_factory=list)

    # Scoring
    composite_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'Long Strike': self.long_strike,
            'Short Strike': self.short_strike,
            'Width': self.spread_width,
            'Net Debit': self.net_debit,
            'Max Profit': self.max_profit,
            'Max Loss': self.max_loss,
            'Breakeven': self.breakeven,
            'R:R Ratio': self.reward_to_risk,
            'Net Delta': self.net_delta,
            'Prob Profit': self.prob_profit,
            'Prob Max Profit': self.prob_max_profit,
            'Expected Value': self.expected_value,
            'Score': self.composite_score,
        }


class BullCallSpreadScanner:
    """
    Comprehensive Bull Call Spread Scanner using OpenBB.

    Workflow:
    1. Pull clean option chains (calls) from OpenBB (obb.derivatives.options)
    2. Generate all valid bull call spread candidates
    3. Compute basic payoff metrics
    4. Scenario analysis (down X%, flat, up Y%)
    5. Probability & expected value calculations
    6. Score and rank spreads
    7. Visualize results

    Data Sources (via OpenBB):
    - obb.derivatives.options.chains - Get full options chain
    - obb.equity.price.quote - Get current underlying price

    Note: Falls back to direct yfinance if OpenBB has compatibility issues.
    """

    def __init__(self):
        """Initialize the scanner with OpenBB."""
        self._obb = None
        self._yf = None  # Fallback
        self._use_openbb = True
        self._initialize_openbb()

    def _initialize_openbb(self):
        """Initialize OpenBB SDK for options data."""
        try:
            from openbb import obb
            self._obb = obb
            logger.info("OpenBB initialized successfully")

            # Test if OpenBB yfinance provider works
            # If not, we'll fall back to direct yfinance
            try:
                # Quick test - this will fail if there's a compatibility issue
                test = self._obb.derivatives.options.chains(symbol='SPY', provider='yfinance')
                test.to_df()
            except Exception as e:
                logger.warning(f"OpenBB yfinance provider has issues: {e}")
                logger.info("Falling back to direct yfinance access")
                self._use_openbb = False
                self._initialize_yfinance_fallback()

        except ImportError:
            logger.warning("OpenBB not installed, using yfinance directly")
            self._use_openbb = False
            self._initialize_yfinance_fallback()

    def _initialize_yfinance_fallback(self):
        """Initialize yfinance as fallback."""
        try:
            import yfinance as yf
            self._yf = yf
            logger.info("yfinance fallback initialized")
        except ImportError:
            raise ImportError("yfinance required. Install with: pip install yfinance")

    # =========================================================================
    # STEP 1: Pull Clean Option Chains (via OpenBB obb.derivatives.options)
    # =========================================================================

    def get_current_price(self, ticker: str) -> float:
        """
        Get current underlying price.

        Uses: obb.equity.price.quote (with yfinance fallback)
        """
        if self._use_openbb:
            try:
                result = self._obb.equity.price.quote(symbol=ticker.upper(), provider='yfinance')
                df = result.to_df()
                for col in ['last_price', 'close', 'price', 'regularMarketPrice', 'previousClose']:
                    if col in df.columns:
                        price = df[col].iloc[0]
                        if price is not None and not pd.isna(price):
                            return float(price)
            except Exception as e:
                logger.debug(f"OpenBB quote failed, using fallback: {e}")

        # Fallback to direct yfinance
        if self._yf is None:
            self._initialize_yfinance_fallback()

        try:
            stock = self._yf.Ticker(ticker.upper())
            info = stock.info
            price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')

            if price is None:
                hist = stock.history(period='1d')
                if not hist.empty:
                    price = hist['Close'].iloc[-1]

            if price is None:
                raise ValueError(f"Could not find price for {ticker}")

            return float(price)
        except Exception as e:
            logger.error(f"Failed to get price for {ticker}: {e}")
            raise

    def get_available_expirations(self, ticker: str) -> List[str]:
        """
        Get available expiration dates for a ticker.

        Uses: Extract from obb.derivatives.options.chains or yfinance.Ticker.options
        """
        if self._use_openbb:
            try:
                result = self._obb.derivatives.options.chains(symbol=ticker.upper(), provider='yfinance')
                df = result.to_df()
                expirations = df['expiration'].unique().tolist()
                expirations = sorted([str(exp)[:10] for exp in expirations])
                logger.info(f"Found {len(expirations)} expirations for {ticker} via OpenBB")
                return expirations
            except Exception as e:
                logger.debug(f"OpenBB expirations failed, using fallback: {e}")

        # Fallback to direct yfinance
        if self._yf is None:
            self._initialize_yfinance_fallback()

        try:
            stock = self._yf.Ticker(ticker.upper())
            expirations = list(stock.options)
            logger.info(f"Found {len(expirations)} expirations for {ticker}")
            return expirations
        except Exception as e:
            logger.error(f"Failed to get expirations: {e}")
            raise

    def get_calls_chain(self, ticker: str, expiry: str) -> pd.DataFrame:
        """
        Pull clean call options chain for ticker & expiry.

        Uses: obb.derivatives.options.chains (with yfinance fallback)

        Returns DataFrame with columns: strike, bid, ask, mid, lastPrice, volume,
        openInterest, impliedVolatility, delta, etc.
        """
        df = None

        # Try OpenBB first
        if self._use_openbb:
            try:
                result = self._obb.derivatives.options.chains(symbol=ticker.upper(), provider='yfinance')
                df = result.to_df()

                # Filter by expiration
                df['expiration_str'] = df['expiration'].astype(str).str[:10]
                df = df[df['expiration_str'] == expiry].copy()

                # Filter for calls only
                df = df[df['option_type'] == 'call'].copy()

                # Standardize column names from OpenBB format
                col_mapping = {
                    'last_trade_price': 'lastPrice',
                    'open_interest': 'openInterest',
                    'implied_volatility': 'impliedVolatility',
                    'mark': 'mid'
                }
                df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

                logger.info(f"Retrieved {len(df)} call options for {ticker} @ {expiry} via OpenBB")

            except Exception as e:
                logger.debug(f"OpenBB chains failed, using fallback: {e}")
                df = None

        # Fallback to direct yfinance
        if df is None or df.empty:
            if self._yf is None:
                self._initialize_yfinance_fallback()

            try:
                stock = self._yf.Ticker(ticker.upper())
                opt_chain = stock.option_chain(expiry)
                df = opt_chain.calls.copy()
                logger.info(f"Retrieved {len(df)} call options for {ticker} @ {expiry} via yfinance")
            except Exception as e:
                logger.error(f"Failed to get options chain: {e}")
                raise

        if df.empty:
            raise ValueError(f"No call options found for {ticker} expiring {expiry}")

        # Calculate mid price if not present
        if 'mid' not in df.columns:
            df['mid'] = (df['bid'].fillna(0) + df['ask'].fillna(0)) / 2

        # Use lastPrice as fallback where mid is 0
        if 'lastPrice' in df.columns:
            df.loc[df['mid'] == 0, 'mid'] = df.loc[df['mid'] == 0, 'lastPrice']

        # Standardize IV column
        if 'impliedVolatility' in df.columns:
            df['iv'] = df['impliedVolatility']
        elif 'implied_volatility' in df.columns:
            df['iv'] = df['implied_volatility']
        else:
            df['iv'] = None

        # Sort by strike
        df = df.sort_values('strike').reset_index(drop=True)

        # Filter out options with no liquidity
        df = df[(df['mid'] > 0) | (df.get('bid', pd.Series([0])) > 0).any()].copy()

        return df

    # =========================================================================
    # STEP 2: Generate All Valid Bull Call Spread Candidates
    # =========================================================================

    def generate_spread_candidates(
        self,
        calls_df: pd.DataFrame,
        underlying_price: float,
        # 2.1 Constraints
        max_debit_per_spread: float = 10.0,      # Max loss per spread in $ (per share)
        min_width: float = 5.0,                   # Min spread width in $
        max_width: float = 20.0,                  # Max spread width in $
        # 2.2 Pre-filter: Liquidity
        min_open_interest: int = 10,
        min_volume: int = 0,
        # 2.2 Pre-filter: Strike band
        strike_range_pct: float = 0.10,           # ±10% of spot
        strike_lower_bound: Optional[float] = None,  # Or explicit bounds
        strike_upper_bound: Optional[float] = None,
        # 2.5 Filter: Reward-to-Risk
        min_reward_to_risk: float = 0.5,
        # 2.5 Filter: Break-even
        max_breakeven_pct: Optional[float] = None,  # e.g., 0.10 = reject if BE > spot * 1.10
        # 2.5 Filter: Delta (optional)
        long_delta_range: Optional[Tuple[float, float]] = None,   # e.g., (0.4, 0.6)
        short_delta_range: Optional[Tuple[float, float]] = None,  # e.g., (0.2, 0.4)
    ) -> pd.DataFrame:
        """
        STEP 2: Generate all valid bull call spread combinations from a calls chain.

        Workflow:
        2.1 Define constraints up front (max_debit, width limits)
        2.2 Pre-filter calls chain (liquidity, valid bid/ask, strike band)
        2.3 Generate all (Long, Short) strike pairs
        2.4 Compute core spread metrics for each pair
        2.5 Apply filters to reject bad spreads
        2.6 Assemble final candidates table

        Args:
            calls_df: Clean calls chain DataFrame for one expiration
                Required columns: [strike, mid, bid, ask]
                Optional columns: [volume, openInterest, iv, delta, theta, vega]
            underlying_price: Current spot price of underlying

            # Constraints (2.1)
            max_debit_per_spread: Maximum debit (max loss) per spread in $ per share
            min_width: Minimum spread width in dollars
            max_width: Maximum spread width in dollars

            # Pre-filter (2.2)
            min_open_interest: Minimum open interest for each leg
            min_volume: Minimum volume for each leg (0 = no filter)
            strike_range_pct: Filter strikes to ±X% of spot (e.g., 0.10 = 10%)
            strike_lower_bound: Explicit lower strike bound (overrides pct if set)
            strike_upper_bound: Explicit upper strike bound (overrides pct if set)

            # Filters (2.5)
            min_reward_to_risk: Minimum R:R ratio (e.g., 0.5 or 1.0)
            max_breakeven_pct: Reject if breakeven > spot * (1 + pct)
            long_delta_range: (min, max) delta for long leg, e.g., (0.4, 0.6)
            short_delta_range: (min, max) delta for short leg, e.g., (0.2, 0.4)

        Returns:
            DataFrame of valid bull call spread candidates with columns:
            [long_strike, short_strike, width, long_mid, short_mid, net_debit,
             max_loss, max_profit, reward_to_risk, breakeven,
             long_delta, short_delta, net_delta,
             long_theta, short_theta, net_theta,
             long_vega, short_vega, net_vega,
             long_iv, short_iv, long_oi, short_oi, long_volume, short_volume,
             long_bid, long_ask, short_bid, short_ask]
        """
        logger.info("Step 2: Generating bull call spread candidates...")

        # =====================================================================
        # 2.1 Define Constraints Up Front
        # =====================================================================
        logger.debug(f"  Constraints: max_debit=${max_debit_per_spread}, "
                    f"width=[${min_width}, ${max_width}]")

        # =====================================================================
        # 2.2 Pre-Filter the Calls Chain
        # =====================================================================
        # Validate required columns
        required_cols = ['strike', 'mid', 'bid', 'ask']
        for col in required_cols:
            if col not in calls_df.columns:
                raise ValueError(f"calls_df missing required column: {col}")

        df = calls_df.copy()
        initial_count = len(df)

        # 2.2.1 Liquidity filter: volume and/or open_interest above thresholds
        if 'openInterest' in df.columns:
            df = df[df['openInterest'].fillna(0) >= min_open_interest]
        if min_volume > 0 and 'volume' in df.columns:
            df = df[df['volume'].fillna(0) >= min_volume]

        # 2.2.2 Valid bid/ask: non-zero, bid < ask
        df = df[(df['bid'] > 0) & (df['ask'] > 0) & (df['bid'] < df['ask'])]

        # 2.2.3 Strike band filter
        if strike_lower_bound is not None and strike_upper_bound is not None:
            # Use explicit bounds
            min_strike = strike_lower_bound
            max_strike = strike_upper_bound
        else:
            # Use percentage of spot
            min_strike = underlying_price * (1 - strike_range_pct)
            max_strike = underlying_price * (1 + strike_range_pct)

        df = df[(df['strike'] >= min_strike) & (df['strike'] <= max_strike)]

        # 2.2.4 Sort by strike ascending
        df = df.sort_values('strike').reset_index(drop=True)

        filtered_count = len(df)
        logger.debug(f"  Pre-filtered: {initial_count} -> {filtered_count} calls "
                    f"(strikes ${min_strike:.0f}-${max_strike:.0f})")

        if filtered_count < 2:
            logger.warning("Not enough tradable calls after pre-filtering")
            return pd.DataFrame()

        # Build array of calls for iteration (sorted by strike)
        calls_array = df.to_dict('records')
        n = len(calls_array)

        # =====================================================================
        # 2.3 Generate All (Long, Short) Strike Pairs
        # 2.4 Compute Core Spread Metrics for Each Pair
        # =====================================================================
        spread_records = []

        for i in range(n):
            long_call = calls_array[i]
            K_long = long_call['strike']
            price_long = long_call['mid']

            # Get long leg attributes
            long_bid = long_call['bid']
            long_ask = long_call['ask']
            long_oi = long_call.get('openInterest', 0) or 0
            long_vol = long_call.get('volume', 0) or 0
            long_delta = long_call.get('delta', None)
            long_theta = long_call.get('theta', None)
            long_vega = long_call.get('vega', None)
            long_iv = long_call.get('iv', None) or long_call.get('impliedVolatility', None)

            for j in range(i + 1, n):
                short_call = calls_array[j]
                K_short = short_call['strike']
                price_short = short_call['mid']

                # Get short leg attributes
                short_bid = short_call['bid']
                short_ask = short_call['ask']
                short_oi = short_call.get('openInterest', 0) or 0
                short_vol = short_call.get('volume', 0) or 0
                short_delta = short_call.get('delta', None)
                short_theta = short_call.get('theta', None)
                short_vega = short_call.get('vega', None)
                short_iv = short_call.get('iv', None) or short_call.get('impliedVolatility', None)

                # ----- 2.4 Compute Core Spread Metrics -----

                # Spread width
                width = K_short - K_long

                # Net debit (entry cost) using mid prices
                debit = price_long - price_short

                # Max loss = debit (per share)
                max_loss = debit

                # Max profit = width - debit
                max_profit = width - debit

                # Reward-to-Risk ratio
                rr = max_profit / max_loss if max_loss > 0 else float('inf')

                # Break-even price at expiry
                breakeven = K_long + debit

                # Net Greeks (if available)
                net_delta = None
                if long_delta is not None and short_delta is not None:
                    net_delta = long_delta - short_delta

                net_theta = None
                if long_theta is not None and short_theta is not None:
                    net_theta = long_theta - short_theta

                net_vega = None
                if long_vega is not None and short_vega is not None:
                    net_vega = long_vega - short_vega

                # =====================================================================
                # 2.5 Apply Filters to Reject Bad Spreads
                # =====================================================================

                # 2.5.1 Debit sanity check: must be a debit spread
                if debit <= 0:
                    continue

                # 2.5.2 Max debit constraint
                if debit > max_debit_per_spread:
                    continue

                # 2.5.3 Spread width constraints
                if width < min_width or width > max_width:
                    continue

                # 2.5.4 Break-even sanity (optional)
                if max_breakeven_pct is not None:
                    max_be = underlying_price * (1 + max_breakeven_pct)
                    if breakeven > max_be:
                        continue

                # 2.5.5 Reward-to-Risk filter
                if rr < min_reward_to_risk:
                    continue

                # 2.5.6 Delta filters (optional)
                if long_delta_range is not None and long_delta is not None:
                    if not (long_delta_range[0] <= long_delta <= long_delta_range[1]):
                        continue

                if short_delta_range is not None and short_delta is not None:
                    if not (short_delta_range[0] <= short_delta <= short_delta_range[1]):
                        continue

                # =====================================================================
                # 2.6 Assemble the Final Candidates Table
                # =====================================================================
                spread_records.append({
                    # Core identifiers
                    'long_strike': K_long,
                    'short_strike': K_short,
                    'width': width,

                    # Premium info
                    'long_mid': round(price_long, 4),
                    'short_mid': round(price_short, 4),
                    'net_debit': round(debit, 4),

                    # Payoff metrics
                    'max_loss': round(max_loss, 4),
                    'max_profit': round(max_profit, 4),
                    'reward_to_risk': round(rr, 4),
                    'breakeven': round(breakeven, 2),

                    # Greeks
                    'long_delta': round(long_delta, 4) if long_delta is not None else None,
                    'short_delta': round(short_delta, 4) if short_delta is not None else None,
                    'net_delta': round(net_delta, 4) if net_delta is not None else None,
                    'long_theta': round(long_theta, 4) if long_theta is not None else None,
                    'short_theta': round(short_theta, 4) if short_theta is not None else None,
                    'net_theta': round(net_theta, 4) if net_theta is not None else None,
                    'long_vega': round(long_vega, 4) if long_vega is not None else None,
                    'short_vega': round(short_vega, 4) if short_vega is not None else None,
                    'net_vega': round(net_vega, 4) if net_vega is not None else None,

                    # IVs
                    'long_iv': round(long_iv, 4) if long_iv is not None else None,
                    'short_iv': round(short_iv, 4) if short_iv is not None else None,

                    # Liquidity metrics
                    'long_oi': int(long_oi),
                    'short_oi': int(short_oi),
                    'long_volume': int(long_vol),
                    'short_volume': int(short_vol),

                    # Bid/Ask for slippage analysis
                    'long_bid': round(long_bid, 4),
                    'long_ask': round(long_ask, 4),
                    'short_bid': round(short_bid, 4),
                    'short_ask': round(short_ask, 4),
                })

        # Create DataFrame
        spreads_df = pd.DataFrame(spread_records)

        if spreads_df.empty:
            logger.warning("No valid spread candidates found with given constraints")
            return spreads_df

        # Sort by reward_to_risk descending (best R:R first)
        spreads_df = spreads_df.sort_values('reward_to_risk', ascending=False).reset_index(drop=True)

        logger.info(f"  Generated {len(spreads_df)} valid bull call spread candidates")
        return spreads_df

    # =========================================================================
    # STEP 3: Compute Basic Payoff Metrics for Each Spread
    # =========================================================================

    def compute_payoff_metrics(
        self,
        candidates_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        STEP 3: Compute and attach basic payoff metrics to each candidate.

        Goal: For each candidate row (K1, K2, debit), compute and add:
            - max_loss = debit
            - max_profit = (K2 - K1) - debit = width - debit
            - reward_to_risk = max_profit / max_loss
            - breakeven = K1 + debit

        Note: Step 2 already computes these during candidate generation for
        filtering purposes. This step ensures the columns are present and
        properly named, and can recompute if needed.

        Args:
            candidates_df: DataFrame from Step 2 with columns:
                Required: [long_strike, short_strike, net_debit]
                Optional (will be computed if missing): [width]

        Returns:
            DataFrame with payoff metric columns added/updated:
                [max_loss, max_profit, reward_to_risk, breakeven]
        """
        logger.info("Step 3: Computing basic payoff metrics for each spread...")

        if candidates_df.empty:
            logger.warning("Empty candidates DataFrame, skipping payoff metrics")
            return candidates_df

        df = candidates_df.copy()

        # Ensure required columns exist
        required = ['long_strike', 'short_strike', 'net_debit']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"candidates_df missing required column: {col}")

        # Extract values for vectorized computation
        K1 = df['long_strike']       # Long strike (lower)
        K2 = df['short_strike']      # Short strike (higher)
        debit = df['net_debit']      # Net debit paid

        # Compute width if not present
        if 'width' not in df.columns:
            df['width'] = K2 - K1

        # =====================================================================
        # Compute Payoff Metrics (vectorized for performance)
        # =====================================================================

        # Max Loss = debit (the most you can lose is what you paid)
        df['max_loss'] = debit.round(4)

        # Max Profit = width - debit (profit if underlying >= short strike at expiry)
        df['max_profit'] = (df['width'] - debit).round(4)

        # Reward-to-Risk Ratio = max_profit / max_loss
        # Handle division by zero (shouldn't happen with valid spreads)
        df['reward_to_risk'] = np.where(
            df['max_loss'] > 0,
            (df['max_profit'] / df['max_loss']).round(4),
            np.inf
        )

        # Break-even Price = K1 + debit (price where P/L = 0 at expiry)
        df['breakeven'] = (K1 + debit).round(2)

        logger.info(f"  Computed payoff metrics for {len(df)} candidates")
        logger.debug(f"  Max Loss range: ${df['max_loss'].min():.2f} - ${df['max_loss'].max():.2f}")
        logger.debug(f"  Max Profit range: ${df['max_profit'].min():.2f} - ${df['max_profit'].max():.2f}")
        logger.debug(f"  R:R range: {df['reward_to_risk'].min():.2f} - {df['reward_to_risk'].max():.2f}")

        return df

    def create_spread_candidate(
        self,
        row: pd.Series,
    ) -> SpreadCandidate:
        """
        Create a SpreadCandidate object from a DataFrame row.

        Helper method to convert a row from the candidates DataFrame
        (after Step 3) into a SpreadCandidate dataclass for use in
        Steps 4-6 (scenario analysis, probability, scoring).

        Args:
            row: A pandas Series (single row from candidates_df)

        Returns:
            SpreadCandidate object with all metrics populated
        """
        return SpreadCandidate(
            long_strike=row['long_strike'],
            short_strike=row['short_strike'],
            spread_width=row['width'],
            long_premium=row.get('long_mid', 0),
            short_premium=row.get('short_mid', 0),
            net_debit=row['net_debit'],
            max_profit=row['max_profit'],
            max_loss=row['max_loss'],
            breakeven=row['breakeven'],
            reward_to_risk=row['reward_to_risk'],
            long_delta=row.get('long_delta'),
            short_delta=row.get('short_delta'),
            net_delta=row.get('net_delta'),
            long_iv=row.get('long_iv'),
            short_iv=row.get('short_iv'),
        )

    # =========================================================================
    # STEP 4: Scenario Analysis - Underlying Down X%, Flat, Up Y%
    # =========================================================================

    def run_scenario_analysis(
        self,
        candidates_df: pd.DataFrame,
        current_price: float,
        scenarios: List[float] = None,  # Uses DEFAULT_SCENARIOS if None
    ) -> pd.DataFrame:
        """
        STEP 4: Scenario Analysis - See P/L at different underlying prices at expiry.

        Goal: For each spread, compute payoff at standardized scenario prices.

        Required Scenarios (9 total):
            -3%, 0% (flat), +1%, +2%, +3%, +4%, +5%, +7%, +10%

        Step-by-Step Pricing Logic:
        Let:
            S0 = current underlying price
            K_long = strike of long call
            K_short = strike of short call
            debit = net entry cost
            width = K_short - K_long

        For each scenario, compute underlying price at expiry:
            S_T = S0 * (1 + scenario_percentage)

        Then compute payoff:
            long_payoff  = max(S_T - K_long, 0)
            short_payoff = max(S_T - K_short, 0)
            spread_payoff = long_payoff - short_payoff - debit

        Output Columns:
            PL_down_3, PL_flat, PL_up_1, PL_up_2, PL_up_3, PL_up_4, PL_up_5, PL_up_7, PL_up_10
            ret_down_3, ret_flat, ret_up_1, ret_up_2, ret_up_3, ret_up_4, ret_up_5, ret_up_7, ret_up_10

        Args:
            candidates_df: DataFrame from Step 3 with columns:
                Required: [long_strike, short_strike, net_debit]
            current_price: Current underlying spot price (S0)
            scenarios: List of percentage moves as decimals
                Default: [-0.03, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]

        Returns:
            DataFrame with scenario P/L columns added:
                - PL_{scenario_name}: payoff per share for each scenario
                - ret_{scenario_name}: percentage return (payoff/debit) for each scenario
        """
        logger.info("Step 4: Running scenario analysis...")

        # Use default scenarios if not provided
        if scenarios is None:
            scenarios = DEFAULT_SCENARIOS

        if candidates_df.empty:
            logger.warning("Empty candidates DataFrame, skipping scenario analysis")
            return candidates_df

        df = candidates_df.copy()

        # Ensure required columns exist
        required_cols = ['long_strike', 'short_strike', 'net_debit']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"candidates_df missing required column: {col}")

        # Extract values for vectorized computation
        k_long = df['long_strike'].values    # Long strike (lower)
        k_short = df['short_strike'].values  # Short strike (higher)
        debit = df['net_debit'].values       # Net debit paid

        logger.debug(f"  Current price: ${current_price:.2f}")
        logger.debug(f"  Scenarios: {[f'{s*100:+.0f}%' for s in scenarios]}")

        # =====================================================================
        # Compute P/L for Each Scenario (vectorized)
        # =====================================================================
        for move_pct in scenarios:
            # 1. Calculate scenario price: S_T = S0 * (1 + move_pct)
            price_at_expiry = current_price * (1 + move_pct)

            # 2. Calculate net payoff at expiry:
            #    payoff = max(S_T - K_long, 0) - max(S_T - K_short, 0) - debit
            long_call_payoff = np.maximum(price_at_expiry - k_long, 0)
            short_call_payoff = np.maximum(price_at_expiry - k_short, 0)
            payoff = long_call_payoff - short_call_payoff - debit

            # 3. Calculate percentage return: payoff / debit
            #    - Positive payoff = profit, negative = loss
            #    - Max loss is -debit (return = -100%)
            return_pct = np.where(debit > 0, payoff / debit, 0)

            # 4. Create column name based on scenario
            scenario_name = self._get_scenario_column_name(move_pct)

            # Store P/L per share and return percentage
            df[f'PL_{scenario_name}'] = np.round(payoff, 4)
            df[f'ret_{scenario_name}'] = np.round(return_pct, 4)

        # Log summary statistics
        flat_col = 'PL_flat' if 'PL_flat' in df.columns else None
        if flat_col:
            logger.debug(f"  P/L at flat: ${df[flat_col].min():.2f} to ${df[flat_col].max():.2f}")

        # Find best/worst scenarios for logging
        pl_cols = [c for c in df.columns if c.startswith('PL_')]
        if pl_cols:
            logger.info(f"  Computed {len(pl_cols)} scenario P/L profiles for {len(df)} spreads")

        return df

    def _get_scenario_column_name(self, move_pct: float) -> str:
        """
        Generate a clean column name for a scenario.

        Examples:
            -0.15 -> 'down_15'
            -0.05 -> 'down_5'
            0 -> 'flat'
            0.03 -> 'up_3'
            0.10 -> 'up_10'
        """
        if move_pct == 0:
            return 'flat'
        elif move_pct < 0:
            return f'down_{abs(int(move_pct * 100))}'
        else:
            return f'up_{int(move_pct * 100)}'

    def get_scenario_results_for_spread(
        self,
        row: pd.Series,
        current_price: float,
        scenarios: List[float],
    ) -> List[ScenarioResult]:
        """
        Get ScenarioResult objects for a single spread (for visualization).

        Helper method to create ScenarioResult list from a DataFrame row
        that has already been processed by run_scenario_analysis().

        Args:
            row: A pandas Series (single row from candidates_df after Step 4)
            current_price: Current underlying price
            scenarios: List of percentage moves used in Step 4

        Returns:
            List of ScenarioResult objects for use in visualizations
        """
        results = []

        K1 = row['long_strike']
        K2 = row['short_strike']
        debit = row['net_debit']
        max_loss = row.get('max_loss', debit)

        for move_pct in scenarios:
            scenario_name = self._get_scenario_column_name(move_pct)
            pl_col = f'PL_{scenario_name}'
            ret_col = f'ret_{scenario_name}'

            # Get values from DataFrame if available, otherwise compute
            if pl_col in row.index:
                payoff = row[pl_col]
                return_pct = row[ret_col]
            else:
                # Compute on the fly
                S_T = current_price * (1 + move_pct)
                payoff = max(S_T - K1, 0) - max(S_T - K2, 0) - debit
                return_pct = payoff / debit if debit > 0 else 0

            # Create display name
            if move_pct < 0:
                display_name = f"Down {abs(move_pct)*100:.0f}%"
            elif move_pct == 0:
                display_name = "Flat"
            else:
                display_name = f"Up {move_pct*100:.0f}%"

            results.append(ScenarioResult(
                scenario_name=display_name,
                underlying_price=round(current_price * (1 + move_pct), 2),
                underlying_move_pct=move_pct,
                pnl_per_share=round(payoff, 4),
                pnl_per_contract=round(payoff * 100, 2),
                pnl_pct_of_max_loss=round(payoff / max_loss * 100, 2) if max_loss > 0 else 0,
                return_pct=round(return_pct * 100, 2),  # As percentage
            ))

        return results

    # =========================================================================
    # STEP 5: Scoring Function to Rank Bull Call Spreads
    # =========================================================================
    #
    # Workflow:
    #   Step a: Hard Filters (reject bad candidates)
    #   Step b: Simple Scoring Mode (rank by single metric)
    #   Step c: Composite Scoring Mode (quant-style weighted formula)
    #   Step d: Output ranked candidates with scores
    # =========================================================================

    def score_and_rank_spreads(
        self,
        candidates_df: pd.DataFrame,
        current_price: float,
        days_to_expiry: int,
        # Section A: Hard filter parameters
        budget: Optional[float] = None,              # Max acceptable max_loss
        max_be_move_pct: Optional[float] = 0.10,     # Reject if BE distance > pct (default 10%)
        rr_min: float = 1.5,                         # Minimum reward-to-risk ratio
        min_oi: int = 10,                            # Min open interest per leg
        min_volume: int = 0,                         # Min volume per leg (0 = no filter)
        max_bid_ask_pct: Optional[float] = None,     # Max bid-ask spread % (optional)
        # Scoring mode
        scoring_mode: str = 'composite',             # 'simple' or 'composite'
        # Simple mode parameters
        sort_by: str = 'reward_to_risk',             # Primary sort metric
        sort_ascending: bool = False,                # False = highest first
        # Composite mode parameters
        profile: str = 'both',                       # 'high_probability', 'convexity', 'both', or others
        weights: Optional[Dict[str, float]] = None,  # Custom weights override (only used if profile != 'both')
        # Risk-free rate for probability calculations
        risk_free_rate: float = 0.05,
    ) -> pd.DataFrame:
        """
        STEP 5: Score and rank bull call spread candidates.

        Computes a single normalized score (0-1) for each spread reflecting:
        - Ease of reaching breakeven
        - Profit potential (reward-to-risk)
        - Scenario performance (+3%, +5%, +10%)
        - Execution quality (liquidity + slippage)

        Section A - Hard Filters (reject before scoring):
        1. Risk Limits: max_loss > budget
        2. Breakeven Too Far: breakeven_distance > max_be_pct
        3. Minimum R:R: reward_to_risk < rr_min
        4. Liquidity: OI and volume must meet minimums
        5. Optional: bid-ask spread <= max_bid_ask_pct

        Section B - Normalized Component Scores (0-1):
        1. ease_score: Ease of breakeven (smooth penalty beyond threshold)
        2. profit_score: R:R ratio (capped at RR_CAP)
        3. scenario_score: Weighted upside performance
        4. liquidity_score: min(OI, volume) / target
        5. slippage_score: 1 / (1 + slippage * penalty)

        Section C - Final Composite Score:
        score = w1*ease + w2*profit + w3*scenario + w4*liquidity + w5*slippage

        Default weights (balanced): w1=0.35, w2=0.25, w3=0.20, w4=0.10, w5=0.10

        Args:
            candidates_df: DataFrame from Step 4 with scenario columns
            current_price: Current underlying price (S0)
            days_to_expiry: Days until expiration
            budget: Max acceptable max_loss (None = no limit)
            max_be_move_pct: Reject if breakeven_distance > pct (default 0.10)
            rr_min: Minimum reward-to-risk ratio
            min_oi: Minimum open interest per leg
            min_volume: Minimum volume per leg (0 = no filter)
            max_bid_ask_pct: Max bid-ask spread % (optional)
            scoring_mode: 'simple' or 'composite'
            sort_by: For simple mode, primary sort column
            sort_ascending: Sort direction for simple mode
            profile: For composite mode: 'high_probability', 'convexity', 'both', 'balanced', etc.
                     'both' ranks by both high_probability and convexity profiles
            weights: Custom weights dict (overrides profile, only used if profile != 'both')
            risk_free_rate: Risk-free rate for probability calculations

        Returns:
            DataFrame with added columns:
                - passed_filters: bool
                - filter_reason: str (why rejected, if applicable)
                - breakeven_distance: % move required to break even
                - ease_score, profit_score, scenario_score: component scores
                - liquidity_score, slippage_score: execution quality scores
                - score: final composite score (0-1)
                - rank: 1 = best, 2 = second best, etc.
        """
        logger.info("Step 5: Scoring and ranking spreads...")

        if candidates_df.empty:
            logger.warning("Empty candidates DataFrame, skipping scoring")
            return candidates_df

        df = candidates_df.copy()

        # Compute breakeven_distance for all candidates (needed for filters and scoring)
        df['breakeven_distance'] = (df['breakeven'] / current_price) - 1

        # =====================================================================
        # Section A: Hard Filters (Reject Before Scoring)
        # =====================================================================
        logger.debug("  Section A: Applying hard filters...")

        df['passed_filters'] = True
        df['filter_reason'] = ''

        # Filter 1: Risk Limits - max_loss > budget
        if budget is not None:
            failed_budget = df['max_loss'] > budget
            df.loc[failed_budget, 'passed_filters'] = False
            df.loc[failed_budget, 'filter_reason'] = f'max_loss > budget (${budget})'
            logger.debug(f"    Budget filter: {failed_budget.sum()} rejected")

        # Filter 2: Breakeven Too Far - breakeven_distance > max_be_pct
        if max_be_move_pct is not None:
            failed_be = df['breakeven_distance'] > max_be_move_pct
            df.loc[failed_be & df['passed_filters'], 'filter_reason'] = \
                f'breakeven_distance > {max_be_move_pct*100:.0f}%'
            df.loc[failed_be, 'passed_filters'] = False
            logger.debug(f"    Breakeven filter: {failed_be.sum()} rejected")

        # Filter 3: Minimum Reward-to-Risk
        failed_rr = df['reward_to_risk'] < rr_min
        df.loc[failed_rr & df['passed_filters'], 'filter_reason'] = f'R:R < {rr_min}'
        df.loc[failed_rr, 'passed_filters'] = False
        logger.debug(f"    R:R filter: {failed_rr.sum()} rejected")

        # Filter 4: Liquidity Requirements - OI
        if min_oi > 0:
            failed_oi = (df['long_oi'] < min_oi) | (df['short_oi'] < min_oi)
            df.loc[failed_oi & df['passed_filters'], 'filter_reason'] = f'OI < {min_oi}'
            df.loc[failed_oi, 'passed_filters'] = False
            logger.debug(f"    OI filter: {failed_oi.sum()} rejected")

        # Filter 4: Liquidity Requirements - Volume
        if min_volume > 0:
            failed_vol = (df['long_volume'] < min_volume) | (df['short_volume'] < min_volume)
            df.loc[failed_vol & df['passed_filters'], 'filter_reason'] = f'Volume < {min_volume}'
            df.loc[failed_vol, 'passed_filters'] = False
            logger.debug(f"    Volume filter: {failed_vol.sum()} rejected")

        # Filter 5: Optional Execution Filter - Bid-Ask Spread
        if max_bid_ask_pct is not None:
            # Compute bid-ask spread percentage for each leg
            long_mid = (df['long_bid'] + df['long_ask']) / 2
            short_mid = (df['short_bid'] + df['short_ask']) / 2
            long_spread_pct = (df['long_ask'] - df['long_bid']) / long_mid.replace(0, np.nan)
            short_spread_pct = (df['short_ask'] - df['short_bid']) / short_mid.replace(0, np.nan)
            max_spread = np.maximum(long_spread_pct.fillna(1), short_spread_pct.fillna(1))
            failed_spread = max_spread > max_bid_ask_pct
            df.loc[failed_spread & df['passed_filters'], 'filter_reason'] = \
                f'bid-ask spread > {max_bid_ask_pct*100:.0f}%'
            df.loc[failed_spread, 'passed_filters'] = False
            logger.debug(f"    Bid-ask filter: {failed_spread.sum()} rejected")

        passed_count = df['passed_filters'].sum()
        total_count = len(df)
        logger.info(f"  Section A: {passed_count}/{total_count} candidates passed hard filters")

        if passed_count == 0:
            logger.warning("No candidates passed hard filters")
            df['prob_profit'] = None
            df['prob_max_profit'] = None
            df['expected_value'] = None
            df['ease_score'] = 0.0
            df['profit_score'] = 0.0
            df['scenario_score'] = 0.0
            df['liquidity_score'] = 0.0
            df['slippage_score'] = 0.0
            df['score'] = 0.0
            df['rank'] = None
            return df

        # =====================================================================
        # Compute Probability Metrics (needed for scoring)
        # =====================================================================
        logger.debug("  Computing probability metrics...")

        # Initialize probability columns
        df['prob_profit'] = np.nan
        df['prob_max_profit'] = np.nan
        df['expected_value'] = np.nan

        # Compute average IV for each spread (vectorized)
        long_iv = df['long_iv'].fillna(0)
        short_iv = df['short_iv'].fillna(0)
        has_both_iv = (long_iv > 0) & (short_iv > 0)

        df['avg_iv'] = np.where(
            has_both_iv,
            (long_iv + short_iv) / 2,
            DEFAULT_IV
        )
        # Ensure IV is in decimal form (convert if > 1)
        df['avg_iv'] = np.where(df['avg_iv'] > 1, df['avg_iv'] / 100, df['avg_iv'])

        # Time to expiry in years
        t = max(days_to_expiry / 365.0, MIN_DAYS_TO_EXPIRY)

        # Compute probabilities for passed candidates only
        passed_idx = df[df['passed_filters']].index

        for idx in passed_idx:
            row = df.loc[idx]
            iv = row['avg_iv']

            # Log-normal distribution parameters
            drift = (risk_free_rate - 0.5 * iv**2) * t
            vol = iv * np.sqrt(t)

            if vol > 0:
                # P(S_T > breakeven) - probability of any profit
                d_be = (np.log(row['breakeven'] / current_price) - drift) / vol
                prob_profit = 1 - norm.cdf(d_be)

                # P(S_T > short_strike) - probability of max profit
                d_mp = (np.log(row['short_strike'] / current_price) - drift) / vol
                prob_max_profit = 1 - norm.cdf(d_mp)
            else:
                # Edge case: zero volatility
                prob_profit = 1.0 if current_price > row['breakeven'] else 0.0
                prob_max_profit = 1.0 if current_price > row['short_strike'] else 0.0

            # Expected value calculation
            # EV = P(max) * max_profit + P(partial) * avg_partial - P(loss) * max_loss
            prob_loss = 1 - prob_profit
            prob_partial = prob_profit - prob_max_profit
            avg_partial = row['max_profit'] / 2  # Simplified: assume uniform dist in partial zone

            ev = (
                prob_max_profit * row['max_profit'] +
                prob_partial * avg_partial -
                prob_loss * row['max_loss']
            )

            df.loc[idx, 'prob_profit'] = round(prob_profit, 4)
            df.loc[idx, 'prob_max_profit'] = round(prob_max_profit, 4)
            df.loc[idx, 'expected_value'] = round(ev, 4)

        # =====================================================================
        # Step 5b: Simple Scoring Mode
        # =====================================================================
        if scoring_mode == 'simple':
            logger.debug(f"  Step 5b: Simple scoring mode (sort by {sort_by})")

            # For simple mode, score = the primary metric value
            if sort_by in df.columns:
                df['score'] = df[sort_by].fillna(0)
            else:
                logger.warning(f"Sort column '{sort_by}' not found, using reward_to_risk")
                df['score'] = df['reward_to_risk']

            # Rank only passed candidates
            passed_df = df[df['passed_filters']].copy()
            passed_df = passed_df.sort_values(
                by=[sort_by, 'max_loss'],  # Secondary sort by lower max_loss
                ascending=[sort_ascending, True]
            )
            passed_df['rank'] = range(1, len(passed_df) + 1)

            # Merge ranks back
            df['rank'] = None
            df.loc[passed_df.index, 'rank'] = passed_df['rank']

            logger.info(f"  Step 5b: Ranked {len(passed_df)} spreads by {sort_by}")

        # =====================================================================
        # Section B & C: Composite Scoring Mode (Normalized 0-1 Score)
        # =====================================================================
        # Objective: Compute a single normalized score (0-1) reflecting:
        #   - Ease of reaching breakeven
        #   - Profit potential (R:R)
        #   - Scenario performance (+3%, +5%, +10% moves)
        #   - Execution quality (liquidity + slippage)
        # =====================================================================
        else:  # scoring_mode == 'composite'
            logger.debug(f"  Section B/C: Composite scoring mode (profile: {profile})")

            # Determine which profiles to use
            if profile == 'both':
                profiles_to_score = ['high_probability', 'convexity']
            else:
                profiles_to_score = [profile]

            # Initialize component score columns (shared across profiles)
            df['ease_score'] = 0.0
            df['profit_score'] = 0.0
            df['scenario_score'] = 0.0
            df['convexity_score'] = 0.0
            df['liquidity_score'] = 0.0
            df['slippage_score'] = 0.0

            # Get passed candidates mask
            passed_mask = df['passed_filters'] & (df['max_loss'] > 0)

            # First, compute all component scores (same for all profiles)
            for idx in df[passed_mask].index:
                row = df.loc[idx]

                # Step 1: Ease of Breakeven Score
                be_distance = row['breakeven_distance']
                ease = compute_ease_score(be_distance)

                # Step 2: Profit Score (R:R based)
                profit = compute_profit_score(row['max_profit'], row['max_loss'])

                # Step 3: Scenario Score (near-term moves: +2%, +3%, +5%)
                scenario = compute_scenario_score(
                    pl_up_2=row.get('PL_up_2', 0) or 0,
                    pl_up_3=row.get('PL_up_3', 0) or 0,
                    pl_up_5=row.get('PL_up_5', 0) or 0,
                    max_loss=row['max_loss'],
                )

                # Step 4: Convexity Score (tail moves: +7%, +10% vs +2%, +3%)
                convexity = compute_convexity_score(
                    pl_up_2=row.get('PL_up_2', 0) or 0,
                    pl_up_3=row.get('PL_up_3', 0) or 0,
                    pl_up_7=row.get('PL_up_7', 0) or 0,
                    pl_up_10=row.get('PL_up_10', 0) or 0,
                    max_loss=row['max_loss'],
                )

                # Step 5: Liquidity Score
                liquidity = compute_liquidity_score(
                    long_oi=row.get('long_oi', 0) or 0,
                    short_oi=row.get('short_oi', 0) or 0,
                    long_volume=row.get('long_volume', 0) or 0,
                    short_volume=row.get('short_volume', 0) or 0,
                )

                # Step 6: Slippage Score
                slippage = compute_slippage_score(
                    long_bid=row.get('long_bid', 0) or 0,
                    long_ask=row.get('long_ask', 0) or 0,
                    short_bid=row.get('short_bid', 0) or 0,
                    short_ask=row.get('short_ask', 0) or 0,
                )

                # Store all component scores
                df.loc[idx, 'ease_score'] = round(ease, 4)
                df.loc[idx, 'profit_score'] = round(profit, 4)
                df.loc[idx, 'scenario_score'] = round(scenario, 4)
                df.loc[idx, 'convexity_score'] = round(convexity, 4)
                df.loc[idx, 'liquidity_score'] = round(liquidity, 4)
                df.loc[idx, 'slippage_score'] = round(slippage, 4)

            # Now compute composite scores and ranks for each profile
            for prof in profiles_to_score:
                # Get weights for this profile
                if weights is not None and profile != 'both':
                    prof_weights = weights
                else:
                    prof_weights = SCORING_PROFILES.get(prof, SCORING_PROFILES['balanced'])

                logger.debug(f"    Profile '{prof}' weights: {prof_weights}")

                # Column names for this profile
                if profile == 'both':
                    score_col = f'score_{prof}'
                    rank_col = f'rank_{prof}'
                else:
                    score_col = 'score'
                    rank_col = 'rank'

                df[score_col] = 0.0

                # Compute composite score for each passed candidate
                for idx in df[passed_mask].index:
                    final = compute_composite_score(
                        ease_score=df.loc[idx, 'ease_score'],
                        profit_score=df.loc[idx, 'profit_score'],
                        scenario_score=df.loc[idx, 'scenario_score'],
                        convexity_score=df.loc[idx, 'convexity_score'],
                        liquidity_score=df.loc[idx, 'liquidity_score'],
                        slippage_score=df.loc[idx, 'slippage_score'],
                        weights=prof_weights,
                    )
                    df.loc[idx, score_col] = round(final, 4)

                # Section D: Rank spreads by score (descending), with tiebreakers
                passed_df = df[df['passed_filters']].copy()
                passed_df = passed_df.sort_values(
                    by=[score_col, 'ease_score', 'profit_score'],
                    ascending=[False, False, False]
                )
                passed_df[rank_col] = range(1, len(passed_df) + 1)

                # Merge ranks back
                df[rank_col] = None
                df.loc[passed_df.index, rank_col] = passed_df[rank_col]

                logger.info(f"  Section D: Scored and ranked {len(passed_df)} spreads (profile: {prof})")

        # Clean up temporary column
        if 'avg_iv' in df.columns:
            df = df.drop(columns=['avg_iv'])

        return df

    def get_score_explanation(
        self,
        row: pd.Series,
        current_price: float,
    ) -> str:
        """
        Generate a brief explanation for why a spread ranks high or low.

        Args:
            row: DataFrame row with spread metrics and score
            current_price: Current underlying price

        Returns:
            Human-readable explanation string
        """
        explanations = []

        # Breakeven ease assessment
        be_dist = row.get('breakeven_distance', 0) * 100
        ease_score = row.get('ease_score', 0)
        if be_dist <= 2:
            explanations.append(f"easy BE (+{be_dist:.1f}%)")
        elif be_dist <= 5:
            explanations.append(f"moderate BE (+{be_dist:.1f}%)")
        else:
            explanations.append(f"hard BE (+{be_dist:.1f}%)")

        # Profit score assessment
        profit_score = row.get('profit_score', 0)
        rr = row.get('reward_to_risk', 0)
        if profit_score >= 0.75:
            explanations.append(f"excellent R:R ({rr:.1f}x)")
        elif profit_score >= 0.5:
            explanations.append(f"good R:R ({rr:.1f}x)")
        elif profit_score >= 0.33:
            explanations.append(f"moderate R:R ({rr:.1f}x)")
        else:
            explanations.append(f"low R:R ({rr:.1f}x)")

        # Liquidity assessment
        liquidity_score = row.get('liquidity_score', 0)
        if liquidity_score >= 0.8:
            explanations.append("high liquidity")
        elif liquidity_score >= 0.5:
            explanations.append("moderate liquidity")
        elif liquidity_score > 0:
            explanations.append("low liquidity")

        # Slippage assessment
        slippage_score = row.get('slippage_score', 0)
        if slippage_score >= 0.8:
            explanations.append("tight spreads")
        elif slippage_score < 0.5:
            explanations.append("wide spreads")

        # Final score summary
        final_score = row.get('score', 0)
        explanations.append(f"score={final_score:.2f}")

        return ", ".join(explanations)

    # =========================================================================
    # STEP 6: Select and Output the "Best" Bull Call Spread Combinations
    # =========================================================================
    #
    # Workflow:
    #   Step 1: Apply Final Filters (tradability - risk/size + liquidity)
    #   Step 2: Sort by Score (or chosen metric)
    #   Step 3: Select the Final "Best" Set (top N)
    #   Step 4: Format a Clean Output Table
    # =========================================================================

    def select_best_spreads(
        self,
        candidates_df: pd.DataFrame,
        current_price: float,
        ticker: str,
        expiry: str,
        # Step 1: Final filter parameters
        max_risk_per_trade: Optional[float] = None,    # Max acceptable max_loss
        min_leg_oi: int = 10,                          # Min open interest per leg
        min_leg_volume: int = 0,                       # Min volume per leg
        # Step 2: Sorting parameters
        sort_metric: str = 'score',                    # 'score', 'reward_to_risk'
        # Step 3: Selection parameters
        top_n: int = 10,                               # Number of best spreads to select
    ) -> Tuple[pd.DataFrame, str]:
        """
        STEP 6: Select and output the "best" bull call spread combinations.

        Produces a ranked, tradable shortlist of the best spreads.

        Workflow:
        1) Apply Final Filters (Tradability):
           - Risk/size constraints: max_loss <= max_risk_per_trade
           - Liquidity constraints: min OI and volume per leg

        2) Sort by Score:
           - Uses composite score from Step 5 (5-component weighted score)
           - score = w1×ease + w2×profit + w3×scenario + w4×liquidity + w5×slippage

        3) Select Final "Best" Set:
           - Take top N spreads from sorted list

        4) Format Clean Output Table:
           - ticker, expiry, K_long, K_short, width, debit, max_loss, max_profit,
             RR, breakeven, PL_up_5, PL_up_10, long_oi, short_oi, score
           - Include textual explanation for top spreads

        Args:
            candidates_df: DataFrame from Step 5 with scores and ranks
            current_price: Current underlying price (S0)
            ticker: Stock ticker
            expiry: Expiration date
            max_risk_per_trade: Max acceptable max_loss (None = use existing)
            min_leg_oi: Minimum open interest per leg
            min_leg_volume: Minimum volume per leg
            sort_metric: Metric to sort by ('score', 'reward_to_risk')
            top_n: Number of best spreads to select

        Returns:
            Tuple of:
                - DataFrame with final selected spreads (clean output table)
                - String with textual summary/explanation
        """
        logger.info("Step 6: Selecting best spread combinations...")

        if candidates_df.empty:
            logger.warning("Empty candidates DataFrame, no spreads to select")
            return pd.DataFrame(), "No spread candidates available."

        df = candidates_df.copy()

        # =====================================================================
        # Step 6.1: Apply Final Filters (Tradability)
        # =====================================================================
        logger.debug("  Step 6.1: Applying tradability filters...")

        df['tradable'] = True
        df['tradability_reason'] = ''

        # Start with candidates that passed Step 5 filters
        if 'passed_filters' in df.columns:
            df.loc[~df['passed_filters'], 'tradable'] = False
            df.loc[~df['passed_filters'], 'tradability_reason'] = 'failed Step 5 filters'

        # Filter 1: Risk/size constraints
        if max_risk_per_trade is not None:
            failed_risk = df['max_loss'] > max_risk_per_trade
            df.loc[failed_risk & df['tradable'], 'tradability_reason'] = \
                f'max_loss > ${max_risk_per_trade:.2f}'
            df.loc[failed_risk, 'tradable'] = False
            logger.debug(f"    Risk filter: {failed_risk.sum()} rejected")

        # Filter 2: Liquidity - Open Interest
        if min_leg_oi > 0:
            failed_oi = (df['long_oi'] < min_leg_oi) | (df['short_oi'] < min_leg_oi)
            df.loc[failed_oi & df['tradable'], 'tradability_reason'] = \
                f'OI < {min_leg_oi}'
            df.loc[failed_oi, 'tradable'] = False
            logger.debug(f"    OI filter: {failed_oi.sum()} rejected")

        # Filter 3: Liquidity - Volume
        if min_leg_volume > 0:
            failed_vol = (df['long_volume'] < min_leg_volume) | (df['short_volume'] < min_leg_volume)
            df.loc[failed_vol & df['tradable'], 'tradability_reason'] = \
                f'Volume < {min_leg_volume}'
            df.loc[failed_vol, 'tradable'] = False
            logger.debug(f"    Volume filter: {failed_vol.sum()} rejected")

        tradable_count = df['tradable'].sum()
        total_count = len(df)
        logger.info(f"  Step 6.1: {tradable_count}/{total_count} candidates are tradable")

        if tradable_count == 0:
            logger.warning("No tradable candidates after final filters")
            return pd.DataFrame(), "No tradable spread candidates after applying filters."

        # =====================================================================
        # Step 6.2: Sort by Score
        # =====================================================================
        logger.debug("  Step 6.2: Sorting by score...")

        # Ensure breakeven_distance exists (should be computed in Step 5)
        if 'breakeven_distance' not in df.columns:
            df['breakeven_distance'] = (df['breakeven'] / current_price) - 1

        # Get tradable candidates and sort by composite score (from Step 5)
        tradable_df = df[df['tradable']].copy()

        if sort_metric == 'score':
            # Sort by score (descending), then ease_score, then profit_score as tiebreakers
            tradable_df = tradable_df.sort_values(
                by=['score', 'ease_score', 'profit_score'],
                ascending=[False, False, False]
            )
        else:
            tradable_df = tradable_df.sort_values(sort_metric, ascending=False)

        # Assign final rank
        tradable_df['final_rank'] = range(1, len(tradable_df) + 1)

        logger.info(f"  Step 6.2: Sorted {len(tradable_df)} tradable spreads by {sort_metric}")

        # =====================================================================
        # Step 6.3: Select the Final "Best" Set
        # =====================================================================
        logger.debug(f"  Step 6.3: Selecting top {top_n} spreads...")

        best_df = tradable_df.head(top_n).copy()
        logger.info(f"  Step 6.3: Selected {len(best_df)} best spreads")

        # =====================================================================
        # Step 6.4: Format Clean Output Table (Section D)
        # =====================================================================
        # Output Fields: ticker, expiry, K_long, K_short, width, debit,
        #   max_loss, max_profit, RR, breakeven, breakeven_distance,
        #   ease_score, profit_score, scenario_score, liquidity_score,
        #   slippage_score, score (final), PL_up_2, PL_up_3, PL_up_5, PL_up_7, PL_up_10
        # =====================================================================
        logger.debug("  Step 6.4: Formatting output table...")

        # Build clean output DataFrame with all required columns
        output_df = pd.DataFrame({
            'ticker': ticker.upper(),
            'expiry': expiry,
            'K_long': best_df['long_strike'],
            'K_short': best_df['short_strike'],
            'width': best_df['width'],
            'debit': best_df['net_debit'],
            'max_loss': best_df['max_loss'],
            'max_profit': best_df['max_profit'],
            'RR': best_df['reward_to_risk'],
            'breakeven': best_df['breakeven'],
            'breakeven_distance': best_df['breakeven_distance'],
            # Component scores
            'ease_score': best_df.get('ease_score', 0),
            'profit_score': best_df.get('profit_score', 0),
            'scenario_score': best_df.get('scenario_score', 0),
            'liquidity_score': best_df.get('liquidity_score', 0),
            'slippage_score': best_df.get('slippage_score', 0),
            # Final score
            'score': best_df['score'],
            # Scenario P/L
            'PL_up_2': best_df.get('PL_up_2', 0),
            'PL_up_3': best_df.get('PL_up_3', 0),
            'PL_up_5': best_df.get('PL_up_5', 0),
            'PL_up_7': best_df.get('PL_up_7', 0),
            'PL_up_10': best_df.get('PL_up_10', 0),
        })

        # Reset index for clean output
        output_df = output_df.reset_index(drop=True)

        # =====================================================================
        # Generate Textual Summary
        # =====================================================================
        summary_lines = []
        summary_lines.append(f"\n{'='*100}")
        summary_lines.append(f"{'BEST BULL CALL SPREADS - FINAL SELECTION':^100}")
        summary_lines.append(f"{ticker.upper()} | Expiry: {expiry} | Current Price: ${current_price:.2f}")
        summary_lines.append(f"{'='*100}")
        summary_lines.append(f"\nSelected {len(best_df)} best tradable spreads from {tradable_count} candidates.\n")

        # Top 3 explanations
        summary_lines.append("TOP SPREAD ANALYSIS:")
        summary_lines.append("-" * 80)

        for i, (_, row) in enumerate(best_df.head(3).iterrows(), 1):
            explanation = self.get_score_explanation(row, current_price)
            be_dist_pct = row['breakeven_distance'] * 100

            summary_lines.append(
                f"\n#{i}: ${row['long_strike']:.0f}/${row['short_strike']:.0f} spread"
            )
            summary_lines.append(
                f"   • Debit: ${row['net_debit']:.2f} | Max Profit: ${row['max_profit']:.2f} | R:R: {row['reward_to_risk']:.2f}"
            )
            summary_lines.append(
                f"   • Breakeven: ${row['breakeven']:.2f} ({be_dist_pct:.1f}% above spot)"
            )
            summary_lines.append(
                f"   • Liquidity: Long OI={row['long_oi']:.0f}, Short OI={row['short_oi']:.0f}"
            )
            summary_lines.append(
                f"   • Assessment: {explanation}"
            )

        summary_lines.append("\n" + "-" * 80)
        summary_lines.append("\nScoring Formula (5-component, 0-1 normalized):")
        summary_lines.append("  score = 0.40×ease + 0.30×profit + 0.10×scenario + 0.15×liquidity + 0.05×slippage")
        summary_lines.append("  • ease_score: 1/BE_dist with exponential penalty if BE_dist > 5%")
        summary_lines.append("  • profit_score: min(R:R, 10) / (min(R:R, 10) + 1)")
        summary_lines.append("  • scenario_score: weighted P/L at +3%, +5%, +10% (capped at 10x debit)")
        summary_lines.append("  • liquidity_score: min(OI, volume) / 200 (capped at 1.0)")
        summary_lines.append("  • slippage_score: 1 / (1 + max_slippage × 5)")

        summary_text = "\n".join(summary_lines)

        return output_df, summary_text

    def print_best_spreads_table(
        self,
        output_df: pd.DataFrame,
        summary_text: str,
    ) -> None:
        """
        Print the best spreads table in a clean, readable format.

        Output columns: ticker, expiry, K_long, K_short, width, debit,
                        max_loss, max_profit, RR, breakeven, breakeven_distance,
                        ease_score, profit_score, scenario_score, liquidity_score,
                        slippage_score, score, PL_up_2, PL_up_3, PL_up_5, PL_up_7, PL_up_10

        Args:
            output_df: DataFrame from select_best_spreads()
            summary_text: Summary text from select_best_spreads()
        """
        if output_df.empty:
            print("No spreads to display.")
            return

        # Print summary
        print(summary_text)

        # Print main table
        print(f"\n{'BEST BULL CALL SPREADS - CORE METRICS':^120}")
        print("=" * 120)

        # Core metrics header
        header1 = (
            f"{'#':<3} {'Ticker':<6} {'Expiry':<12} {'K_long':>7} {'K_short':>8} "
            f"{'Width':>6} {'Debit':>7} {'MaxLoss':>8} {'MaxProf':>8} "
            f"{'R:R':>5} {'B/E':>8} {'BE_Dist':>8} {'Score':>6}"
        )
        print(header1)
        print("-" * 120)

        # Print core metrics rows
        for i, (_, row) in enumerate(output_df.iterrows(), 1):
            be_dist_pct = row.get('breakeven_distance', 0) * 100
            line = (
                f"{i:<3} {row['ticker']:<6} {row['expiry']:<12} "
                f"${row['K_long']:>5.0f} ${row['K_short']:>6.0f} "
                f"${row['width']:>4.0f} ${row['debit']:>5.2f} "
                f"${row['max_loss']:>6.2f} ${row['max_profit']:>6.2f} "
                f"{row['RR']:>4.2f} ${row['breakeven']:>6.2f} "
                f"{be_dist_pct:>6.1f}% {row['score']:>5.3f}"
            )
            print(line)

        print("=" * 120)

        # Print component scores table
        print(f"\n{'COMPONENT SCORES (0-1 Normalized)':^130}")
        print("=" * 130)

        header2 = (
            f"{'#':<3} {'K_long':>7} {'K_short':>8} "
            f"{'Ease':>6} {'Profit':>7} {'Scen':>6} {'Liq':>6} {'Slip':>6} "
            f"{'Score':>6} {'PL+2%':>7} {'PL+3%':>7} {'PL+5%':>7} {'PL+7%':>7} {'PL+10%':>8}"
        )
        print(header2)
        print("-" * 130)

        for i, (_, row) in enumerate(output_df.iterrows(), 1):
            line = (
                f"{i:<3} ${row['K_long']:>5.0f} ${row['K_short']:>6.0f} "
                f"{row.get('ease_score', 0):>5.3f} {row.get('profit_score', 0):>6.3f} "
                f"{row.get('scenario_score', 0):>5.3f} {row.get('liquidity_score', 0):>5.3f} "
                f"{row.get('slippage_score', 0):>5.3f} {row['score']:>5.3f} "
                f"${row.get('PL_up_2', 0):>5.2f} ${row.get('PL_up_3', 0):>5.2f} "
                f"${row.get('PL_up_5', 0):>5.2f} ${row.get('PL_up_7', 0):>5.2f} "
                f"${row.get('PL_up_10', 0):>6.2f}"
            )
            print(line)

        print("=" * 130)
        print("\nNote: All P/L figures are per share. Multiply by 100 for per-contract values.")

    # =========================================================================
    # Helper: Probability Estimation (for backward compatibility)
    # =========================================================================

    def estimate_probabilities(
        self,
        spread: SpreadCandidate,
        current_price: float,
        days_to_expiry: int,
        iv: Optional[float] = None,
        risk_free_rate: float = 0.05,
    ) -> Tuple[float, float, float]:
        """
        Estimate probability of profit, probability of max profit, and expected value.

        Uses log-normal distribution assumption.

        Args:
            spread: The spread candidate
            current_price: Current underlying price
            days_to_expiry: Days until expiration
            iv: Implied volatility (annualized, as decimal, e.g., 0.25 for 25%)
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            (prob_profit, prob_max_profit, expected_value)
        """
        # Use average IV if not provided
        if iv is None:
            if spread.long_iv and spread.short_iv:
                iv = (spread.long_iv + spread.short_iv) / 2
            else:
                iv = 0.25  # Default to 25% if no IV available

        # Ensure IV is in decimal form
        if iv > 1:
            iv = iv / 100

        t = days_to_expiry / 365.0
        if t <= 0:
            t = 1/365  # Minimum 1 day

        # Log-normal parameters
        drift = (risk_free_rate - 0.5 * iv**2) * t
        vol = iv * np.sqrt(t)

        # Probability of profit: P(S > breakeven)
        if vol > 0:
            d_breakeven = (np.log(spread.breakeven / current_price) - drift) / vol
            prob_profit = 1 - norm.cdf(d_breakeven)

            # Probability of max profit: P(S > short_strike)
            d_max_profit = (np.log(spread.short_strike / current_price) - drift) / vol
            prob_max_profit = 1 - norm.cdf(d_max_profit)
        else:
            prob_profit = 1.0 if current_price > spread.breakeven else 0.0
            prob_max_profit = 1.0 if current_price > spread.short_strike else 0.0

        # Expected value calculation
        prob_loss = 1 - prob_profit
        prob_partial = prob_profit - prob_max_profit
        avg_partial_profit = spread.max_profit / 2

        expected_value = (
            prob_max_profit * spread.max_profit +
            prob_partial * avg_partial_profit +
            prob_loss * (-spread.max_loss)
        )

        return (
            round(prob_profit, 4),
            round(prob_max_profit, 4),
            round(expected_value, 4)
        )

    # =========================================================================
    # MAIN SCAN FUNCTION
    # =========================================================================

    def scan(
        self,
        ticker: str,
        expiry: str,
        scenarios: List[float] = None,  # Uses DEFAULT_SCENARIOS if None
        # Step 2 parameters
        max_debit_per_spread: float = 10.0,
        min_width: float = 5.0,
        max_width: float = 20.0,
        min_open_interest: int = 10,
        min_volume: int = 0,
        strike_range_pct: float = 0.10,
        min_reward_risk: float = 0.5,
        max_breakeven_pct: Optional[float] = None,
        # Step 5 parameters
        scoring_mode: str = 'composite',
        scoring_profile: str = 'balanced',
        scoring_weights: Optional[Dict[str, float]] = None,
        # Step 6 parameters
        max_risk_per_trade: Optional[float] = None,
        best_n: int = 10,
        # Output control
        top_n: int = 20,
    ) -> Tuple[List[SpreadCandidate], float, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        """
        Main scanning function - orchestrates all steps (1-6).

        Args:
            ticker: Stock ticker (e.g., 'QQQ')
            expiry: Expiration date (YYYY-MM-DD)
            scenarios: List of % moves to analyze
                Default: [-0.03, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
                (9 scenarios: down 3%, flat, up 1%, up 2%, up 3%, up 4%, up 5%, up 7%, up 10%)

            # Step 2 Parameters
            max_debit_per_spread: Maximum debit (max loss) per spread in $
            min_width: Minimum spread width in dollars
            max_width: Maximum spread width in dollars
            min_open_interest: Minimum open interest per leg
            min_volume: Minimum volume per leg
            strike_range_pct: Filter strikes to ±X% of spot
            min_reward_risk: Minimum reward-to-risk ratio
            max_breakeven_pct: Reject if BE > spot * (1 + pct)

            # Step 5 Parameters
            scoring_mode: 'simple' or 'composite'
            scoring_profile: 'aggressive', 'conservative', or 'balanced'
            scoring_weights: Custom weights dict (overrides profile)

            # Step 6 Parameters
            max_risk_per_trade: Max risk filter for final selection (defaults to max_debit_per_spread)
            best_n: Number of best spreads to select in Step 6

            # Output control
            top_n: Number of top spreads to return from Step 5

        Returns:
            Tuple of:
                - list of SpreadCandidate (for visualization)
                - current_price
                - calls_df (raw calls chain)
                - spreads_df (all candidates with scores)
                - best_spreads_df (Step 6 final selection table)
                - best_spreads_summary (Step 6 textual summary)
        """
        logger.info(f"Scanning {ticker} for {expiry} expiry...")

        # Use default scenarios if not provided
        if scenarios is None:
            scenarios = DEFAULT_SCENARIOS

        # =====================================================================
        # STEP 1: Pull clean option chains (calls) from OpenBB
        # =====================================================================
        current_price = self.get_current_price(ticker)
        logger.info(f"Current {ticker} price: ${current_price:.2f}")

        calls_df = self.get_calls_chain(ticker, expiry)

        # Calculate days to expiry
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
        today = date.today()
        days_to_expiry = (expiry_date - today).days
        logger.info(f"Days to expiry: {days_to_expiry}")

        # =====================================================================
        # STEP 2: Generate all valid bull call spread candidates (DataFrame)
        # =====================================================================
        spreads_df = self.generate_spread_candidates(
            calls_df=calls_df,
            underlying_price=current_price,
            max_debit_per_spread=max_debit_per_spread,
            min_width=min_width,
            max_width=max_width,
            min_open_interest=min_open_interest,
            min_volume=min_volume,
            strike_range_pct=strike_range_pct,
            min_reward_to_risk=min_reward_risk,
            max_breakeven_pct=max_breakeven_pct,
        )

        if spreads_df.empty:
            logger.warning("No spread candidates generated")
            return [], current_price, calls_df, spreads_df, pd.DataFrame(), "No spread candidates generated."

        # =====================================================================
        # STEP 3: Compute Basic Payoff Metrics for Each Spread
        # =====================================================================
        # Ensures payoff columns are present and properly computed:
        #   max_loss = debit
        #   max_profit = width - debit
        #   reward_to_risk = max_profit / max_loss
        #   breakeven = K1 + debit
        spreads_df = self.compute_payoff_metrics(spreads_df)

        # =====================================================================
        # STEP 4: Scenario Analysis - P/L at different underlying prices
        # =====================================================================
        # Computes for each spread and scenario:
        #   payoff = max(S_T - K1, 0) - max(S_T - K2, 0) - debit
        #   return = payoff / debit
        # Adds columns: PL_down_5, PL_flat, PL_up_5, ret_down_5, ret_flat, etc.
        spreads_df = self.run_scenario_analysis(spreads_df, current_price, scenarios)

        # =====================================================================
        # STEP 5: Score and Rank Spreads
        # =====================================================================
        # a) Hard filters: budget, breakeven, R:R minimum
        # b) Simple mode: rank by single metric
        # c) Composite mode: weighted scoring formula
        # Adds columns: passed_filters, filter_reason, prob_profit, prob_max_profit,
        #               expected_value, score, rank
        spreads_df = self.score_and_rank_spreads(
            candidates_df=spreads_df,
            current_price=current_price,
            days_to_expiry=days_to_expiry,
            budget=max_debit_per_spread,
            max_be_move_pct=max_breakeven_pct,
            rr_min=min_reward_risk,
            scoring_mode=scoring_mode,
            profile=scoring_profile,
            weights=scoring_weights,
        )

        # =====================================================================
        # STEP 6: Select and Output the "Best" Spread Combinations
        # =====================================================================
        # 1) Apply final tradability filters (risk/size + liquidity)
        # 2) Sort by composite score from Step 5
        # 3) Select top N best spreads
        # 4) Format clean output table
        best_spreads_df, best_spreads_summary = self.select_best_spreads(
            candidates_df=spreads_df,
            current_price=current_price,
            ticker=ticker,
            expiry=expiry,
            max_risk_per_trade=max_risk_per_trade if max_risk_per_trade else max_debit_per_spread,
            min_leg_oi=min_open_interest,
            min_leg_volume=min_volume,
            top_n=best_n,
        )

        # =====================================================================
        # Convert to SpreadCandidate objects for visualization compatibility
        # =====================================================================
        processed_spreads = []

        # Only process passed candidates, sorted by rank
        passed_df = spreads_df[spreads_df['passed_filters'] == True].copy()
        passed_df = passed_df.sort_values('rank')

        for _, row in passed_df.head(top_n).iterrows():
            # Convert DataFrame row to SpreadCandidate object
            spread = self.create_spread_candidate(row)

            # Attach scenario results for visualization
            spread.scenarios = self.get_scenario_results_for_spread(row, current_price, scenarios)

            # Attach probability metrics from Step 5
            spread.prob_profit = row.get('prob_profit')
            spread.prob_max_profit = row.get('prob_max_profit')
            spread.expected_value = row.get('expected_value')

            # Attach score
            spread.composite_score = row.get('score', 0)

            processed_spreads.append(spread)

        logger.info(f"Processed {len(processed_spreads)} spreads meeting all criteria")

        # Return includes the new best_spreads outputs from Step 6
        return processed_spreads, current_price, calls_df, spreads_df, best_spreads_df, best_spreads_summary

    # =========================================================================
    # STEP 7: Visualization
    # =========================================================================

    def calculate_spread_pnl(
        self,
        spread: SpreadCandidate,
        underlying_price: float,
    ) -> float:
        """
        Calculate spread P/L at a given underlying price (at expiry).

        Used by visualization functions to generate payoff curves.

        Formula: payoff = max(S_T - K1, 0) - max(S_T - K2, 0) - debit

        Args:
            spread: SpreadCandidate object
            underlying_price: Price of underlying at expiry (S_T)

        Returns:
            P/L per share at the given underlying price
        """
        K1 = spread.long_strike
        K2 = spread.short_strike
        debit = spread.net_debit

        # payoff = max(S_T - K1, 0) - max(S_T - K2, 0) - debit
        long_call_payoff = max(underlying_price - K1, 0)
        short_call_payoff = max(underlying_price - K2, 0)
        payoff = long_call_payoff - short_call_payoff - debit

        return payoff

    def plot_payoff_diagram(
        self,
        spread: SpreadCandidate,
        current_price: float,
        ticker: str = "",
        expiry: str = "",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot payoff diagram at expiry for a spread."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))
        else:
            fig = ax.get_figure()

        # Generate price range
        padding = spread.spread_width * 1.5
        min_price = spread.long_strike - padding
        max_price = spread.short_strike + padding
        prices = np.linspace(min_price, max_price, 200)

        # Calculate payoffs
        payoffs = []
        for price in prices:
            pnl = self.calculate_spread_pnl(spread, price)
            payoffs.append(pnl * 100)  # Per contract
        payoffs = np.array(payoffs)

        # Plot payoff curve
        ax.plot(prices, payoffs, 'b-', linewidth=2.5, label='P/L at Expiry')

        # Fill profit/loss areas
        ax.fill_between(prices, payoffs, 0, where=(payoffs > 0),
                       color='green', alpha=0.3, label='Profit')
        ax.fill_between(prices, payoffs, 0, where=(payoffs < 0),
                       color='red', alpha=0.3, label='Loss')

        # Zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Key price levels
        ax.axvline(x=spread.long_strike, color='blue', linestyle='--',
                  alpha=0.7, label=f'Long Strike: ${spread.long_strike:.2f}')
        ax.axvline(x=spread.short_strike, color='orange', linestyle='--',
                  alpha=0.7, label=f'Short Strike: ${spread.short_strike:.2f}')
        ax.axvline(x=spread.breakeven, color='purple', linestyle=':',
                  linewidth=2, label=f'Breakeven: ${spread.breakeven:.2f}')
        ax.axvline(x=current_price, color='green', linestyle='-',
                  linewidth=2, alpha=0.8, label=f'Current: ${current_price:.2f}')

        # Max profit/loss lines
        ax.axhline(y=spread.max_profit * 100, color='green', linestyle=':',
                  alpha=0.5)
        ax.axhline(y=-spread.max_loss * 100, color='red', linestyle=':',
                  alpha=0.5)

        # Annotations
        ax.annotate(f'Max Profit: ${spread.max_profit * 100:.2f}',
                   xy=(spread.short_strike + padding * 0.1, spread.max_profit * 100),
                   fontsize=9, color='green')
        ax.annotate(f'Max Loss: -${spread.max_loss * 100:.2f}',
                   xy=(spread.long_strike - padding * 0.5, -spread.max_loss * 100),
                   fontsize=9, color='red')

        # Labels
        title = f"Bull Call Spread Payoff Diagram"
        if ticker:
            title = f"{ticker} {title}"
        if expiry:
            title += f" (Exp: {expiry})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Underlying Price at Expiry ($)', fontsize=12)
        ax.set_ylabel('P/L per Contract ($)', fontsize=12)

        # Stats box
        stats_text = (
            f"Strikes: ${spread.long_strike:.0f} / ${spread.short_strike:.0f}\n"
            f"Net Debit: ${spread.net_debit:.2f}\n"
            f"Max Profit: ${spread.max_profit * 100:.2f}\n"
            f"Max Loss: ${spread.max_loss * 100:.2f}\n"
            f"R:R Ratio: {spread.reward_to_risk:.2f}\n"
            f"Prob Profit: {spread.prob_profit*100:.1f}%" if spread.prob_profit else ""
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        return fig

    def plot_scenario_chart(
        self,
        spread: SpreadCandidate,
        ticker: str = "",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot scenario analysis bar chart."""
        if not spread.scenarios:
            raise ValueError("No scenarios to plot")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        scenarios = spread.scenarios
        names = [s.scenario_name for s in scenarios]
        pnls = [s.pnl_per_contract for s in scenarios]

        # Use lighter colors for better text contrast
        bar_colors = ['#90EE90' if p > 0 else '#FFB6C1' for p in pnls]  # Light green/pink

        bars = ax.bar(names, pnls, color=bar_colors, edgecolor='black', linewidth=1.2)

        # Calculate y-axis range for proper label positioning
        max_pnl = max(pnls) if pnls else 0
        min_pnl = min(pnls) if pnls else 0
        y_range = max_pnl - min_pnl if max_pnl != min_pnl else abs(max_pnl) or 100

        # Add value labels ABOVE bars (outside the bar area for clarity)
        for bar, pnl in zip(bars, pnls):
            height = bar.get_height()
            # Position label outside the bar with adequate spacing
            if height >= 0:
                y_pos = height + y_range * 0.03
                va = 'bottom'
            else:
                y_pos = height - y_range * 0.03
                va = 'top'

            ax.annotate(f'${pnl:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                       ha='center', va=va,
                       fontsize=12, fontweight='bold', color='black')

        # Set x-axis tick labels with scenario names (larger font)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=11, fontweight='medium')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=spread.max_profit * 100, color='green', linestyle='--',
                  linewidth=1.5, alpha=0.7, label=f'Max Profit: ${spread.max_profit * 100:.0f}')
        ax.axhline(y=-spread.max_loss * 100, color='red', linestyle='--',
                  linewidth=1.5, alpha=0.7, label=f'Max Loss: -${spread.max_loss * 100:.0f}')

        title = f"Scenario Analysis: P/L by Underlying Move"
        if ticker:
            title = f"{ticker} {title}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Scenario', fontsize=12)
        ax.set_ylabel('P/L per Contract ($)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

        # Add padding to y-axis to prevent label clipping
        y_min, y_max = ax.get_ylim()
        padding = (y_max - y_min) * 0.15
        ax.set_ylim(y_min - padding * 0.5, y_max + padding)

        return fig

    def plot_compare_spreads(
        self,
        spreads: List[SpreadCandidate],
        current_price: float,
        ticker: str = "",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Compare multiple spreads visually."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 8))
        else:
            fig = ax.get_figure()

        # Price range
        all_strikes = []
        for s in spreads:
            all_strikes.extend([s.long_strike, s.short_strike])
        min_strike = min(all_strikes)
        max_strike = max(all_strikes)
        padding = (max_strike - min_strike) * 0.3
        prices = np.linspace(min_strike - padding, max_strike + padding, 200)

        # Plot each spread
        colors = plt.cm.tab10(np.linspace(0, 1, len(spreads)))
        for i, (spread, color) in enumerate(zip(spreads, colors)):
            payoffs = [self.calculate_spread_pnl(spread, p) * 100 for p in prices]
            label = f"${spread.long_strike:.0f}/${spread.short_strike:.0f} (R:R={spread.reward_to_risk:.2f})"
            ax.plot(prices, payoffs, color=color, linewidth=2, label=label)

        # Zero line and current price
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=current_price, color='green', linestyle='--',
                  linewidth=2, alpha=0.8, label=f'Current: ${current_price:.2f}')

        title = "Comparison of Top Bull Call Spreads"
        if ticker:
            title = f"{ticker} {title}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Underlying Price at Expiry ($)', fontsize=12)
        ax.set_ylabel('P/L per Contract ($)', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        return fig

    def create_full_report(
        self,
        spreads: List[SpreadCandidate],
        current_price: float,
        ticker: str,
        expiry: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Create comprehensive visual report with all charts."""
        fig = plt.figure(figsize=(18, 14))

        # Top spread for detailed analysis
        best_spread = spreads[0] if spreads else None

        # Layout: 2x2 grid
        # Top left: Payoff diagram for best spread
        ax1 = fig.add_subplot(2, 2, 1)
        if best_spread:
            self.plot_payoff_diagram(best_spread, current_price, ticker, expiry, ax=ax1)

        # Top right: Scenario analysis for best spread
        ax2 = fig.add_subplot(2, 2, 2)
        if best_spread and best_spread.scenarios:
            self.plot_scenario_chart(best_spread, ticker, ax=ax2)

        # Bottom left: Compare top spreads
        ax3 = fig.add_subplot(2, 2, 3)
        top_spreads = spreads[:5] if len(spreads) >= 5 else spreads
        if top_spreads:
            self.plot_compare_spreads(top_spreads, current_price, ticker, ax=ax3)

        # Bottom right: Summary table
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        if spreads:
            table_data = []
            headers = ['Strikes', 'Width', 'Debit', 'Max P', 'Max L', 'R:R', 'Prob%', 'Score']
            for s in spreads[:10]:
                table_data.append([
                    f"{s.long_strike:.0f}/{s.short_strike:.0f}",
                    f"${s.spread_width:.0f}",
                    f"${s.net_debit:.2f}",
                    f"${s.max_profit*100:.0f}",
                    f"${s.max_loss*100:.0f}",
                    f"{s.reward_to_risk:.2f}",
                    f"{s.prob_profit*100:.0f}%" if s.prob_profit else "N/A",
                    f"{s.composite_score:.1f}",
                ])

            table = ax4.table(
                cellText=table_data,
                colLabels=headers,
                loc='center',
                cellLoc='center',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)  # Larger font for readability
            table.scale(1.3, 2.0)   # More horizontal and vertical spacing

            # Style header row
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=12)
                table[(0, i)].set_height(0.08)  # Taller header row

            # Style data rows with alternating colors for readability
            for row_idx in range(1, len(table_data) + 1):
                for col_idx in range(len(headers)):
                    cell = table[(row_idx, col_idx)]
                    cell.set_height(0.06)  # Taller data rows
                    # Alternating row colors
                    if row_idx % 2 == 0:
                        cell.set_facecolor('#F0F0F0')
                    else:
                        cell.set_facecolor('#FFFFFF')

            ax4.set_title(f'Top 10 Spreads by Score\n{ticker} @ ${current_price:.2f} | Exp: {expiry}',
                         fontsize=13, fontweight='bold', pad=15)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved report to {save_path}")

        return fig


def print_results_table(spreads: List[SpreadCandidate], current_price: float):
    """Print formatted results table."""
    if not spreads:
        print("No spreads found matching criteria.")
        return

    print("\n" + "=" * 120)
    print(f"{'BULL CALL SPREAD SCAN RESULTS':^120}")
    print(f"{'Current Price: $' + f'{current_price:.2f}':^120}")
    print("=" * 120)
    print(f"{'Rank':<5} {'Strikes':<15} {'Width':>8} {'Debit':>10} {'Max Prof':>10} "
          f"{'Max Loss':>10} {'B/E':>10} {'R:R':>8} {'Prob%':>8} {'EV':>10} {'Score':>8}")
    print("-" * 120)

    for i, s in enumerate(spreads, 1):
        prob_str = f"{s.prob_profit*100:.1f}%" if s.prob_profit else "N/A"
        ev_str = f"${s.expected_value:.2f}" if s.expected_value else "N/A"
        print(f"{i:<5} ${s.long_strike:.0f}/${s.short_strike:.0f}{'':5} "
              f"${s.spread_width:>6.0f} ${s.net_debit:>8.2f} ${s.max_profit*100:>8.0f} "
              f"${s.max_loss*100:>8.0f} ${s.breakeven:>8.2f} {s.reward_to_risk:>7.2f} "
              f"{prob_str:>7} {ev_str:>9} {s.composite_score:>7.1f}")

    print("=" * 120)
    print("\nNote: Max Profit/Loss shown per contract (100 shares)")
    print("R:R = Reward-to-Risk ratio | Prob% = Probability of Profit | EV = Expected Value per share")


def print_scenario_results(spread: SpreadCandidate):
    """Print scenario analysis results."""
    if not spread.scenarios:
        return

    print(f"\n{'SCENARIO ANALYSIS':^95}")
    print(f"Spread: ${spread.long_strike:.0f} / ${spread.short_strike:.0f}")
    print("-" * 95)
    print(f"{'Scenario':<15} {'Underlying':>12} {'Move %':>10} {'P/L/Share':>12} {'P/L/Contract':>15} {'Return %':>12}")
    print("-" * 95)

    for s in spread.scenarios:
        print(f"{s.scenario_name:<15} ${s.underlying_price:>10.2f} {s.underlying_move_pct*100:>9.1f}% "
              f"${s.pnl_per_share:>10.2f} ${s.pnl_per_contract:>13.2f} {s.return_pct:>11.1f}%")
    print("-" * 95)


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bull Call Spread Scanner - Find optimal spread opportunities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan QQQ with default settings (composite scoring, balanced profile)
  python openbb_options.py -t QQQ -e 2025-01-17

  # Custom move scenarios and spread width
  python openbb_options.py -t QQQ -e 2025-01-17 --scenarios=-5,0,5,10 --min-width 5 --max-width 15

  # Set max debit and strike range constraints
  python openbb_options.py -t QQQ -e 2025-01-17 --max-debit 8.0 --strike-range-pct 0.15

  # Use aggressive scoring profile (favors high R:R and upside)
  python openbb_options.py -t QQQ -e 2025-01-17 --profile aggressive

  # Use conservative scoring profile (favors high probability)
  python openbb_options.py -t QQQ -e 2025-01-17 --profile conservative

  # Use simple scoring mode (rank by single metric)
  python openbb_options.py -t QQQ -e 2025-01-17 --scoring-mode simple

  # List available expirations
  python openbb_options.py -t QQQ --list-expirations

  # Save report and export to CSV
  python openbb_options.py -t QQQ -e 2025-01-17 --save report.png --export-csv spreads.csv
        """
    )

    parser.add_argument("-t", "--ticker", required=True, help="Stock ticker (e.g., QQQ)")
    parser.add_argument("-e", "--expiry", help="Expiration date (YYYY-MM-DD)")
    parser.add_argument("--scenarios", default="-3,0,1,2,3,4,5,7,10",
                       help="Comma-separated %% moves (default: -3,0,1,2,3,4,5,7,10)")
    parser.add_argument("--min-width", type=float, default=5.0,
                       help="Minimum spread width in $ (default: 5)")
    parser.add_argument("--max-width", type=float, default=25.0,
                       help="Maximum spread width in $ (default: 20)")
    parser.add_argument("--max-debit", type=float, default=10.0,
                       help="Maximum debit (max loss) per spread in $ (default: 10.0)")
    parser.add_argument("--min-oi", type=int, default=10,
                       help="Minimum open interest per leg (default: 10)")
    parser.add_argument("--min-volume", type=int, default=10,
                       help="Minimum volume per leg (default: 0)")
    parser.add_argument("--strike-range-pct", type=float, default=0.10,
                       help="Filter strikes to ±X%% of spot (default: 0.10 = 10%%)")
    parser.add_argument("--min-rr", type=float, default=1.5,
                       help="Minimum reward-to-risk ratio (default: 0.5)")
    parser.add_argument("--max-be-pct", type=float, default=None,
                       help="Reject if breakeven > spot * (1 + pct) (default: no limit)")
    # Step 5 scoring parameters
    parser.add_argument("--scoring-mode", choices=['simple', 'composite'], default='composite',
                       help="Scoring mode: 'simple' (rank by single metric) or 'composite' (weighted formula)")
    parser.add_argument("--profile", choices=['aggressive', 'conservative', 'balanced'], default='balanced',
                       help="Scoring profile for composite mode (default: balanced)")
    parser.add_argument("--top-n", type=int, default=20,
                       help="Number of top spreads to show (default: 20)")
    # Step 6 selection parameters
    parser.add_argument("--max-risk", type=float, default=None,
                       help="Step 6: Max risk per trade filter (default: no limit)")
    parser.add_argument("--best-n", type=int, default=20,
                       help="Step 6: Number of best spreads to select (default: 10)")
    parser.add_argument("--export-best-csv", help="Export best spreads (Step 6) to CSV")
    # General options
    parser.add_argument("--list-expirations", action="store_true",
                       help="List available expiration dates")
    parser.add_argument("--no-plot", action="store_true",
                       help="Don't display plots")
    parser.add_argument("--save", help="Save report to file")
    parser.add_argument("--export-csv", help="Export all spreads DataFrame to CSV")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        scanner = BullCallSpreadScanner()

        # List expirations
        if args.list_expirations:
            expirations = scanner.get_available_expirations(args.ticker)
            print(f"\nAvailable expirations for {args.ticker.upper()}:")
            for exp in expirations:
                print(f"  {exp}")
            return 0

        if not args.expiry:
            parser.error("--expiry required (use --list-expirations to see available)")

        # Parse scenarios
        scenarios = [float(x.strip()) / 100 for x in args.scenarios.split(',')]

        # Run scan
        spreads, current_price, calls_df, spreads_df, best_spreads_df, best_spreads_summary = scanner.scan(
            ticker=args.ticker,
            expiry=args.expiry,
            scenarios=scenarios,
            # Step 2 parameters
            max_debit_per_spread=args.max_debit,
            min_width=args.min_width,
            max_width=args.max_width,
            min_open_interest=args.min_oi,
            min_volume=args.min_volume,
            strike_range_pct=args.strike_range_pct,
            min_reward_risk=args.min_rr,
            max_breakeven_pct=args.max_be_pct,
            # Step 5 parameters
            scoring_mode=args.scoring_mode,
            scoring_profile=args.profile,
            top_n=args.top_n,
            # Step 6 parameters
            max_risk_per_trade=args.max_risk,
            best_n=args.best_n,
        )

        # Export spreads DataFrame to CSV if requested
        if args.export_csv and not spreads_df.empty:
            spreads_df.to_csv(args.export_csv, index=False)
            logger.info(f"Exported {len(spreads_df)} spreads to {args.export_csv}")

        # Export best spreads (Step 6) to CSV if requested
        if args.export_best_csv and not best_spreads_df.empty:
            best_spreads_df.to_csv(args.export_best_csv, index=False)
            logger.info(f"Exported {len(best_spreads_df)} best spreads to {args.export_best_csv}")

        # Print Step 6 results: Best Spreads Table
        if not best_spreads_df.empty:
            print("\n" + "="*80)
            print("STEP 6: BEST BULL CALL SPREAD SELECTIONS")
            print("="*80)
            scanner.print_best_spreads_table(best_spreads_df, best_spreads_summary)

        # Print detailed results for all scored spreads (verbose)
        if args.verbose:
            print("\n" + "-"*80)
            print("DETAILED SPREAD CANDIDATES (Step 5 Scored)")
            print("-"*80)
            print_results_table(spreads, current_price)

        # Print scenario analysis for top spread
        if spreads:
            print("\n" + "-"*80)
            print("SCENARIO ANALYSIS FOR TOP SPREAD")
            print("-"*80)
            print_scenario_results(spreads[0])

        # Generate visualizations
        if not args.no_plot and spreads:
            fig = scanner.create_full_report(
                spreads,
                current_price,
                args.ticker.upper(),
                args.expiry,
                save_path=args.save,
            )
            plt.show()

        return 0

    except KeyboardInterrupt:
        print("\nCancelled.")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
