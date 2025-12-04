"""
Constants and type aliases for the backtest engine.

This module centralizes magic numbers, strings, and common type definitions.
"""

from typing import Dict, Any, List, TypeVar
import pandas as pd

# Type aliases
YAMLData = Dict[str, Any]
PathMapping = Dict[str, float]

# Signal constants
class Signal:
    """Trading signal constants."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Action:
    """Rule action constants."""
    BUY = "buy"
    SELL = "sell"


class CombineMode:
    """Strategy combine mode constants."""
    ALL = "all"
    ANY = "any"


class SignalMode:
    """Signal generation mode constants."""
    DISCRETE = "discrete"  # Binary rule evaluation (existing behavior)
    SCORE = "score"  # Continuous scoring engine (new)


# Default thresholds (can be overridden)
DEFAULT_BUY_THRESHOLD = 0.3
DEFAULT_SELL_THRESHOLD = -0.3

# Benchmark constants
RISK_FREE_RATE = 0.05  # 5% annual
TRADING_DAYS_PER_MONTH = 20
TRADING_DAYS_PER_YEAR = 252

# Optimizer constants
MAX_SEARCH_SPACE_SIZE = 100000  # Maximum combinations before warning

# Validation constants
REQUIRED_DATA_COLUMNS = ["Date", "Close"]
SUPPORTED_OPERATORS = ["<", "<=", ">", ">=", "==", "!="]
