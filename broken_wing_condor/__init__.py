"""
Broken-Wing Condor with Free Call Spread Screener

A quantitative options screening tool that discovers and ranks broken-wing condor
strategies where the call spread is (nearly) free, funded by deep put spread credit.

Usage as library:
    from broken_wing_condor import screen_condors

    result = screen_condors("SPY", min_dte=7, max_dte=14, top_n=5)
    print(result.to_report())

    # Or get ranked condors directly
    for rc in result.ranked_condors:
        print(f"#{rc.rank}: Score {rc.final_score:.2f}")

Usage as CLI:
    python -m broken_wing_condor SPY
    python -m broken_wing_condor SPY --min-dte 7 --max-dte 14 --top 5
    python -m broken_wing_condor SPY --direction bullish --csv results.csv
"""

from broken_wing_condor.config import CondorConfig, ScoringWeights
from broken_wing_condor.models import (
    OptionLeg,
    BrokenWingCondor,
    CondorScore,
    PayoffScenario,
)
from broken_wing_condor.screener import screen_condors, CondorScreener, ScreenerResult
from broken_wing_condor.ranking import RankedCondor

__version__ = "1.0.0"

__all__ = [
    "CondorConfig",
    "ScoringWeights",
    "OptionLeg",
    "BrokenWingCondor",
    "CondorScore",
    "PayoffScenario",
    "screen_condors",
    "CondorScreener",
    "ScreenerResult",
    "RankedCondor",
]
