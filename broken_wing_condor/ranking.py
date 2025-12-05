"""
Ranking system for broken-wing condors.

Sorts and filters scored condors to produce a ranked list of trade candidates.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from broken_wing_condor.config import CondorConfig, ScoringWeights
from broken_wing_condor.models import BrokenWingCondor, CondorScore

logger = logging.getLogger(__name__)


@dataclass
class RankedCondor:
    """A condor with its score and rank."""

    condor: BrokenWingCondor
    score: CondorScore
    rank: int
    expiration: date
    days_to_expiration: int

    @property
    def final_score(self) -> float:
        """Convenience accessor for final score."""
        return self.score.final_score

    def summary(self) -> str:
        """Return a human-readable summary of the trade."""
        return (
            f"Rank #{self.rank} | Score: {self.final_score:.2f}\n"
            f"Expiration: {self.expiration} ({self.days_to_expiration} DTE)\n"
            f"Put Spread: Sell {self.condor.short_put.strike} / Buy {self.condor.long_put.strike} "
            f"(${self.condor.put_spread_credit:.2f} credit)\n"
            f"Call Spread: Buy {self.condor.short_call.strike} / Sell {self.condor.long_call.strike} "
            f"(${self.condor.call_spread_debit:.2f} debit)\n"
            f"Net Credit: ${self.condor.net_credit:.2f} | "
            f"Max Loss: ${self.condor.max_loss:.0f} | "
            f"Max Profit: ${self.condor.max_profit_with_calls:.0f}"
        )

    def payoff_structure(self) -> str:
        """Return payoff structure breakdown by price region."""
        c = self.condor
        net_credit = c.net_credit
        put_long = c.long_put.strike
        put_short = c.short_put.strike
        call_short = c.short_call.strike
        call_long = c.long_call.strike
        put_width = c.put_spread_width
        call_width = c.call_spread_width

        # Calculate P/L for each region
        # Region A: Max Loss (S < PutLong)
        pl_a = (net_credit - put_width) * 100

        # Region C: Credit Plateau (PutShort ≤ S < CallShort)
        pl_c = net_credit * 100

        # Region E: Max Profit (S ≥ CallLong)
        pl_e = (net_credit + call_width) * 100

        lines = [
            "Payoff Structure:",
            f"  Region A (S < {put_long:.0f}):      Max Loss     = ${pl_a:.0f}",
            f"  Region B ({put_long:.0f} ≤ S < {put_short:.0f}): Recovery     = ${pl_a:.0f} → ${pl_c:.0f}",
            f"  Region C ({put_short:.0f} ≤ S < {call_short:.0f}): Credit Zone  = ${pl_c:.0f}",
            f"  Region D ({call_short:.0f} ≤ S < {call_long:.0f}): Convexity    = ${pl_c:.0f} → ${pl_e:.0f}",
            f"  Region E (S ≥ {call_long:.0f}):     Max Profit   = ${pl_e:.0f}",
        ]
        return "\n".join(lines)

    def to_json_dict(self, symbol: str) -> dict:
        """Return JSON-serializable dictionary with payoff regions."""
        c = self.condor
        net_credit = c.net_credit
        put_long = c.long_put.strike
        put_short = c.short_put.strike
        call_short = c.short_call.strike
        call_long = c.long_call.strike
        put_width = c.put_spread_width
        call_width = c.call_spread_width

        # Calculate P/L for each region
        pl_a = (net_credit - put_width) * 100  # Max Loss
        pl_c = net_credit * 100                 # Credit Plateau
        pl_e = (net_credit + call_width) * 100  # Max Profit

        return {
            "rank": self.rank,
            "symbol": symbol,
            "expiration": self.expiration.isoformat(),
            "dte": self.days_to_expiration,
            "put_long": put_long,
            "put_short": put_short,
            "call_short": call_short,
            "call_long": call_long,
            "put_width": put_width,
            "call_width": call_width,
            "put_credit": round(c.put_spread_credit, 2),
            "call_debit": round(c.call_spread_debit, 2),
            "net_credit": round(net_credit, 2),
            "max_loss": round(c.max_loss, 2),
            "max_profit": round(c.max_profit_with_calls, 2),
            "final_score": round(self.final_score, 3),
            "scores": {
                "risk": round(self.score.risk_score, 3),
                "credit": round(self.score.credit_score, 3),
                "skew": round(self.score.skew_score, 3),
                "call": round(self.score.call_score, 3),
                "rrr": round(self.score.rrr_score, 3),
                "ev": round(self.score.ev_score, 3),
                "pop": round(self.score.pop_score, 3),
            },
            "payoff_regions": [
                {
                    "name": "Region A",
                    "type": "negative",
                    "description": "Max loss region",
                    "underlying_range": {"min": None, "max": put_long},
                    "payoff": {"min": round(pl_a, 2), "max": round(pl_a, 2)}
                },
                {
                    "name": "Region B",
                    "type": "sloping_up",
                    "description": "Loss to credit transition",
                    "underlying_range": {"min": put_long, "max": put_short},
                    "payoff": {"min": round(pl_a, 2), "max": round(pl_c, 2)}
                },
                {
                    "name": "Region C",
                    "type": "flat",
                    "description": "Net credit plateau",
                    "underlying_range": {"min": put_short, "max": call_short},
                    "payoff": {"min": round(pl_c, 2), "max": round(pl_c, 2)}
                },
                {
                    "name": "Region D",
                    "type": "sloping_up",
                    "description": "Convexity build region",
                    "underlying_range": {"min": call_short, "max": call_long},
                    "payoff": {"min": round(pl_c, 2), "max": round(pl_e, 2)}
                },
                {
                    "name": "Region E",
                    "type": "capped",
                    "description": "Max profit region",
                    "underlying_range": {"min": call_long, "max": None},
                    "payoff": {"min": round(pl_e, 2), "max": round(pl_e, 2)}
                },
            ],
        }


def rank_condors(
    condors_with_scores: list[tuple[BrokenWingCondor, CondorScore, date, int]],
    top_n: int = 10,
    min_score: Optional[float] = None,
) -> list[RankedCondor]:
    """
    Rank condors by their final score.

    Args:
        condors_with_scores: List of (condor, score, expiration, dte) tuples
        top_n: Number of top condors to return
        min_score: Optional minimum score threshold

    Returns:
        List of RankedCondor objects, sorted by score descending
    """
    # Filter by minimum score if specified
    if min_score is not None:
        condors_with_scores = [
            (c, s, e, d) for c, s, e, d in condors_with_scores
            if s.final_score >= min_score
        ]

    # Sort by final score descending
    sorted_condors = sorted(
        condors_with_scores,
        key=lambda x: x[1].final_score,
        reverse=True,
    )

    # Take top N
    top_condors = sorted_condors[:top_n]

    # Create ranked condors with rank numbers
    ranked = []
    for i, (condor, score, expiration, dte) in enumerate(top_condors, start=1):
        ranked.append(RankedCondor(
            condor=condor,
            score=score,
            rank=i,
            expiration=expiration,
            days_to_expiration=dte,
        ))

    logger.info(f"Ranked {len(ranked)} condors from {len(condors_with_scores)} candidates")

    return ranked


def filter_by_direction(
    condors: list[BrokenWingCondor],
    direction: str,
    underlying_price: float,
) -> list[BrokenWingCondor]:
    """
    Filter condors by directional bias.

    Args:
        condors: List of condors to filter
        direction: 'bullish', 'bearish', or 'neutral'
        underlying_price: Current underlying price

    Returns:
        Filtered list of condors
    """
    if direction.lower() == "neutral":
        # For neutral, prefer condors where short strikes are equidistant from current price
        return condors

    elif direction.lower() == "bullish":
        # For bullish, prefer condors with:
        # - Short put strike closer to ATM (more credit)
        # - Wider call spread (more upside)
        return [
            c for c in condors
            if c.short_put.strike <= underlying_price * 1.02  # Short put at or below price
            and c.call_spread_width >= c.put_spread_width  # Call spread >= put spread
        ]

    elif direction.lower() == "bearish":
        # For bearish, prefer condors with:
        # - Short put strike further OTM (more protection)
        # - Narrower position overall
        return [
            c for c in condors
            if c.short_put.strike <= underlying_price * 0.98  # Short put OTM
        ]

    else:
        logger.warning(f"Unknown direction '{direction}', returning all condors")
        return condors


def group_by_expiration(
    ranked_condors: list[RankedCondor],
) -> dict[date, list[RankedCondor]]:
    """
    Group ranked condors by expiration date.

    Args:
        ranked_condors: List of ranked condors

    Returns:
        Dictionary mapping expiration dates to lists of condors
    """
    groups: dict[date, list[RankedCondor]] = {}

    for condor in ranked_condors:
        if condor.expiration not in groups:
            groups[condor.expiration] = []
        groups[condor.expiration].append(condor)

    return groups


def get_best_per_expiration(
    ranked_condors: list[RankedCondor],
    top_per_exp: int = 3,
) -> list[RankedCondor]:
    """
    Get the top N condors for each expiration date.

    Args:
        ranked_condors: List of ranked condors
        top_per_exp: Number of condors to return per expiration

    Returns:
        List of top condors for each expiration
    """
    groups = group_by_expiration(ranked_condors)

    result = []
    for exp_date in sorted(groups.keys()):
        exp_condors = groups[exp_date]
        # Already sorted by score, take top N
        result.extend(exp_condors[:top_per_exp])

    return result


def format_ranking_report(
    ranked_condors: list[RankedCondor],
    symbol: str,
    underlying_price: float,
) -> str:
    """
    Format a human-readable ranking report.

    Args:
        ranked_condors: List of ranked condors
        symbol: Underlying symbol
        underlying_price: Current price

    Returns:
        Formatted report string
    """
    if not ranked_condors:
        return f"No broken-wing condor candidates found for {symbol}"

    lines = [
        "=" * 70,
        f"BROKEN-WING CONDOR SCREENER RESULTS",
        f"Symbol: {symbol} | Price: ${underlying_price:.2f}",
        "=" * 70,
        "",
    ]

    for rc in ranked_condors:
        lines.append("-" * 70)
        lines.append(rc.summary())
        lines.append("")

        # Add payoff structure
        lines.append(rc.payoff_structure())
        lines.append("")

        # Add score breakdown
        lines.append("Score Breakdown:")
        lines.append(
            f"  Risk: {rc.score.risk_score:.2f} | "
            f"Credit: {rc.score.credit_score:.2f} | "
            f"Skew: {rc.score.skew_score:.2f} | "
            f"Call: {rc.score.call_score:.2f}"
        )
        lines.append(
            f"  RRR: {rc.score.rrr_score:.2f} | "
            f"EV: {rc.score.ev_score:.2f} | "
            f"PoP: {rc.score.pop_score:.2f}"
        )
        lines.append("")

    lines.append("=" * 70)
    lines.append(f"Total candidates found: {len(ranked_condors)}")
    lines.append("=" * 70)

    return "\n".join(lines)


def format_csv_output(
    ranked_condors: list[RankedCondor],
    symbol: str,
    underlying_price: float,
) -> str:
    """
    Format ranking results as CSV.

    Args:
        ranked_condors: List of ranked condors
        symbol: Underlying symbol
        underlying_price: Current price

    Returns:
        CSV formatted string
    """
    headers = [
        "rank", "symbol", "underlying_price", "expiration", "dte",
        "short_put", "long_put", "short_call", "long_call",
        "put_spread_width", "call_spread_width",
        "put_credit", "call_debit", "net_credit",
        "max_loss", "max_profit",
        "risk_score", "credit_score", "skew_score", "call_score",
        "rrr_score", "ev_score", "pop_score", "final_score",
    ]

    lines = [",".join(headers)]

    for rc in ranked_condors:
        row = [
            str(rc.rank),
            symbol,
            f"{underlying_price:.2f}",
            rc.expiration.isoformat(),
            str(rc.days_to_expiration),
            f"{rc.condor.short_put.strike:.2f}",
            f"{rc.condor.long_put.strike:.2f}",
            f"{rc.condor.short_call.strike:.2f}",
            f"{rc.condor.long_call.strike:.2f}",
            f"{rc.condor.put_spread_width:.0f}",
            f"{rc.condor.call_spread_width:.0f}",
            f"{rc.condor.put_spread_credit:.2f}",
            f"{rc.condor.call_spread_debit:.2f}",
            f"{rc.condor.net_credit:.2f}",
            f"{rc.condor.max_loss:.0f}",
            f"{rc.condor.max_profit_with_calls:.0f}",
            f"{rc.score.risk_score:.3f}",
            f"{rc.score.credit_score:.3f}",
            f"{rc.score.skew_score:.3f}",
            f"{rc.score.call_score:.3f}",
            f"{rc.score.rrr_score:.3f}",
            f"{rc.score.ev_score:.3f}",
            f"{rc.score.pop_score:.3f}",
            f"{rc.score.final_score:.3f}",
        ]
        lines.append(",".join(row))

    return "\n".join(lines)
