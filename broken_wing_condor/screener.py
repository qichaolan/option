"""
Main screener module for broken-wing condors.

Provides both a class-based API and a simple function interface.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from broken_wing_condor.config import CondorConfig, ScoringWeights
from broken_wing_condor.data_fetcher import OptionsDataFetcher, get_fetcher
from broken_wing_condor.discovery import discover_condors_for_expiration
from broken_wing_condor.models import BrokenWingCondor, CondorScore
from broken_wing_condor.ranking import (
    RankedCondor,
    filter_by_direction,
    format_csv_output,
    format_ranking_report,
    rank_condors,
)
from broken_wing_condor.scoring import score_condor

logger = logging.getLogger(__name__)


@dataclass
class ScreenerResult:
    """Result from screening operation."""

    symbol: str
    underlying_price: float
    ranked_condors: list[RankedCondor]
    total_candidates: int
    expirations_scanned: list[date]
    config: CondorConfig

    def to_report(self) -> str:
        """Generate human-readable report."""
        return format_ranking_report(
            self.ranked_condors,
            self.symbol,
            self.underlying_price,
        )

    def to_csv(self) -> str:
        """Generate CSV output."""
        return format_csv_output(
            self.ranked_condors,
            self.symbol,
            self.underlying_price,
        )


@dataclass
class CondorScreener:
    """
    Screener for broken-wing condor trades.

    Example usage:
        screener = CondorScreener(symbol="SPY")
        result = screener.screen()
        print(result.to_report())
    """

    symbol: str
    config: CondorConfig = field(default_factory=CondorConfig)
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    fetcher: Optional[OptionsDataFetcher] = None
    direction: str = "neutral"
    annual_volatility: float = 0.20

    def __post_init__(self) -> None:
        """Initialize fetcher if not provided."""
        if self.fetcher is None:
            self.fetcher = get_fetcher(prefer_openbb=True)

    def screen(self, top_n: Optional[int] = None) -> ScreenerResult:
        """
        Run the screening process.

        Args:
            top_n: Number of top results to return (overrides config.top_n)

        Returns:
            ScreenerResult with ranked condors
        """
        if top_n is None:
            top_n = self.config.top_n

        logger.info(f"Starting screen for {self.symbol}")

        # Get underlying price
        underlying_price = self.fetcher.get_underlying_price(self.symbol)
        logger.info(f"{self.symbol} price: ${underlying_price:.2f}")

        # Get available expirations
        expirations = self.fetcher.get_expirations(
            self.symbol,
            self.config.min_dte,
            self.config.max_dte,
        )
        logger.info(f"Found {len(expirations)} expirations in DTE range")

        if not expirations:
            logger.warning("No expirations found in DTE range")
            return ScreenerResult(
                symbol=self.symbol,
                underlying_price=underlying_price,
                ranked_condors=[],
                total_candidates=0,
                expirations_scanned=[],
                config=self.config,
            )

        # Discover and score condors for each expiration
        all_scored: list[tuple[BrokenWingCondor, CondorScore, date, int]] = []
        today = date.today()

        for exp in expirations:
            dte = (exp - today).days
            logger.debug(f"Processing expiration {exp} ({dte} DTE)")

            # Get options chain
            calls, puts = self.fetcher.get_options_chain(self.symbol, exp)

            if not calls or not puts:
                logger.debug(f"No options found for {exp}")
                continue

            # Discover condors
            condors = discover_condors_for_expiration(
                calls, puts, underlying_price, exp, self.config
            )

            # Filter by direction
            if self.direction.lower() != "neutral":
                condors = filter_by_direction(condors, self.direction, underlying_price)

            logger.debug(f"Found {len(condors)} condors for {exp}")

            # Score each condor
            for condor in condors:
                score = score_condor(
                    condor,
                    underlying_price,
                    dte,
                    self.config,
                    self.weights,
                    self.annual_volatility,
                )
                all_scored.append((condor, score, exp, dte))

        logger.info(f"Total scored candidates: {len(all_scored)}")

        # Rank and return top N
        ranked = rank_condors(all_scored, top_n=top_n)

        return ScreenerResult(
            symbol=self.symbol,
            underlying_price=underlying_price,
            ranked_condors=ranked,
            total_candidates=len(all_scored),
            expirations_scanned=expirations,
            config=self.config,
        )


def screen_condors(
    symbol: str,
    min_dte: int = 3,
    max_dte: int = 14,
    max_call_cost: float = 0.05,
    min_put_credit_pct: float = 0.90,
    safety_margin_pct: float = 0.03,
    top_n: int = 20,
    direction: str = "neutral",
    weights: Optional[ScoringWeights] = None,
    prefer_openbb: bool = True,
) -> ScreenerResult:
    """
    Simple function interface to screen for broken-wing condors.

    This is the main entry point for using the screener as a library.

    Args:
        symbol: Underlying symbol (e.g., "SPY")
        min_dte: Minimum days to expiration (default: 3)
        max_dte: Maximum days to expiration (default: 14)
        max_call_cost: Maximum cost for call spread (near-free)
        min_put_credit_pct: Minimum put credit as % of spread width (default: 0.90)
        safety_margin_pct: Long put must be at least this % below spot (default: 0.03)
        top_n: Number of top results to return (default: 20)
        direction: Trade direction ("neutral", "bullish", "bearish")
        weights: Custom scoring weights
        prefer_openbb: Whether to prefer OpenBB over yfinance

    Returns:
        ScreenerResult with ranked condors

    Example:
        from broken_wing_condor import screen_condors

        result = screen_condors("SPY", min_dte=7, max_dte=14, top_n=5)
        print(result.to_report())

        # Or get as CSV
        print(result.to_csv())

        # Access ranked condors directly
        for rc in result.ranked_condors:
            print(f"#{rc.rank}: {rc.final_score:.2f}")
    """
    config = CondorConfig(
        min_dte=min_dte,
        max_dte=max_dte,
        max_call_cost=max_call_cost,
        min_put_credit_pct=min_put_credit_pct,
        safety_margin_pct=safety_margin_pct,
        top_n=top_n,
    )

    if weights is None:
        weights = ScoringWeights()

    fetcher = get_fetcher(prefer_openbb=prefer_openbb)

    screener = CondorScreener(
        symbol=symbol,
        config=config,
        weights=weights,
        fetcher=fetcher,
        direction=direction,
    )

    return screener.screen()
