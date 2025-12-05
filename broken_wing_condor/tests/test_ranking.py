"""Tests for ranking system."""

import pytest
from datetime import date

from broken_wing_condor.models import OptionLeg, BrokenWingCondor, CondorScore
from broken_wing_condor.ranking import (
    RankedCondor,
    rank_condors,
    filter_by_direction,
    group_by_expiration,
    get_best_per_expiration,
    format_ranking_report,
    format_csv_output,
)


def create_condor(put_strike, call_strike, expiration):
    """Helper to create a condor for testing."""
    return BrokenWingCondor(
        long_put=OptionLeg("", put_strike - 10, "put", expiration, 1, 1, 1),
        short_put=OptionLeg("", put_strike, "put", expiration, 1, 1, 1),
        short_call=OptionLeg("", call_strike, "call", expiration, 1, 1, 1),
        long_call=OptionLeg("", call_strike + 10, "call", expiration, 1, 1, 1),
        put_spread_credit=3.00,
        call_spread_debit=0.00,
        net_credit=3.00,
        put_spread_width=10.0,
        call_spread_width=10.0,
        max_loss=700.0,
        max_profit_credit_only=300.0,
        max_profit_with_calls=1300.0,
    )


def create_score(final_score):
    """Helper to create a score for testing."""
    return CondorScore(
        risk_score=0.5,
        credit_score=0.5,
        skew_score=0.5,
        call_score=1.0,
        rrr_score=0.5,
        ev_score=0.5,
        pop_score=0.5,
        final_score=final_score,
        max_risk=700.0,
        reward_to_risk=1.5,
        put_credit_pct=0.30,
        call_spread_cost=0.0,
        iv_skew=0.05,
        pop=0.50,
        expected_value=50.0,
    )


@pytest.fixture
def sample_condors_with_scores():
    """Create sample condors with scores for testing."""
    expiration1 = date(2024, 3, 15)
    expiration2 = date(2024, 3, 22)

    return [
        (create_condor(590, 610, expiration1), create_score(0.85), expiration1, 30),
        (create_condor(585, 615, expiration1), create_score(0.75), expiration1, 30),
        (create_condor(595, 605, expiration2), create_score(0.90), expiration2, 37),
        (create_condor(580, 620, expiration2), create_score(0.65), expiration2, 37),
        (create_condor(588, 612, expiration1), create_score(0.80), expiration1, 30),
    ]


class TestRankedCondor:
    """Tests for RankedCondor dataclass."""

    def test_final_score_property(self):
        """Test final_score property accessor."""
        expiration = date(2024, 3, 15)

        condor = create_condor(590, 610, expiration)
        score = create_score(0.75)

        ranked = RankedCondor(
            condor=condor,
            score=score,
            rank=1,
            expiration=expiration,
            days_to_expiration=30,
        )

        assert ranked.final_score == 0.75

    def test_summary_method(self):
        """Test summary method generates readable output."""
        expiration = date(2024, 3, 15)

        condor = create_condor(590, 610, expiration)
        score = create_score(0.75)

        ranked = RankedCondor(
            condor=condor,
            score=score,
            rank=1,
            expiration=expiration,
            days_to_expiration=30,
        )

        summary = ranked.summary()

        assert "Rank #1" in summary
        assert "Score: 0.75" in summary
        assert "590" in summary  # Short put strike
        assert "610" in summary  # Short call strike


class TestRankCondors:
    """Tests for rank_condors function."""

    def test_ranks_by_score_descending(self, sample_condors_with_scores):
        """Test that condors are ranked by score descending."""
        ranked = rank_condors(sample_condors_with_scores, top_n=10)

        # Check descending order
        for i in range(len(ranked) - 1):
            assert ranked[i].final_score >= ranked[i + 1].final_score

    def test_top_n_limit(self, sample_condors_with_scores):
        """Test that only top N are returned."""
        ranked = rank_condors(sample_condors_with_scores, top_n=3)

        assert len(ranked) == 3

    def test_rank_numbers_assigned(self, sample_condors_with_scores):
        """Test that rank numbers are correctly assigned."""
        ranked = rank_condors(sample_condors_with_scores, top_n=5)

        for i, rc in enumerate(ranked, start=1):
            assert rc.rank == i

    def test_min_score_filter(self, sample_condors_with_scores):
        """Test filtering by minimum score."""
        ranked = rank_condors(
            sample_condors_with_scores,
            top_n=10,
            min_score=0.80,
        )

        # Only scores >= 0.80 should be included
        for rc in ranked:
            assert rc.final_score >= 0.80


class TestFilterByDirection:
    """Tests for filter_by_direction function."""

    @pytest.fixture
    def condors_for_direction(self):
        """Create condors with various configurations."""
        expiration = date(2024, 3, 15)

        def create_direction_condor(short_put, short_call, put_width, call_width):
            return BrokenWingCondor(
                long_put=OptionLeg("", short_put - put_width, "put", expiration, 1, 1, 1),
                short_put=OptionLeg("", short_put, "put", expiration, 1, 1, 1),
                short_call=OptionLeg("", short_call, "call", expiration, 1, 1, 1),
                long_call=OptionLeg("", short_call + call_width, "call", expiration, 1, 1, 1),
                put_spread_credit=3.00,
                call_spread_debit=0.00,
                net_credit=3.00,
                put_spread_width=float(put_width),
                call_spread_width=float(call_width),
                max_loss=700.0,
                max_profit_credit_only=300.0,
                max_profit_with_calls=1300.0,
            )

        return [
            create_direction_condor(600, 620, 10, 20),  # ATM put, wide call
            create_direction_condor(590, 620, 10, 10),  # OTM put
            create_direction_condor(605, 615, 5, 10),   # ITM put
        ]

    def test_neutral_returns_all(self, condors_for_direction):
        """Test neutral direction returns all condors."""
        filtered = filter_by_direction(
            condors_for_direction,
            "neutral",
            underlying_price=600.0,
        )

        assert len(filtered) == len(condors_for_direction)

    def test_bullish_filter(self, condors_for_direction):
        """Test bullish filter criteria."""
        filtered = filter_by_direction(
            condors_for_direction,
            "bullish",
            underlying_price=600.0,
        )

        # Should include condors with short put at/below price and wider call spread
        assert len(filtered) >= 1

    def test_bearish_filter(self, condors_for_direction):
        """Test bearish filter criteria."""
        filtered = filter_by_direction(
            condors_for_direction,
            "bearish",
            underlying_price=600.0,
        )

        # Should include condors with OTM short put
        for condor in filtered:
            assert condor.short_put.strike <= 600.0 * 0.98


class TestGroupByExpiration:
    """Tests for group_by_expiration function."""

    def test_groups_correctly(self, sample_condors_with_scores):
        """Test that condors are grouped by expiration."""
        # First rank them
        ranked = rank_condors(sample_condors_with_scores, top_n=10)

        # Then group
        groups = group_by_expiration(ranked)

        # Should have 2 groups (2 different expirations)
        assert len(groups) == 2

        # Check each group has correct expiration
        for exp, condors in groups.items():
            for condor in condors:
                assert condor.expiration == exp


class TestGetBestPerExpiration:
    """Tests for get_best_per_expiration function."""

    def test_returns_top_per_exp(self, sample_condors_with_scores):
        """Test returning top N per expiration."""
        ranked = rank_condors(sample_condors_with_scores, top_n=10)
        best = get_best_per_expiration(ranked, top_per_exp=2)

        # Group results by expiration
        groups = group_by_expiration(best)

        # Each group should have at most 2
        for condors in groups.values():
            assert len(condors) <= 2


class TestFormatRankingReport:
    """Tests for format_ranking_report function."""

    def test_report_contains_key_info(self, sample_condors_with_scores):
        """Test that report contains key information."""
        ranked = rank_condors(sample_condors_with_scores, top_n=3)
        report = format_ranking_report(ranked, "SPY", 600.0)

        assert "SPY" in report
        assert "600.00" in report
        assert "Rank #1" in report

    def test_empty_results_message(self):
        """Test message for empty results."""
        report = format_ranking_report([], "SPY", 600.0)

        assert "No broken-wing condor candidates found" in report


class TestFormatCSVOutput:
    """Tests for format_csv_output function."""

    def test_csv_has_headers(self, sample_condors_with_scores):
        """Test that CSV output has headers."""
        ranked = rank_condors(sample_condors_with_scores, top_n=3)
        csv = format_csv_output(ranked, "SPY", 600.0)

        lines = csv.split("\n")
        headers = lines[0]

        assert "rank" in headers
        assert "symbol" in headers
        assert "final_score" in headers

    def test_csv_has_data_rows(self, sample_condors_with_scores):
        """Test that CSV output has data rows."""
        ranked = rank_condors(sample_condors_with_scores, top_n=3)
        csv = format_csv_output(ranked, "SPY", 600.0)

        lines = csv.split("\n")

        # Header + 3 data rows
        assert len(lines) == 4
