"""Tests for scoring framework."""

import pytest
from datetime import date

from broken_wing_condor.config import CondorConfig, ScoringWeights
from broken_wing_condor.models import OptionLeg, BrokenWingCondor, CondorScore
from broken_wing_condor.scoring import (
    calculate_risk_score,
    calculate_credit_score,
    calculate_skew_score,
    calculate_call_score,
    calculate_rrr_score,
    calculate_ev_score,
    calculate_pop_score,
    estimate_probability_of_profit,
    calculate_pl_at_expiration,
    calculate_put_spread_pl,
    calculate_call_spread_pl,
    score_condor,
)


@pytest.fixture
def sample_condor():
    """Create a sample condor for testing."""
    expiration = date(2024, 3, 15)

    long_put = OptionLeg(
        contract_symbol="SPY_580_P",
        strike=580.0,
        option_type="put",
        expiration=expiration,
        bid=1.90, ask=2.10, mid=2.00,
    )
    short_put = OptionLeg(
        contract_symbol="SPY_590_P",
        strike=590.0,
        option_type="put",
        expiration=expiration,
        bid=4.90, ask=5.10, mid=5.00,
    )
    short_call = OptionLeg(
        contract_symbol="SPY_610_C",
        strike=610.0,
        option_type="call",
        expiration=expiration,
        bid=2.40, ask=2.60, mid=2.50,
    )
    long_call = OptionLeg(
        contract_symbol="SPY_620_C",
        strike=620.0,
        option_type="call",
        expiration=expiration,
        bid=1.40, ask=1.60, mid=1.50,
    )

    return BrokenWingCondor(
        long_put=long_put,
        short_put=short_put,
        short_call=short_call,
        long_call=long_call,
        put_spread_credit=3.00,
        call_spread_debit=0.00,  # Free call spread
        net_credit=3.00,
        put_spread_width=10.0,
        call_spread_width=10.0,
        max_loss=700.0,
        max_profit_credit_only=300.0,
        max_profit_with_calls=1300.0,
    )


@pytest.fixture
def config():
    """Create sample configuration."""
    return CondorConfig()


class TestCalculateRiskScore:
    """Tests for calculate_risk_score function."""

    def test_low_risk_high_score(self, config):
        """Test that low risk results in high score."""
        expiration = date(2024, 3, 15)

        # Create condor with low max_loss relative to spread width
        condor = BrokenWingCondor(
            long_put=OptionLeg("", 580.0, "put", expiration, 1, 1, 1),
            short_put=OptionLeg("", 590.0, "put", expiration, 1, 1, 1),
            short_call=OptionLeg("", 610.0, "call", expiration, 1, 1, 1),
            long_call=OptionLeg("", 620.0, "call", expiration, 1, 1, 1),
            put_spread_credit=8.00,  # High credit
            call_spread_debit=0.00,
            net_credit=8.00,
            put_spread_width=10.0,
            call_spread_width=10.0,
            max_loss=200.0,  # Only 20% of spread at risk
            max_profit_credit_only=800.0,
            max_profit_with_calls=1800.0,
        )

        score = calculate_risk_score(condor, config)
        assert score >= 0.7  # Low risk = high score

    def test_score_bounds(self, sample_condor, config):
        """Test score is within 0-1 bounds."""
        score = calculate_risk_score(sample_condor, config)
        assert 0.0 <= score <= 1.0


class TestCalculateCreditScore:
    """Tests for calculate_credit_score function."""

    def test_high_credit_high_score(self, config):
        """Test that high credit capture results in high score."""
        expiration = date(2024, 3, 15)

        condor = BrokenWingCondor(
            long_put=OptionLeg("", 580.0, "put", expiration, 1, 1, 1),
            short_put=OptionLeg("", 590.0, "put", expiration, 1, 1, 1),
            short_call=OptionLeg("", 610.0, "call", expiration, 1, 1, 1),
            long_call=OptionLeg("", 620.0, "call", expiration, 1, 1, 1),
            put_spread_credit=7.00,  # 70% credit capture
            call_spread_debit=0.00,
            net_credit=7.00,
            put_spread_width=10.0,
            call_spread_width=10.0,
            max_loss=300.0,
            max_profit_credit_only=700.0,
            max_profit_with_calls=1700.0,
        )

        score = calculate_credit_score(condor, config)
        assert score >= 0.6  # High credit = high score

    def test_score_bounds(self, sample_condor, config):
        """Test score is within 0-1 bounds."""
        score = calculate_credit_score(sample_condor, config)
        assert 0.0 <= score <= 1.0


class TestCalculateSkewScore:
    """Tests for calculate_skew_score function."""

    def test_wide_call_spread_high_score(self):
        """Test that wider call spread results in high score."""
        expiration = date(2024, 3, 15)

        condor = BrokenWingCondor(
            long_put=OptionLeg("", 585.0, "put", expiration, 1, 1, 1),
            short_put=OptionLeg("", 590.0, "put", expiration, 1, 1, 1),
            short_call=OptionLeg("", 610.0, "call", expiration, 1, 1, 1),
            long_call=OptionLeg("", 630.0, "call", expiration, 1, 1, 1),  # 20pt wide
            put_spread_credit=3.00,
            call_spread_debit=0.00,
            net_credit=3.00,
            put_spread_width=5.0,  # 5pt put spread
            call_spread_width=20.0,  # 20pt call spread (4:1 ratio)
            max_loss=200.0,
            max_profit_credit_only=300.0,
            max_profit_with_calls=2300.0,
        )

        score = calculate_skew_score(condor)
        assert score >= 0.8  # Wide call spread = high skew score

    def test_symmetric_spread_mid_score(self):
        """Test that symmetric spread gets mid-range score."""
        expiration = date(2024, 3, 15)

        condor = BrokenWingCondor(
            long_put=OptionLeg("", 580.0, "put", expiration, 1, 1, 1),
            short_put=OptionLeg("", 590.0, "put", expiration, 1, 1, 1),
            short_call=OptionLeg("", 610.0, "call", expiration, 1, 1, 1),
            long_call=OptionLeg("", 620.0, "call", expiration, 1, 1, 1),
            put_spread_credit=3.00,
            call_spread_debit=0.00,
            net_credit=3.00,
            put_spread_width=10.0,
            call_spread_width=10.0,  # 1:1 ratio
            max_loss=700.0,
            max_profit_credit_only=300.0,
            max_profit_with_calls=1300.0,
        )

        score = calculate_skew_score(condor)
        # Ratio of 1.0 should give roughly 0.33 score based on formula
        assert 0.2 <= score <= 0.5


class TestCalculateCallScore:
    """Tests for calculate_call_score function."""

    def test_free_call_spread_perfect_score(self, config):
        """Test that free call spread gets score of 1.0."""
        expiration = date(2024, 3, 15)

        condor = BrokenWingCondor(
            long_put=OptionLeg("", 580.0, "put", expiration, 1, 1, 1),
            short_put=OptionLeg("", 590.0, "put", expiration, 1, 1, 1),
            short_call=OptionLeg("", 610.0, "call", expiration, 1, 1, 1),
            long_call=OptionLeg("", 620.0, "call", expiration, 1, 1, 1),
            put_spread_credit=3.00,
            call_spread_debit=0.00,  # Free!
            net_credit=3.00,
            put_spread_width=10.0,
            call_spread_width=10.0,
            max_loss=700.0,
            max_profit_credit_only=300.0,
            max_profit_with_calls=1300.0,
        )

        score = calculate_call_score(condor, config)
        assert score == 1.0

    def test_credit_call_spread_perfect_score(self, config):
        """Test that call spread for credit also gets 1.0."""
        expiration = date(2024, 3, 15)

        condor = BrokenWingCondor(
            long_put=OptionLeg("", 580.0, "put", expiration, 1, 1, 1),
            short_put=OptionLeg("", 590.0, "put", expiration, 1, 1, 1),
            short_call=OptionLeg("", 610.0, "call", expiration, 1, 1, 1),
            long_call=OptionLeg("", 620.0, "call", expiration, 1, 1, 1),
            put_spread_credit=3.00,
            call_spread_debit=-0.50,  # Credit!
            net_credit=3.50,
            put_spread_width=10.0,
            call_spread_width=10.0,
            max_loss=650.0,
            max_profit_credit_only=350.0,
            max_profit_with_calls=1350.0,
        )

        score = calculate_call_score(condor, config)
        assert score == 1.0


class TestCalculateRRRScore:
    """Tests for calculate_rrr_score function."""

    def test_high_rrr_high_score(self):
        """Test that high risk/reward ratio results in high score."""
        expiration = date(2024, 3, 15)

        condor = BrokenWingCondor(
            long_put=OptionLeg("", 580.0, "put", expiration, 1, 1, 1),
            short_put=OptionLeg("", 590.0, "put", expiration, 1, 1, 1),
            short_call=OptionLeg("", 610.0, "call", expiration, 1, 1, 1),
            long_call=OptionLeg("", 620.0, "call", expiration, 1, 1, 1),
            put_spread_credit=5.00,
            call_spread_debit=0.00,
            net_credit=5.00,
            put_spread_width=10.0,
            call_spread_width=10.0,
            max_loss=500.0,
            max_profit_credit_only=500.0,
            max_profit_with_calls=1500.0,  # 3:1 RRR
        )

        score = calculate_rrr_score(condor)
        assert score >= 0.7

    def test_score_bounds(self, sample_condor):
        """Test score is within 0-1 bounds."""
        score = calculate_rrr_score(sample_condor)
        assert 0.0 <= score <= 1.0


class TestEstimateProbabilityOfProfit:
    """Tests for estimate_probability_of_profit function."""

    def test_atm_short_put_roughly_50_percent(self, sample_condor):
        """Test that ATM short put has roughly 50% PoP."""
        # Short put at 590 with underlying at 590
        pop = estimate_probability_of_profit(
            sample_condor,
            underlying_price=590.0,
            days_to_expiration=30,
            annual_volatility=0.20,
        )

        # Should be close to 50% for ATM
        assert 0.4 <= pop <= 0.6

    def test_deep_otm_short_put_high_pop(self, sample_condor):
        """Test that deep OTM short put has high PoP."""
        # Short put at 590 with underlying at 650 (way above)
        pop = estimate_probability_of_profit(
            sample_condor,
            underlying_price=650.0,
            days_to_expiration=30,
            annual_volatility=0.20,
        )

        assert pop >= 0.8


class TestPLCalculations:
    """Tests for P/L calculation functions."""

    def test_put_spread_pl_above_short_strike(self):
        """Test put spread P/L when price is above short strike."""
        pl = calculate_put_spread_pl(
            long_strike=580.0,
            short_strike=590.0,
            price=600.0,  # Above both strikes
        )

        # Both puts expire worthless, P/L = 0 (excluding premium)
        assert pl == 0.0

    def test_put_spread_pl_below_long_strike(self):
        """Test put spread P/L when price is below long strike."""
        pl = calculate_put_spread_pl(
            long_strike=580.0,
            short_strike=590.0,
            price=570.0,  # Below both strikes
        )

        # Max loss = -10 (width of spread)
        assert pl == -10.0

    def test_put_spread_pl_between_strikes(self):
        """Test put spread P/L when price is between strikes."""
        pl = calculate_put_spread_pl(
            long_strike=580.0,
            short_strike=590.0,
            price=585.0,  # Between strikes
        )

        # Short put ITM by 5, long put OTM
        # P/L = 0 - 5 = -5
        assert pl == -5.0

    def test_call_spread_pl_below_short_strike(self):
        """Test call spread P/L when price is below short strike."""
        pl = calculate_call_spread_pl(
            short_strike=610.0,
            long_strike=620.0,
            price=600.0,
        )

        # Both calls expire worthless
        assert pl == 0.0

    def test_call_spread_pl_above_long_strike(self):
        """Test call spread P/L when price is above long strike."""
        pl = calculate_call_spread_pl(
            short_strike=610.0,
            long_strike=620.0,
            price=640.0,
        )

        # For a bull call spread (upside participation in BWC):
        # - We're LONG the 610 call: value = 640 - 610 = 30 (we gain this)
        # - We're SHORT the 620 call: liability = 640 - 620 = 20 (we owe this)
        # Net = 30 - 20 = +10 (max profit = spread width)
        assert pl == 10.0


class TestCalculatePLAtExpiration:
    """Tests for calculate_pl_at_expiration function."""

    def test_pl_in_profit_zone(self, sample_condor):
        """Test P/L when price is in profit zone."""
        # Price between short strikes
        pl = calculate_pl_at_expiration(sample_condor, 600.0)

        # Should keep full credit
        assert pl == pytest.approx(300.0, rel=0.1)  # net_credit * 100

    def test_pl_at_max_loss(self, sample_condor):
        """Test P/L at max loss scenario."""
        # Price way below long put
        pl = calculate_pl_at_expiration(sample_condor, 560.0)

        # Should be max loss
        assert pl < 0


class TestScoreCondor:
    """Tests for score_condor function."""

    def test_returns_valid_score(self, sample_condor, config):
        """Test that score_condor returns valid CondorScore."""
        score = score_condor(
            sample_condor,
            underlying_price=600.0,
            days_to_expiration=30,
            config=config,
        )

        assert isinstance(score, CondorScore)
        assert 0.0 <= score.final_score <= 1.0

    def test_all_components_populated(self, sample_condor, config):
        """Test that all score components are populated."""
        score = score_condor(
            sample_condor,
            underlying_price=600.0,
            days_to_expiration=30,
            config=config,
        )

        assert score.risk_score is not None
        assert score.credit_score is not None
        assert score.skew_score is not None
        assert score.call_score is not None
        assert score.rrr_score is not None
        assert score.ev_score is not None
        assert score.pop_score is not None
        assert score.max_risk is not None
        assert score.reward_to_risk is not None

    def test_custom_weights(self, sample_condor, config):
        """Test scoring with custom weights."""
        weights = ScoringWeights(
            risk_weight=0.50,  # Heavy risk weighting
            credit_weight=0.10,
            skew_weight=0.10,
            call_weight=0.10,
            rrr_weight=0.10,
            ev_weight=0.05,
            pop_weight=0.05,
        )

        score = score_condor(
            sample_condor,
            underlying_price=600.0,
            days_to_expiration=30,
            config=config,
            weights=weights,
        )

        assert 0.0 <= score.final_score <= 1.0
