"""Tests for data models."""

import pytest
from datetime import date

from broken_wing_condor.models import (
    OptionLeg,
    BrokenWingCondor,
    CondorScore,
    PayoffScenario,
)


class TestOptionLeg:
    """Tests for OptionLeg dataclass."""

    def test_create_call_option(self):
        """Test creating a call option leg."""
        leg = OptionLeg(
            contract_symbol="SPY_600_C",
            strike=600.0,
            option_type="call",
            expiration=date(2024, 3, 15),
            bid=5.00,
            ask=5.20,
            mid=5.10,
        )

        assert leg.strike == 600.0
        assert leg.option_type == "call"
        assert leg.mid == 5.10

    def test_create_put_option(self):
        """Test creating a put option leg."""
        leg = OptionLeg(
            contract_symbol="SPY_590_P",
            strike=590.0,
            option_type="put",
            expiration=date(2024, 3, 15),
            bid=3.00,
            ask=3.20,
            mid=3.10,
        )

        assert leg.strike == 590.0
        assert leg.option_type == "put"
        assert leg.mid == 3.10

    def test_optional_fields(self):
        """Test that optional fields have defaults."""
        leg = OptionLeg(
            contract_symbol="SPY_600_C",
            strike=600.0,
            option_type="call",
            expiration=date(2024, 3, 15),
            bid=5.00,
            ask=5.20,
            mid=5.10,
        )

        assert leg.iv is None
        assert leg.delta is None
        assert leg.volume is None
        assert leg.open_interest is None

    def test_is_liquid_property(self):
        """Test is_liquid property."""
        # Liquid option
        liquid_leg = OptionLeg(
            contract_symbol="SPY_600_C",
            strike=600.0,
            option_type="call",
            expiration=date(2024, 3, 15),
            bid=5.00,
            ask=5.20,
            mid=5.10,
            open_interest=500,
        )
        assert liquid_leg.is_liquid is True

        # Illiquid option
        illiquid_leg = OptionLeg(
            contract_symbol="SPY_600_C",
            strike=600.0,
            option_type="call",
            expiration=date(2024, 3, 15),
            bid=5.00,
            ask=5.20,
            mid=5.10,
            open_interest=50,
        )
        assert illiquid_leg.is_liquid is False

    def test_spread_width_property(self):
        """Test spread_width property."""
        leg = OptionLeg(
            contract_symbol="SPY_600_C",
            strike=600.0,
            option_type="call",
            expiration=date(2024, 3, 15),
            bid=5.00,
            ask=5.20,
            mid=5.10,
        )
        assert leg.spread_width == pytest.approx(0.20, rel=0.01)


class TestBrokenWingCondor:
    """Tests for BrokenWingCondor dataclass."""

    @pytest.fixture
    def sample_condor(self):
        """Create a sample broken-wing condor for testing."""
        long_put = OptionLeg(
            contract_symbol="SPY_580_P",
            strike=580.0,
            option_type="put",
            expiration=date(2024, 3, 15),
            bid=2.00,
            ask=2.20,
            mid=2.10,
        )
        short_put = OptionLeg(
            contract_symbol="SPY_590_P",
            strike=590.0,
            option_type="put",
            expiration=date(2024, 3, 15),
            bid=4.00,
            ask=4.20,
            mid=4.10,
        )
        short_call = OptionLeg(
            contract_symbol="SPY_610_C",
            strike=610.0,
            option_type="call",
            expiration=date(2024, 3, 15),
            bid=2.00,
            ask=2.20,
            mid=2.10,
        )
        long_call = OptionLeg(
            contract_symbol="SPY_620_C",
            strike=620.0,
            option_type="call",
            expiration=date(2024, 3, 15),
            bid=1.00,
            ask=1.20,
            mid=1.10,
        )

        return BrokenWingCondor(
            long_put=long_put,
            short_put=short_put,
            short_call=short_call,
            long_call=long_call,
            put_spread_credit=2.00,  # 4.10 - 2.10
            call_spread_debit=0.00,  # Assuming free call spread
            net_credit=2.00,
            put_spread_width=10.0,  # 590 - 580
            call_spread_width=10.0,  # 620 - 610
            max_loss=800.0,  # (10 - 2) * 100
            max_profit_credit_only=200.0,  # 2.00 * 100
            max_profit_with_calls=1200.0,  # (2 + 10) * 100
        )

    def test_strike_ordering(self, sample_condor):
        """Test that strikes are in correct order."""
        assert sample_condor.long_put.strike < sample_condor.short_put.strike
        assert sample_condor.short_put.strike < sample_condor.short_call.strike
        assert sample_condor.short_call.strike < sample_condor.long_call.strike

    def test_credit_capture_pct(self, sample_condor):
        """Test credit capture percentage calculation."""
        expected_pct = 2.00 / 10.0  # credit / put_width = 20%
        assert sample_condor.credit_capture_pct == pytest.approx(expected_pct, rel=0.01)

    def test_is_free_call_spread(self, sample_condor):
        """Test is_free_call_spread property."""
        assert sample_condor.is_free_call_spread is True

    def test_expiration_property(self, sample_condor):
        """Test expiration property."""
        assert sample_condor.expiration == date(2024, 3, 15)

    def test_get_payoff_table(self, sample_condor):
        """Test payoff table generation."""
        payoffs = sample_condor.get_payoff_table()

        assert len(payoffs) == 5
        # Check scenarios exist
        scenario_names = [p.scenario_name for p in payoffs]
        assert "Below Long Put" in scenario_names
        assert "Above Long Call" in scenario_names

    def test_to_dict(self, sample_condor):
        """Test to_dict serialization."""
        data = sample_condor.to_dict()

        assert "expiration" in data
        assert "legs" in data
        assert "premium" in data
        assert "risk_reward" in data
        assert "payoffs" in data


class TestCondorScore:
    """Tests for CondorScore dataclass."""

    def test_score_creation(self):
        """Test creating a condor score."""
        score = CondorScore(
            risk_score=0.7,
            credit_score=0.6,
            skew_score=0.5,
            call_score=1.0,
            rrr_score=0.4,
            ev_score=0.6,
            pop_score=0.75,
            final_score=0.65,
            max_risk=700.0,
            reward_to_risk=1.5,
            put_credit_pct=0.30,
            call_spread_cost=0.0,
            iv_skew=0.05,
            pop=0.75,
            expected_value=50.0,
        )

        assert score.final_score == 0.65
        assert score.call_score == 1.0  # Free call spread

    def test_score_bounds(self):
        """Test that scores are within valid bounds."""
        score = CondorScore(
            risk_score=0.0,
            credit_score=1.0,
            skew_score=0.5,
            call_score=0.0,
            rrr_score=1.0,
            ev_score=0.5,
            pop_score=0.5,
            final_score=0.5,
            max_risk=500.0,
            reward_to_risk=2.0,
            put_credit_pct=0.50,
            call_spread_cost=0.05,
            iv_skew=0.10,
            pop=0.50,
            expected_value=25.0,
        )

        for attr in ['risk_score', 'credit_score', 'skew_score', 'call_score',
                     'rrr_score', 'ev_score', 'pop_score', 'final_score']:
            value = getattr(score, attr)
            assert 0.0 <= value <= 1.0, f"{attr} should be between 0 and 1"

    def test_to_dict(self):
        """Test to_dict serialization."""
        score = CondorScore(
            risk_score=0.7,
            credit_score=0.6,
            skew_score=0.5,
            call_score=1.0,
            rrr_score=0.4,
            ev_score=0.6,
            pop_score=0.75,
            final_score=0.65,
            max_risk=700.0,
            reward_to_risk=1.5,
            put_credit_pct=0.30,
            call_spread_cost=0.0,
            iv_skew=0.05,
            pop=0.75,
            expected_value=50.0,
        )

        data = score.to_dict()
        assert "final_score" in data
        assert data["final_score"] == pytest.approx(0.65, rel=0.01)


class TestPayoffScenario:
    """Tests for PayoffScenario dataclass."""

    def test_scenario_creation(self):
        """Test creating a payoff scenario."""
        scenario = PayoffScenario(
            scenario_name="Between Put & Call Spreads",
            price_condition="$590 < Price < $610",
            profit_loss=200.0,
            is_max_profit=True,
        )

        assert scenario.scenario_name == "Between Put & Call Spreads"
        assert scenario.profit_loss == 200.0
        assert scenario.is_max_profit is True

    def test_max_loss_scenario(self):
        """Test scenario with max loss."""
        scenario = PayoffScenario(
            scenario_name="Below Long Put",
            price_condition="Price < $580",
            profit_loss=-800.0,
            is_max_loss=True,
        )

        assert scenario.profit_loss < 0
        assert scenario.is_max_loss is True
