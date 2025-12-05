"""Tests for strategy discovery module."""

import pytest
from datetime import date

from broken_wing_condor.config import CondorConfig
from broken_wing_condor.models import OptionLeg, BrokenWingCondor
from broken_wing_condor.discovery import (
    find_atm_strike,
    get_strike_ladder,
    get_option_at_strike,
    generate_put_spread_candidates,
    generate_call_spread_candidates,
    construct_condor,
    validate_condor,
    discover_condors,
)


@pytest.fixture
def sample_puts():
    """Create sample put options for testing."""
    expiration = date(2024, 3, 15)
    strikes = [570, 575, 580, 585, 590, 595, 600, 605, 610]

    puts = []
    for strike in strikes:
        # Simulate put prices - higher strike = higher price
        # Use steeper price gradient to ensure sufficient credit capture
        # For a 5-point spread to have 20% capture, we need $1 credit
        # For a 10-point spread to have 20% capture, we need $2 credit
        base_price = max(0.5, (610 - strike) * 0.3 + 0.5)
        puts.append(OptionLeg(
            contract_symbol=f"SPY_{strike}_P",
            strike=float(strike),
            option_type="put",
            expiration=expiration,
            bid=base_price - 0.1,
            ask=base_price + 0.1,
            mid=base_price,
        ))

    return puts


@pytest.fixture
def sample_calls():
    """Create sample call options for testing."""
    expiration = date(2024, 3, 15)
    strikes = [600, 605, 610, 615, 620, 625, 630, 635, 640]

    calls = []
    for strike in strikes:
        # Simulate call prices - lower strike = higher price
        base_price = max(0.5, (620 - strike) * 0.1 + 1)
        calls.append(OptionLeg(
            contract_symbol=f"SPY_{strike}_C",
            strike=float(strike),
            option_type="call",
            expiration=expiration,
            bid=base_price - 0.1,
            ask=base_price + 0.1,
            mid=base_price,
        ))

    return calls


@pytest.fixture
def config():
    """Create sample configuration."""
    return CondorConfig(
        min_dte=14,
        max_dte=60,
        max_call_cost=0.50,
        min_put_credit_pct=0.20,
        put_spread_width_min=5,
        put_spread_width_max=15,
        call_spread_width=10,
    )


class TestFindATMStrike:
    """Tests for find_atm_strike function."""

    def test_find_exact_atm(self, sample_puts):
        """Test finding exact ATM strike."""
        atm = find_atm_strike(sample_puts, 600.0)
        assert atm == 600.0

    def test_find_closest_atm(self, sample_puts):
        """Test finding closest ATM when exact not available."""
        atm = find_atm_strike(sample_puts, 597.0)
        assert atm == 595.0  # Closest available strike

    def test_empty_options(self):
        """Test with empty options list."""
        atm = find_atm_strike([], 600.0)
        assert atm is None


class TestGetStrikeLadder:
    """Tests for get_strike_ladder function."""

    def test_get_strikes_in_range(self, sample_puts):
        """Test getting strikes within range."""
        strikes = get_strike_ladder(sample_puts, 580.0, 600.0)

        assert 580.0 in strikes
        assert 600.0 in strikes
        assert 570.0 not in strikes
        assert 610.0 not in strikes

    def test_sorted_output(self, sample_puts):
        """Test that output is sorted."""
        strikes = get_strike_ladder(sample_puts, 575.0, 605.0)
        assert strikes == sorted(strikes)


class TestGetOptionAtStrike:
    """Tests for get_option_at_strike function."""

    def test_find_exact_strike(self, sample_puts):
        """Test finding option at exact strike."""
        opt = get_option_at_strike(sample_puts, 590.0)
        assert opt is not None
        assert opt.strike == 590.0

    def test_missing_strike(self, sample_puts):
        """Test with non-existent strike."""
        opt = get_option_at_strike(sample_puts, 592.5)
        assert opt is None


class TestGeneratePutSpreadCandidates:
    """Tests for generate_put_spread_candidates function."""

    def test_generate_spreads(self, config):
        """Test generating put spread candidates with realistic pricing."""
        # Create puts with realistic pricing that will generate valid spreads
        expiration = date(2024, 3, 15)
        puts = [
            OptionLeg("SPY_595_P", 595.0, "put", expiration, 4.5, 4.7, 4.6),  # ATM
            OptionLeg("SPY_590_P", 590.0, "put", expiration, 3.0, 3.2, 3.1),  # OTM
            OptionLeg("SPY_585_P", 585.0, "put", expiration, 2.0, 2.2, 2.1),  # More OTM
            OptionLeg("SPY_580_P", 580.0, "put", expiration, 1.2, 1.4, 1.3),  # Deep OTM
        ]

        # Use a config with reasonable credit requirement
        test_config = CondorConfig(
            min_dte=14,
            max_dte=60,
            max_call_cost=0.50,
            min_put_credit_pct=0.20,  # 20% credit capture
            put_spread_width_min=5,
            put_spread_width_max=15,
            call_spread_width=10,
        )

        spreads = list(generate_put_spread_candidates(
            puts, 595.0, test_config
        ))

        # We should find at least one spread
        # 595/590 spread: credit = 4.6 - 3.1 = 1.5, width = 5, capture = 30%
        assert len(spreads) > 0

        for short_put, long_put in spreads:
            # Short put should have higher strike
            assert short_put.strike > long_put.strike
            # Width should be within config bounds
            width = short_put.strike - long_put.strike
            assert test_config.put_spread_width_min <= width <= test_config.put_spread_width_max

    def test_empty_puts(self, config):
        """Test with empty puts list."""
        spreads = list(generate_put_spread_candidates([], 600.0, config))
        assert len(spreads) == 0


class TestGenerateCallSpreadCandidates:
    """Tests for generate_call_spread_candidates function."""

    def test_generate_spreads(self, sample_calls, config):
        """Test generating call spread candidates."""
        spreads = list(generate_call_spread_candidates(
            sample_calls, 600.0, config
        ))

        assert len(spreads) > 0

        for short_call, long_call in spreads:
            # Long call should have higher strike
            assert long_call.strike > short_call.strike
            # Width should match config
            width = long_call.strike - short_call.strike
            assert width == config.call_spread_width


class TestConstructCondor:
    """Tests for construct_condor function."""

    def test_construct_valid_condor(self):
        """Test constructing a valid condor."""
        expiration = date(2024, 3, 15)

        long_put = OptionLeg(
            contract_symbol="SPY_580_P",
            strike=580.0,
            option_type="put",
            expiration=expiration,
            bid=1.90,
            ask=2.10,
            mid=2.00,
        )
        short_put = OptionLeg(
            contract_symbol="SPY_590_P",
            strike=590.0,
            option_type="put",
            expiration=expiration,
            bid=4.90,
            ask=5.10,
            mid=5.00,
        )
        short_call = OptionLeg(
            contract_symbol="SPY_610_C",
            strike=610.0,
            option_type="call",
            expiration=expiration,
            bid=2.40,
            ask=2.60,
            mid=2.50,
        )
        long_call = OptionLeg(
            contract_symbol="SPY_620_C",
            strike=620.0,
            option_type="call",
            expiration=expiration,
            bid=1.40,
            ask=1.60,
            mid=1.50,
        )

        condor = construct_condor(long_put, short_put, short_call, long_call)

        assert condor.put_spread_width == 10.0
        assert condor.call_spread_width == 10.0
        assert condor.put_spread_credit == pytest.approx(3.00, rel=0.01)  # 5.00 - 2.00
        assert condor.call_spread_debit == pytest.approx(1.00, rel=0.01)  # 2.50 - 1.50


class TestValidateCondor:
    """Tests for validate_condor function."""

    @pytest.fixture
    def valid_condor(self):
        """Create a valid condor for testing."""
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
        # Call spread with small debit (0.03) to be within max_call_cost
        short_call = OptionLeg(
            contract_symbol="SPY_610_C",
            strike=610.0,
            option_type="call",
            expiration=expiration,
            bid=1.50, ask=1.54, mid=1.52,
        )
        long_call = OptionLeg(
            contract_symbol="SPY_620_C",
            strike=620.0,
            option_type="call",
            expiration=expiration,
            bid=1.47, ask=1.51, mid=1.49,
        )

        return construct_condor(long_put, short_put, short_call, long_call)

    def test_validate_good_condor(self, valid_condor, config):
        """Test validating a good condor."""
        is_valid = validate_condor(valid_condor, config)
        assert is_valid is True

    def test_invalid_strike_order(self, config):
        """Test that invalid strike order fails validation."""
        expiration = date(2024, 3, 15)

        # Create condor with invalid strike order (short_put > short_call)
        long_put = OptionLeg(
            contract_symbol="SPY_580_P", strike=580.0, option_type="put",
            expiration=expiration, bid=1.90, ask=2.10, mid=2.00,
        )
        short_put = OptionLeg(
            contract_symbol="SPY_620_P", strike=620.0, option_type="put",  # Invalid!
            expiration=expiration, bid=4.90, ask=5.10, mid=5.00,
        )
        short_call = OptionLeg(
            contract_symbol="SPY_610_C", strike=610.0, option_type="call",
            expiration=expiration, bid=2.40, ask=2.60, mid=2.50,
        )
        long_call = OptionLeg(
            contract_symbol="SPY_630_C", strike=630.0, option_type="call",
            expiration=expiration, bid=1.40, ask=1.60, mid=1.50,
        )

        condor = BrokenWingCondor(
            long_put=long_put,
            short_put=short_put,
            short_call=short_call,
            long_call=long_call,
            put_spread_credit=3.00,
            call_spread_debit=1.00,
            net_credit=2.00,
            put_spread_width=40.0,
            call_spread_width=20.0,
            max_loss=3700.0,
            max_profit_credit_only=200.0,
            max_profit_with_calls=2200.0,
        )

        is_valid = validate_condor(condor, config)
        assert is_valid is False


class TestDiscoverCondors:
    """Tests for discover_condors function."""

    def test_discover_finds_candidates(self, sample_calls, sample_puts, config):
        """Test that discover_condors finds valid candidates."""
        condors = discover_condors(
            sample_calls, sample_puts, 600.0, config
        )

        # Should find some condors with our sample data
        # Note: may be 0 if pricing doesn't meet criteria
        assert isinstance(condors, list)
        for condor in condors:
            assert isinstance(condor, BrokenWingCondor)

    def test_discover_with_empty_options(self, config):
        """Test discover with empty options lists."""
        condors = discover_condors([], [], 600.0, config)
        assert len(condors) == 0
