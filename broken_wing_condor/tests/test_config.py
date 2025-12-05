"""Tests for configuration classes."""

import pytest

from broken_wing_condor.config import CondorConfig, ScoringWeights


class TestCondorConfig:
    """Tests for CondorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CondorConfig()

        assert config.symbol == "SPY"
        assert config.min_dte == 3
        assert config.max_dte == 14
        assert config.max_call_cost == 0.05
        assert config.min_put_credit_pct == 0.90
        assert config.top_n == 20

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CondorConfig(
            symbol="QQQ",
            min_dte=7,
            max_dte=21,
            max_call_cost=0.10,
            min_put_credit_pct=0.85,
            top_n=5,
        )

        assert config.symbol == "QQQ"
        assert config.min_dte == 7
        assert config.max_dte == 21
        assert config.max_call_cost == 0.10
        assert config.min_put_credit_pct == 0.85
        assert config.top_n == 5

    def test_invalid_dte_range(self):
        """Test that invalid DTE range raises error."""
        with pytest.raises(ValueError, match="max_dte must be >= min_dte"):
            CondorConfig(min_dte=60, max_dte=30)

    def test_invalid_max_call_cost(self):
        """Test that negative max_call_cost raises error."""
        with pytest.raises(ValueError, match="max_call_cost must be non-negative"):
            CondorConfig(max_call_cost=-0.05)

    def test_invalid_put_credit_pct(self):
        """Test that invalid put_credit_pct raises error."""
        with pytest.raises(ValueError, match="min_put_credit_pct must be between 0 and 1"):
            CondorConfig(min_put_credit_pct=1.5)

        with pytest.raises(ValueError, match="min_put_credit_pct must be between 0 and 1"):
            CondorConfig(min_put_credit_pct=-0.1)

    def test_invalid_top_n(self):
        """Test that invalid top_n raises error."""
        with pytest.raises(ValueError, match="top_n must be at least 1"):
            CondorConfig(top_n=0)

    def test_spread_width_validation(self):
        """Test spread width validation."""
        with pytest.raises(ValueError, match="put_spread_width_min must be at least 1"):
            CondorConfig(put_spread_width_min=0)

        with pytest.raises(ValueError, match="put_spread_width_max must be >= put_spread_width_min"):
            CondorConfig(put_spread_width_min=20, put_spread_width_max=10)


class TestScoringWeights:
    """Tests for ScoringWeights dataclass."""

    def test_default_weights(self):
        """Test default scoring weights."""
        weights = ScoringWeights()

        assert weights.risk_weight == 0.25
        assert weights.credit_weight == 0.20
        assert weights.skew_weight == 0.20
        assert weights.call_weight == 0.10
        assert weights.rrr_weight == 0.10
        assert weights.ev_weight == 0.10
        assert weights.pop_weight == 0.05

    def test_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        weights = ScoringWeights()

        total = (
            weights.risk_weight + weights.credit_weight + weights.skew_weight +
            weights.call_weight + weights.rrr_weight + weights.ev_weight + weights.pop_weight
        )

        assert total == pytest.approx(1.0, rel=0.001)

    def test_custom_weights(self):
        """Test custom scoring weights."""
        weights = ScoringWeights(
            risk_weight=0.30,
            credit_weight=0.25,
            skew_weight=0.15,
            call_weight=0.10,
            rrr_weight=0.10,
            ev_weight=0.05,
            pop_weight=0.05,
        )

        assert weights.risk_weight == 0.30
        assert weights.credit_weight == 0.25

    def test_weights_must_sum_to_one(self):
        """Test that weights not summing to 1.0 raises error."""
        with pytest.raises(ValueError, match="Scoring weights must sum to 1.0"):
            ScoringWeights(
                risk_weight=0.50,
                credit_weight=0.50,
                skew_weight=0.10,  # Total > 1.0
                call_weight=0.10,
                rrr_weight=0.10,
                ev_weight=0.10,
                pop_weight=0.05,
            )

    def test_validate_method(self):
        """Test validate method for non-negative weights."""
        weights = ScoringWeights()
        assert weights.validate() is True
