"""
Unit tests for the scoring engine module.

Tests all scoring functions including:
- Rule score evaluation
- Strategy score aggregation (combine="all" vs combine="any")
- Global score aggregation with weights
- Score range invariants
"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine.constants import Action, CombineMode
from backtest.engine.scoring import (
    evaluate_rule_score,
    evaluate_strategy_score,
    aggregate_global_score,
    build_scores_dataframe,
    run_scoring_engine,
    RuleScore,
    StrategyScore,
    ScoringResult,
)
from backtest.engine.strategy_loader import Rule, Strategy


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with indicator data."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "close": [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0, 105.0],
        "rsi_14": [45.0, 35.0, 25.0, 40.0, 20.0, 50.0, 30.0, 55.0, 28.0, 60.0],
        "mfi_14": [50.0, 40.0, 20.0, 45.0, 15.0, 55.0, 25.0, 60.0, 18.0, 65.0],
        "sma_200": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "bb_lower_20": [95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0],
        "bb_upper_20": [105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0],
        "bb_middle_20": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "macd_hist": [0.5, -0.3, 0.8, -0.5, 1.0, -0.2, 0.3, -0.8, 0.2, 0.6],
    })


@pytest.fixture
def buy_rule():
    """Create a simple buy rule."""
    return Rule(
        indicator="rsi_14",
        operator="<",
        value=30.0,
        action=Action.BUY,
        strength=1.0,
    )


@pytest.fixture
def sell_rule():
    """Create a simple sell rule."""
    return Rule(
        indicator="rsi_14",
        operator=">",
        value=70.0,
        action=Action.SELL,
        strength=1.0,
    )


class TestEvaluateRuleScore:
    """Tests for evaluate_rule_score function."""

    def test_buy_rule_score_positive(self, sample_df, buy_rule):
        """Buy rule should produce positive scores when activated."""
        rs = evaluate_rule_score(sample_df, buy_rule)

        assert isinstance(rs, RuleScore)
        assert len(rs.activation) == len(sample_df)
        assert len(rs.score) == len(sample_df)
        # Buy rule: action_sign = +1, so positive activation -> positive score
        assert (rs.score >= 0.0).all()

    def test_sell_rule_score_negative(self, sample_df, sell_rule):
        """Sell rule should produce negative scores when activated."""
        rs = evaluate_rule_score(sample_df, sell_rule)

        # Sell rule: action_sign = -1, so positive activation -> negative score
        assert (rs.score <= 0.0).all()

    def test_activation_range(self, sample_df, buy_rule):
        """Activation should always be in [0, 1]."""
        rs = evaluate_rule_score(sample_df, buy_rule)

        assert (rs.activation >= 0.0).all()
        assert (rs.activation <= 1.0).all()

    def test_score_formula(self, sample_df):
        """Verify score = action_sign * strength * activation."""
        rule = Rule(
            indicator="rsi_14",
            operator="<",
            value=30.0,
            action=Action.BUY,
            strength=0.5,
        )
        rs = evaluate_rule_score(sample_df, rule)

        # action_sign = +1 for buy, strength = 0.5
        expected_score = 1.0 * 0.5 * rs.activation
        pd.testing.assert_series_equal(rs.score, expected_score, check_names=False)

    def test_rule_name_with_value(self, sample_df, buy_rule):
        """Rule name should include threshold value."""
        rs = evaluate_rule_score(sample_df, buy_rule, rule_index=0)

        assert "rsi_14" in rs.rule_name
        assert "30.0" in rs.rule_name

    def test_rule_with_value_indicator(self, sample_df):
        """Rule comparing two indicators should work."""
        rule = Rule(
            indicator="close",
            operator="<",
            value_indicator="sma_200",
            action=Action.SELL,
            strength=1.0,
        )
        rs = evaluate_rule_score(sample_df, rule)

        assert len(rs.activation) == len(sample_df)
        assert "sma_200" in rs.rule_name


class TestEvaluateStrategyScore:
    """Tests for evaluate_strategy_score function."""

    def test_empty_rules(self, sample_df):
        """Strategy with no rules should return zero score."""
        strategy = Strategy(
            name="empty",
            rules=[],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)

        assert (ss.raw_score == 0.0).all()

    def test_single_rule(self, sample_df, buy_rule):
        """Strategy with single rule should match rule score."""
        strategy = Strategy(
            name="single_rule",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)

        assert len(ss.rule_scores) == 1
        # Raw score should be clipped version of rule score
        assert (ss.raw_score >= -1.0).all()
        assert (ss.raw_score <= 1.0).all()

    def test_combine_any_sums_scores(self, sample_df):
        """combine='any' should sum rule scores."""
        rule1 = Rule(
            indicator="rsi_14",
            operator="<",
            value=30.0,
            action=Action.BUY,
            strength=0.3,
        )
        rule2 = Rule(
            indicator="mfi_14",
            operator="<",
            value=25.0,
            action=Action.BUY,
            strength=0.3,
        )
        strategy = Strategy(
            name="combine_any",
            rules=[rule1, rule2],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)

        # Raw score should be sum of rule scores, clipped
        rule_sum = ss.rule_scores[0].score + ss.rule_scores[1].score
        expected = rule_sum.clip(lower=-1.0, upper=1.0)
        pd.testing.assert_series_equal(ss.raw_score, expected, check_names=False)

    def test_combine_all_uses_min_activation(self, sample_df):
        """combine='all' should gate by minimum activation."""
        rule1 = Rule(
            indicator="rsi_14",
            operator="<",
            value=30.0,
            action=Action.BUY,
            strength=0.5,
        )
        rule2 = Rule(
            indicator="mfi_14",
            operator="<",
            value=25.0,
            action=Action.BUY,
            strength=0.5,
        )
        strategy = Strategy(
            name="combine_all",
            rules=[rule1, rule2],
            combine=CombineMode.ALL,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)

        # For combine="all", score should be less than or equal to combine="any"
        strategy_any = Strategy(
            name="combine_any",
            rules=[rule1, rule2],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss_any = evaluate_strategy_score(sample_df, strategy_any)

        # Combine all should generally produce lower or equal scores
        # (due to minimum activation gating)
        assert (ss.raw_score.abs() <= ss_any.raw_score.abs() + 0.001).all()

    def test_raw_score_clipped(self, sample_df):
        """Raw score should be clipped to [-1, +1]."""
        # Create rules that would sum to more than 1
        rules = [
            Rule(
                indicator="rsi_14",
                operator="<",
                value=50.0,  # High threshold to ensure activation
                action=Action.BUY,
                strength=1.0,
            ),
            Rule(
                indicator="mfi_14",
                operator="<",
                value=50.0,
                action=Action.BUY,
                strength=1.0,
            ),
        ]
        strategy = Strategy(
            name="high_score",
            rules=rules,
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)

        assert (ss.raw_score >= -1.0).all()
        assert (ss.raw_score <= 1.0).all()

    def test_mixed_buy_sell_rules(self, sample_df):
        """Strategy with both buy and sell rules should handle both."""
        buy_rule = Rule(
            indicator="rsi_14",
            operator="<",
            value=30.0,
            action=Action.BUY,
            strength=0.5,
        )
        sell_rule = Rule(
            indicator="rsi_14",
            operator=">",
            value=70.0,
            action=Action.SELL,
            strength=0.5,
        )
        strategy = Strategy(
            name="mixed",
            rules=[buy_rule, sell_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)

        # Score should be the sum of buy (positive) and sell (negative) contributions
        assert len(ss.rule_scores) == 2


class TestAggregateGlobalScore:
    """Tests for aggregate_global_score function."""

    def test_empty_strategies(self, sample_df):
        """Empty strategies should return neutral score."""
        signal_raw, signal_0_1 = aggregate_global_score([], sample_df.index)

        assert (signal_raw == 0.0).all()
        assert (signal_0_1 == 0.5).all()

    def test_single_strategy(self, sample_df, buy_rule):
        """Single strategy should return its weighted score."""
        strategy = Strategy(
            name="single",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)

        signal_raw, signal_0_1 = aggregate_global_score([ss], sample_df.index)

        # With single strategy at weight 1.0, should equal strategy score
        pd.testing.assert_series_equal(signal_raw, ss.raw_score, check_names=False)

    def test_weighted_aggregation(self, sample_df):
        """Multiple strategies should be weighted correctly."""
        rule1 = Rule(
            indicator="rsi_14",
            operator="<",
            value=30.0,
            action=Action.BUY,
            strength=1.0,
        )
        rule2 = Rule(
            indicator="rsi_14",
            operator=">",
            value=70.0,
            action=Action.SELL,
            strength=1.0,
        )

        strategy1 = Strategy(
            name="buy_strategy",
            rules=[rule1],
            combine=CombineMode.ANY,
            weight=0.7,
        )
        strategy2 = Strategy(
            name="sell_strategy",
            rules=[rule2],
            combine=CombineMode.ANY,
            weight=0.3,
        )

        ss1 = evaluate_strategy_score(sample_df, strategy1)
        ss2 = evaluate_strategy_score(sample_df, strategy2)

        signal_raw, signal_0_1 = aggregate_global_score([ss1, ss2], sample_df.index)

        # Verify weighted sum
        total_weight = 0.7 + 0.3
        expected = (0.7 / total_weight * ss1.raw_score +
                   0.3 / total_weight * ss2.raw_score)
        expected = expected.clip(lower=-1.0, upper=1.0)

        pd.testing.assert_series_equal(signal_raw, expected, check_names=False)

    def test_signal_0_1_transformation(self, sample_df, buy_rule):
        """signal_0_1 should be (signal_raw + 1) / 2."""
        strategy = Strategy(
            name="test",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)

        signal_raw, signal_0_1 = aggregate_global_score([ss], sample_df.index)

        expected_0_1 = (signal_raw + 1.0) / 2.0
        pd.testing.assert_series_equal(signal_0_1, expected_0_1, check_names=False)

    def test_signal_ranges(self, sample_df, buy_rule):
        """Signals should always be in correct ranges."""
        strategy = Strategy(
            name="test",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)

        signal_raw, signal_0_1 = aggregate_global_score([ss], sample_df.index)

        assert (signal_raw >= -1.0).all()
        assert (signal_raw <= 1.0).all()
        assert (signal_0_1 >= 0.0).all()
        assert (signal_0_1 <= 1.0).all()


class TestBuildScoresDataframe:
    """Tests for build_scores_dataframe function."""

    def test_includes_original_columns(self, sample_df, buy_rule):
        """Result should include all original columns."""
        strategy = Strategy(
            name="test",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)
        signal_raw, signal_0_1 = aggregate_global_score([ss], sample_df.index)

        result = build_scores_dataframe(
            sample_df, [ss], signal_raw, signal_0_1
        )

        for col in sample_df.columns:
            assert col in result.columns

    def test_includes_strategy_scores(self, sample_df, buy_rule):
        """Result should include strategy score columns."""
        strategy = Strategy(
            name="test strategy",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)
        signal_raw, signal_0_1 = aggregate_global_score([ss], sample_df.index)

        result = build_scores_dataframe(
            sample_df, [ss], signal_raw, signal_0_1
        )

        assert "score_test_strategy" in result.columns

    def test_includes_global_scores(self, sample_df, buy_rule):
        """Result should include signal_raw and signal_0_1."""
        strategy = Strategy(
            name="test",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)
        signal_raw, signal_0_1 = aggregate_global_score([ss], sample_df.index)

        result = build_scores_dataframe(
            sample_df, [ss], signal_raw, signal_0_1
        )

        assert "signal_raw" in result.columns
        assert "signal_0_1" in result.columns

    def test_include_rule_scores(self, sample_df, buy_rule):
        """Should include individual rule scores when requested."""
        strategy = Strategy(
            name="test",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        ss = evaluate_strategy_score(sample_df, strategy)
        signal_raw, signal_0_1 = aggregate_global_score([ss], sample_df.index)

        result = build_scores_dataframe(
            sample_df, [ss], signal_raw, signal_0_1, include_rule_scores=True
        )

        # Should have rule score and activation columns
        rule_cols = [c for c in result.columns if "rule_0" in c]
        assert len(rule_cols) > 0


class TestRunScoringEngine:
    """Tests for run_scoring_engine integration function."""

    def test_returns_scoring_result(self, sample_df, buy_rule):
        """Should return a ScoringResult object."""
        strategy = Strategy(
            name="test",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        result = run_scoring_engine(sample_df, [strategy])

        assert isinstance(result, ScoringResult)
        assert isinstance(result.signal_raw, pd.Series)
        assert isinstance(result.signal_0_1, pd.Series)
        assert isinstance(result.scores_df, pd.DataFrame)

    def test_multiple_strategies(self, sample_df):
        """Should handle multiple strategies."""
        rule1 = Rule(
            indicator="rsi_14",
            operator="<",
            value=30.0,
            action=Action.BUY,
            strength=0.5,
        )
        rule2 = Rule(
            indicator="mfi_14",
            operator="<",
            value=25.0,
            action=Action.BUY,
            strength=0.5,
        )

        strategies = [
            Strategy(name="s1", rules=[rule1], combine=CombineMode.ANY, weight=0.6),
            Strategy(name="s2", rules=[rule2], combine=CombineMode.ANY, weight=0.4),
        ]

        result = run_scoring_engine(sample_df, strategies)

        assert len(result.strategy_scores) == 2
        assert "score_s1" in result.scores_df.columns
        assert "score_s2" in result.scores_df.columns

    def test_get_strategy_score(self, sample_df, buy_rule):
        """Should be able to retrieve specific strategy score."""
        strategy = Strategy(
            name="test_strategy",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        result = run_scoring_engine(sample_df, [strategy])

        score = result.get_strategy_score("test_strategy")
        assert score is not None
        assert len(score) == len(sample_df)

        # Non-existent strategy should return None
        assert result.get_strategy_score("nonexistent") is None

    def test_signal_range_invariants(self, sample_df, buy_rule):
        """Signal values should always be in correct ranges."""
        strategy = Strategy(
            name="test",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )
        result = run_scoring_engine(sample_df, [strategy])

        assert (result.signal_raw >= -1.0).all()
        assert (result.signal_raw <= 1.0).all()
        assert (result.signal_0_1 >= 0.0).all()
        assert (result.signal_0_1 <= 1.0).all()


class TestScoringEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe_raises_error(self):
        """Empty DataFrame should raise ValueError."""
        df = pd.DataFrame()
        rule = Rule(
            indicator="rsi_14",
            operator="<",
            value=30.0,
            action=Action.BUY,
            strength=1.0,
        )
        strategy = Strategy(
            name="test",
            rules=[rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )

        with pytest.raises(ValueError, match="empty DataFrame"):
            run_scoring_engine(df, [strategy])

    def test_all_zero_activations(self):
        """Strategy with no activations should produce zero score."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "rsi_14": [50.0, 55.0, 60.0, 65.0, 70.0],  # All above 30
        })

        rule = Rule(
            indicator="rsi_14",
            operator="<",
            value=30.0,  # Never activated
            action=Action.BUY,
            strength=1.0,
        )
        strategy = Strategy(
            name="no_activation",
            rules=[rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )

        result = run_scoring_engine(df, [strategy])

        # No activation means zero score
        assert (result.signal_raw == 0.0).all()
        assert (result.signal_0_1 == 0.5).all()

    def test_full_activation(self):
        """Strategy with full activations should produce max score."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "rsi_14": [0.0, 5.0, 10.0, 15.0, 20.0],  # All deeply oversold
        })

        rule = Rule(
            indicator="rsi_14",
            operator="<",
            value=30.0,
            action=Action.BUY,
            strength=1.0,
        )
        strategy = Strategy(
            name="full_activation",
            rules=[rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )

        result = run_scoring_engine(df, [strategy])

        # RSI at 0 should give activation = 1.0
        assert result.signal_raw.iloc[0] == 1.0
        assert result.signal_0_1.iloc[0] == 1.0

    def test_zero_weight_strategy(self, sample_df, buy_rule):
        """Strategy with zero weight should not contribute."""
        strategy1 = Strategy(
            name="zero_weight",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=0.0,
        )
        strategy2 = Strategy(
            name="full_weight",
            rules=[buy_rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )

        result = run_scoring_engine(sample_df, [strategy1, strategy2])

        # Result should be dominated by full_weight strategy
        # (though total weight normalization will make zero-weight irrelevant)
        assert len(result.strategy_scores) == 2

    def test_nan_handling(self):
        """Should handle NaN values gracefully."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "rsi_14": [25.0, np.nan, 20.0, np.nan, 15.0],
        })

        rule = Rule(
            indicator="rsi_14",
            operator="<",
            value=30.0,
            action=Action.BUY,
            strength=1.0,
        )
        strategy = Strategy(
            name="nan_test",
            rules=[rule],
            combine=CombineMode.ANY,
            weight=1.0,
        )

        result = run_scoring_engine(df, [strategy])

        # Non-NaN values should be computed correctly
        assert result.signal_raw.iloc[0] > 0.0
        assert result.signal_raw.iloc[2] > 0.0
        assert result.signal_raw.iloc[4] > 0.0


class TestScoringIntegration:
    """Integration tests simulating real strategy configurations."""

    def test_mean_reversion_strategy(self, sample_df):
        """Test mean reversion strategy (BB + RSI)."""
        rules = [
            Rule(
                indicator="close",
                operator="<",
                value_indicator="bb_lower_20",
                action=Action.BUY,
                strength=0.5,
            ),
            Rule(
                indicator="rsi_14",
                operator="<",
                value=30.0,
                action=Action.BUY,
                strength=0.5,
            ),
        ]
        strategy = Strategy(
            name="MeanReversion",
            rules=rules,
            combine=CombineMode.ANY,
            weight=1.0,
        )

        result = run_scoring_engine(sample_df, [strategy])

        assert isinstance(result, ScoringResult)
        assert "score_MeanReversion" in result.scores_df.columns

    def test_multi_strategy_portfolio(self, sample_df):
        """Test multiple strategies with different weights."""
        strategies = [
            Strategy(
                name="MeanReversion",
                rules=[
                    Rule(
                        indicator="rsi_14",
                        operator="<",
                        value=30.0,
                        action=Action.BUY,
                        strength=1.0,
                    ),
                ],
                combine=CombineMode.ANY,
                weight=0.4,
            ),
            Strategy(
                name="TrendFollowing",
                rules=[
                    Rule(
                        indicator="close",
                        operator=">",
                        value_indicator="sma_200",
                        action=Action.BUY,
                        strength=1.0,
                    ),
                ],
                combine=CombineMode.ANY,
                weight=0.6,
            ),
        ]

        result = run_scoring_engine(sample_df, strategies)

        assert len(result.strategy_scores) == 2

        # Verify weight normalization
        # 0.4 + 0.6 = 1.0, so weights should be 0.4 and 0.6
        total_weight = sum(ss.weight for ss in result.strategy_scores)
        assert abs(total_weight - 1.0) < 0.001
