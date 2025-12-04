"""
Tests for the rule evaluation engine module.

This module tests all rule evaluation functionality including:
- Single rule evaluation
- Strategy evaluation with all/any modes
- Multi-strategy aggregation
- Signal generation
"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine.exceptions import MissingIndicatorError
from backtest.engine.rule_engine import (
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    evaluate_all_strategies,
    evaluate_rule,
    evaluate_strategy,
    generate_signals,
    get_column_case_insensitive,
    run_rule_engine,
    validate_indicators_present,
)
from backtest.engine.strategy_loader import Rule, Strategy


class TestValidateIndicatorsPresent:
    """Tests for validate_indicators_present function."""

    def test_all_indicators_present(self, sample_indicator_data):
        """Test validation passes when all indicators present."""
        rules = [Rule(indicator="rsi_14", operator="<", action="buy", value=30)]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)
        validate_indicators_present(sample_indicator_data, [strategy])  # Should not raise

    def test_missing_indicator(self, sample_indicator_data):
        """Test validation fails when indicator missing."""
        rules = [Rule(indicator="nonexistent_indicator", operator="<", action="buy", value=30)]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)
        with pytest.raises(MissingIndicatorError) as exc_info:
            validate_indicators_present(sample_indicator_data, [strategy])
        assert "nonexistent_indicator" in exc_info.value.missing_indicators

    def test_case_insensitive_match(self, sample_indicator_data):
        """Test that indicator matching is case-insensitive."""
        rules = [Rule(indicator="RSI_14", operator="<", action="buy", value=30)]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)
        validate_indicators_present(sample_indicator_data, [strategy])  # Should not raise

    def test_multiple_missing_indicators(self, sample_indicator_data):
        """Test multiple missing indicators are all reported."""
        rules = [
            Rule(indicator="missing1", operator="<", action="buy", value=30),
            Rule(indicator="missing2", operator=">", action="sell", value=70),
        ]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)
        with pytest.raises(MissingIndicatorError) as exc_info:
            validate_indicators_present(sample_indicator_data, [strategy])
        assert len(exc_info.value.missing_indicators) == 2


class TestGetColumnCaseInsensitive:
    """Tests for get_column_case_insensitive function."""

    def test_exact_match(self, sample_indicator_data):
        """Test exact case match."""
        series = get_column_case_insensitive(sample_indicator_data, "Close")
        assert len(series) == len(sample_indicator_data)

    def test_lowercase_match(self, sample_indicator_data):
        """Test lowercase match."""
        series = get_column_case_insensitive(sample_indicator_data, "close")
        assert len(series) == len(sample_indicator_data)

    def test_uppercase_match(self, sample_indicator_data):
        """Test uppercase match."""
        series = get_column_case_insensitive(sample_indicator_data, "CLOSE")
        assert len(series) == len(sample_indicator_data)

    def test_column_not_found(self, sample_indicator_data):
        """Test missing column raises KeyError."""
        with pytest.raises(KeyError):
            get_column_case_insensitive(sample_indicator_data, "nonexistent")


class TestEvaluateRule:
    """Tests for evaluate_rule function."""

    def test_less_than_operator(self):
        """Test < operator evaluation."""
        df = pd.DataFrame({"rsi_14": [25, 35, 45]})
        rule = Rule(indicator="rsi_14", operator="<", action="buy", value=30, strength=1.0)
        triggered, scores = evaluate_rule(df, rule)
        assert triggered.tolist() == [True, False, False]
        assert scores.iloc[0] == 1.0
        assert scores.iloc[1] == 0.0

    def test_less_than_or_equal_operator(self):
        """Test <= operator evaluation."""
        df = pd.DataFrame({"rsi_14": [25, 30, 35]})
        rule = Rule(indicator="rsi_14", operator="<=", action="buy", value=30, strength=1.0)
        triggered, scores = evaluate_rule(df, rule)
        assert triggered.tolist() == [True, True, False]

    def test_greater_than_operator(self):
        """Test > operator evaluation."""
        df = pd.DataFrame({"rsi_14": [65, 75, 85]})
        rule = Rule(indicator="rsi_14", operator=">", action="sell", value=70, strength=1.0)
        triggered, scores = evaluate_rule(df, rule)
        assert triggered.tolist() == [False, True, True]
        assert scores.iloc[1] == -1.0  # Sell action = negative score

    def test_greater_than_or_equal_operator(self):
        """Test >= operator evaluation."""
        df = pd.DataFrame({"rsi_14": [65, 70, 75]})
        rule = Rule(indicator="rsi_14", operator=">=", action="sell", value=70, strength=1.0)
        triggered, scores = evaluate_rule(df, rule)
        assert triggered.tolist() == [False, True, True]

    def test_equal_operator(self):
        """Test == operator evaluation."""
        df = pd.DataFrame({"rsi_14": [49, 50, 51]})
        rule = Rule(indicator="rsi_14", operator="==", action="buy", value=50, strength=1.0)
        triggered, scores = evaluate_rule(df, rule)
        assert triggered.tolist() == [False, True, False]

    def test_not_equal_operator(self):
        """Test != operator evaluation."""
        df = pd.DataFrame({"rsi_14": [49, 50, 51]})
        rule = Rule(indicator="rsi_14", operator="!=", action="buy", value=50, strength=1.0)
        triggered, scores = evaluate_rule(df, rule)
        assert triggered.tolist() == [True, False, True]

    def test_indicator_comparison(self):
        """Test comparing two indicators."""
        df = pd.DataFrame({
            "close": [100, 105, 95],
            "bb_upper_20_2": [102, 102, 102],
        })
        rule = Rule(
            indicator="close",
            operator=">",
            action="sell",
            value_indicator="bb_upper_20_2",
            strength=1.0,
        )
        triggered, scores = evaluate_rule(df, rule)
        assert triggered.tolist() == [False, True, False]

    def test_strength_affects_score(self):
        """Test that strength affects score magnitude."""
        df = pd.DataFrame({"rsi_14": [25]})
        rule = Rule(indicator="rsi_14", operator="<", action="buy", value=30, strength=0.5)
        triggered, scores = evaluate_rule(df, rule)
        assert scores.iloc[0] == 0.5

    def test_sell_action_negative_score(self):
        """Test that sell action produces negative score."""
        df = pd.DataFrame({"rsi_14": [75]})
        rule = Rule(indicator="rsi_14", operator=">", action="sell", value=70, strength=0.8)
        triggered, scores = evaluate_rule(df, rule)
        assert scores.iloc[0] == -0.8


class TestEvaluateStrategy:
    """Tests for evaluate_strategy function."""

    def test_any_mode_single_rule_triggered(self):
        """Test 'any' mode with single rule triggered."""
        df = pd.DataFrame({"rsi_14": [25, 50, 75]})
        rules = [
            Rule(indicator="rsi_14", operator="<", action="buy", value=30, strength=1.0),
            Rule(indicator="rsi_14", operator=">", action="sell", value=70, strength=1.0),
        ]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)
        scores = evaluate_strategy(df, strategy)
        assert scores.iloc[0] > 0  # Buy signal
        assert scores.iloc[1] == 0  # No signal
        assert scores.iloc[2] < 0  # Sell signal

    def test_all_mode_all_rules_required(self):
        """Test 'all' mode requires all rules to trigger."""
        df = pd.DataFrame({
            "rsi_14": [25, 25, 50],
            "macd_hist_12_26_9": [1, -1, 1],
        })
        rules = [
            Rule(indicator="rsi_14", operator="<", action="buy", value=30, strength=1.0),
            Rule(indicator="macd_hist_12_26_9", operator=">", action="buy", value=0, strength=1.0),
        ]
        strategy = Strategy(name="Test", weight=1.0, combine="all", rules=rules)
        scores = evaluate_strategy(df, strategy)
        assert scores.iloc[0] > 0  # Both rules triggered
        assert scores.iloc[1] == 0  # Only RSI triggered (MACD negative)
        assert scores.iloc[2] == 0  # Only MACD triggered (RSI too high)

    def test_empty_rules(self):
        """Test strategy with no rules returns zeros."""
        df = pd.DataFrame({"rsi_14": [25, 50, 75]})
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=[])
        scores = evaluate_strategy(df, strategy)
        assert (scores == 0).all()

    def test_buy_and_sell_cancel(self):
        """Test that simultaneous buy and sell signals cancel."""
        df = pd.DataFrame({"rsi_14": [50]})  # 50 triggers both rules
        rules = [
            Rule(indicator="rsi_14", operator="<", action="buy", value=60, strength=1.0),
            Rule(indicator="rsi_14", operator=">", action="sell", value=40, strength=1.0),
        ]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)
        scores = evaluate_strategy(df, strategy)
        assert scores.iloc[0] == 0  # Buy and sell cancel out


class TestEvaluateAllStrategies:
    """Tests for evaluate_all_strategies function."""

    def test_single_strategy(self, sample_indicator_data):
        """Test evaluating a single strategy."""
        rules = [Rule(indicator="rsi_14", operator="<", action="buy", value=30, strength=1.0)]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)
        scores = evaluate_all_strategies(sample_indicator_data, [strategy])
        assert len(scores) == len(sample_indicator_data)

    def test_weighted_aggregation(self):
        """Test that strategy weights affect aggregation."""
        df = pd.DataFrame({"rsi_14": [25]})

        strategy1 = Strategy(
            name="S1",
            weight=0.7,
            combine="any",
            rules=[Rule(indicator="rsi_14", operator="<", action="buy", value=30, strength=1.0)],
        )
        strategy2 = Strategy(
            name="S2",
            weight=0.3,
            combine="any",
            rules=[Rule(indicator="rsi_14", operator=">", action="sell", value=20, strength=1.0)],
        )

        scores = evaluate_all_strategies(df, [strategy1, strategy2])
        # 0.7 * 1.0 + 0.3 * (-1.0) = 0.4
        assert abs(scores.iloc[0] - 0.4) < 0.0001

    def test_empty_strategies(self, sample_indicator_data):
        """Test empty strategies list returns zeros."""
        scores = evaluate_all_strategies(sample_indicator_data, [])
        assert (scores == 0).all()


class TestGenerateSignals:
    """Tests for generate_signals function."""

    def test_buy_signal(self):
        """Test BUY signal generation."""
        scores = pd.Series([BUY_THRESHOLD, BUY_THRESHOLD + 0.1, 0.5])
        signals = generate_signals(scores)
        assert (signals == "BUY").all()

    def test_sell_signal(self):
        """Test SELL signal generation."""
        scores = pd.Series([SELL_THRESHOLD, SELL_THRESHOLD - 0.1, -0.5])
        signals = generate_signals(scores)
        assert (signals == "SELL").all()

    def test_hold_signal(self):
        """Test HOLD signal generation."""
        scores = pd.Series([0, 0.1, -0.1, BUY_THRESHOLD - 0.01, SELL_THRESHOLD + 0.01])
        signals = generate_signals(scores)
        assert (signals == "HOLD").all()

    def test_mixed_signals(self):
        """Test mixed signal generation."""
        scores = pd.Series([0.5, -0.5, 0])
        signals = generate_signals(scores)
        assert signals.iloc[0] == "BUY"
        assert signals.iloc[1] == "SELL"
        assert signals.iloc[2] == "HOLD"


class TestRunRuleEngine:
    """Tests for run_rule_engine function."""

    def test_full_pipeline(self, sample_indicator_data, simple_strategy_yaml):
        """Test running the full rule engine pipeline."""
        from backtest.engine.strategy_loader import load_strategies, normalize_weights

        strategies = load_strategies(str(simple_strategy_yaml))
        strategies = normalize_weights(strategies)

        signals, scores = run_rule_engine(sample_indicator_data, strategies)

        assert len(signals) == len(sample_indicator_data)
        assert len(scores) == len(sample_indicator_data)
        assert set(signals.unique()).issubset({"BUY", "SELL", "HOLD"})

    def test_missing_indicator_raises(self, sample_indicator_data):
        """Test that missing indicator raises error."""
        rules = [Rule(indicator="nonexistent", operator="<", action="buy", value=30)]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)

        with pytest.raises(MissingIndicatorError):
            run_rule_engine(sample_indicator_data, [strategy])

    def test_trending_data_signals(self, trending_up_data, simple_strategy_yaml):
        """Test signals on trending data."""
        from backtest.engine.strategy_loader import load_strategies, normalize_weights

        strategies = load_strategies(str(simple_strategy_yaml))
        strategies = normalize_weights(strategies)

        signals, scores = run_rule_engine(trending_up_data, strategies)

        # With RSI starting at 25 and ending at 75:
        # Should have BUY signals early (RSI < 30)
        # Should have SELL signals late (RSI > 70)
        assert "BUY" in signals.values
        assert "SELL" in signals.values

    def test_oscillating_data_multiple_signals(self, oscillating_data, simple_strategy_yaml):
        """Test multiple buy/sell signals on oscillating data."""
        from backtest.engine.strategy_loader import load_strategies, normalize_weights

        strategies = load_strategies(str(simple_strategy_yaml))
        strategies = normalize_weights(strategies)

        signals, scores = run_rule_engine(oscillating_data, strategies)

        # Should have multiple buy and sell signals
        buy_count = (signals == "BUY").sum()
        sell_count = (signals == "SELL").sum()

        assert buy_count > 0
        assert sell_count > 0


class TestEvaluateStrategyEdgeCases:
    """Edge case tests for evaluate_strategy."""

    def test_all_mode_no_buy_rules(self):
        """Test 'all' mode with only sell rules."""
        df = pd.DataFrame({"rsi_14": [75, 80, 85]})
        rules = [
            Rule(indicator="rsi_14", operator=">", action="sell", value=70, strength=1.0),
        ]
        strategy = Strategy(name="Test", weight=1.0, combine="all", rules=rules)
        scores = evaluate_strategy(df, strategy)
        assert all(scores < 0)  # All should be sell signals

    def test_all_mode_no_sell_rules(self):
        """Test 'all' mode with only buy rules."""
        df = pd.DataFrame({"rsi_14": [25, 20, 15]})
        rules = [
            Rule(indicator="rsi_14", operator="<", action="buy", value=30, strength=1.0),
        ]
        strategy = Strategy(name="Test", weight=1.0, combine="all", rules=rules)
        scores = evaluate_strategy(df, strategy)
        assert all(scores > 0)  # All should be buy signals

    def test_any_mode_no_buy_rules(self):
        """Test 'any' mode with only sell rules."""
        df = pd.DataFrame({"rsi_14": [75, 80, 85]})
        rules = [
            Rule(indicator="rsi_14", operator=">", action="sell", value=70, strength=1.0),
        ]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)
        scores = evaluate_strategy(df, strategy)
        assert all(scores < 0)

    def test_any_mode_no_sell_rules(self):
        """Test 'any' mode with only buy rules."""
        df = pd.DataFrame({"rsi_14": [25, 20, 15]})
        rules = [
            Rule(indicator="rsi_14", operator="<", action="buy", value=30, strength=1.0),
        ]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)
        scores = evaluate_strategy(df, strategy)
        assert all(scores > 0)

    def test_invalid_operator_returns_no_trigger(self):
        """Test that invalid operator in rule returns no triggers."""
        df = pd.DataFrame({"rsi_14": [25, 50, 75]})
        rule = Rule(indicator="rsi_14", operator="??", action="buy", value=30, strength=1.0)
        triggered, scores = evaluate_rule(df, rule)
        assert not triggered.any()
        assert (scores == 0).all()


class TestLagIndicators:
    """Tests for lag indicator support."""

    def test_parse_lag_indicator_with_lag(self):
        """Test parsing indicator with lag suffix."""
        from backtest.engine.rule_engine import parse_lag_indicator

        base, lag = parse_lag_indicator("ema_21_lag_1")
        assert base == "ema_21"
        assert lag == 1

    def test_parse_lag_indicator_without_lag(self):
        """Test parsing indicator without lag suffix."""
        from backtest.engine.rule_engine import parse_lag_indicator

        base, lag = parse_lag_indicator("rsi_14")
        assert base == "rsi_14"
        assert lag is None

    def test_parse_lag_indicator_multiple_digits(self):
        """Test parsing indicator with multi-digit lag."""
        from backtest.engine.rule_engine import parse_lag_indicator

        base, lag = parse_lag_indicator("close_lag_10")
        assert base == "close"
        assert lag == 10

    def test_get_base_indicators(self):
        """Test getting base indicators from mixed list."""
        from backtest.engine.rule_engine import get_base_indicators

        indicators = ["ema_21", "ema_21_lag_1", "rsi_14", "close_lag_5"]
        base = get_base_indicators(indicators)
        assert set(base) == {"ema_21", "rsi_14", "close"}

    def test_get_column_with_lag(self):
        """Test getting column with lag applied."""
        df = pd.DataFrame({
            "ema_21": [100, 101, 102, 103, 104],
        })
        result = get_column_case_insensitive(df, "ema_21_lag_1")
        assert pd.isna(result.iloc[0])  # First value is NaN due to shift
        assert result.iloc[1] == 100
        assert result.iloc[2] == 101

    def test_get_column_with_lag_2(self):
        """Test getting column with lag of 2."""
        df = pd.DataFrame({
            "close": [100, 101, 102, 103, 104],
        })
        result = get_column_case_insensitive(df, "close_lag_2")
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 100
        assert result.iloc[3] == 101

    def test_validate_lag_indicator_present(self):
        """Test validation accepts lag indicator if base exists."""
        df = pd.DataFrame({
            "ema_21": [100, 101, 102],
            "Close": [100, 101, 102],
        })
        rules = [
            Rule(indicator="ema_21", operator=">", action="buy",
                 value_indicator="ema_21_lag_1", strength=1.0),
        ]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)
        # Should not raise - ema_21_lag_1 is valid because ema_21 exists
        validate_indicators_present(df, [strategy])

    def test_validate_lag_indicator_missing_base(self):
        """Test validation fails if base indicator missing."""
        df = pd.DataFrame({
            "rsi_14": [50, 55, 60],
            "Close": [100, 101, 102],
        })
        rules = [
            Rule(indicator="rsi_14", operator=">", action="buy",
                 value_indicator="ema_21_lag_1", strength=1.0),
        ]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)
        with pytest.raises(MissingIndicatorError) as exc_info:
            validate_indicators_present(df, [strategy])
        assert "ema_21" in exc_info.value.missing_indicators

    def test_evaluate_rule_with_lag_indicator(self):
        """Test rule evaluation with lag indicator comparison."""
        df = pd.DataFrame({
            "ema_21": [100, 101, 102, 103, 104],  # Rising values
        })
        rule = Rule(
            indicator="ema_21",
            operator=">",
            action="buy",
            value_indicator="ema_21_lag_1",
            strength=1.0,
        )
        triggered, scores = evaluate_rule(df, rule)
        # First row: ema_21[0]=100 > NaN = False
        # Row 1: ema_21[1]=101 > ema_21[0]=100 = True
        # Row 2: ema_21[2]=102 > ema_21[1]=101 = True
        assert not triggered.iloc[0]  # NaN comparison is False
        assert triggered.iloc[1]
        assert triggered.iloc[2]
        assert triggered.iloc[3]
        assert triggered.iloc[4]

    def test_evaluate_rule_with_lag_indicator_declining(self):
        """Test rule evaluation with lag indicator for declining values."""
        df = pd.DataFrame({
            "ema_21": [104, 103, 102, 101, 100],  # Declining values
        })
        rule = Rule(
            indicator="ema_21",
            operator="<",
            action="sell",
            value_indicator="ema_21_lag_1",
            strength=1.0,
        )
        triggered, scores = evaluate_rule(df, rule)
        # First row: NaN comparison = False
        # Row 1: 103 < 104 = True (declining)
        assert not triggered.iloc[0]
        assert triggered.iloc[1]
        assert triggered.iloc[2]
        assert scores.iloc[1] == -1.0  # Sell action

    def test_full_pipeline_with_lag_indicator(self):
        """Test full rule engine with lag indicators."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=10),
            "Close": [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
            "ema_21": [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
        })
        rules = [
            Rule(indicator="ema_21", operator=">", action="buy",
                 value_indicator="ema_21_lag_1", strength=1.0),
            Rule(indicator="ema_21", operator="<", action="sell",
                 value_indicator="ema_21_lag_1", strength=1.0),
        ]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)

        signals, scores = run_rule_engine(df, [strategy])

        # Rising period (rows 1-5): should have BUY signals
        # Declining period (rows 6-9): should have SELL signals
        assert len(signals) == 10
        assert "BUY" in signals.values
        assert "SELL" in signals.values
