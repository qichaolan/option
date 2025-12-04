"""
Tests for the strategy loader module.

This module tests all strategy loading and validation functionality including:
- Valid YAML loading
- Invalid YAML handling
- Rule validation
- Strategy validation
- Multi-strategy loading
"""

import pytest

from backtest.engine.exceptions import (
    FileNotFoundError,
    InvalidRuleError,
    InvalidStrategyError,
)
from backtest.engine.strategy_loader import (
    VALID_ACTIONS,
    VALID_COMBINE_MODES,
    VALID_OPERATORS,
    Rule,
    Strategy,
    get_all_required_indicators,
    load_strategies,
    load_strategy_file,
    normalize_weights,
    validate_rule,
    validate_strategy,
)


class TestRule:
    """Tests for Rule class."""

    def test_rule_creation_with_value(self):
        """Test creating a rule with static value."""
        rule = Rule(
            indicator="rsi_14",
            operator="<",
            action="buy",
            strength=1.0,
            value=30,
        )
        assert rule.indicator == "rsi_14"
        assert rule.operator == "<"
        assert rule.action == "buy"
        assert rule.strength == 1.0
        assert rule.value == 30
        assert rule.value_indicator is None

    def test_rule_creation_with_indicator(self):
        """Test creating a rule with indicator comparison."""
        rule = Rule(
            indicator="close",
            operator=">",
            action="sell",
            strength=0.8,
            value_indicator="bb_upper_20_2",
        )
        assert rule.indicator == "close"
        assert rule.value is None
        assert rule.value_indicator == "bb_upper_20_2"

    def test_rule_repr_with_value(self):
        """Test rule string representation with value."""
        rule = Rule(
            indicator="rsi_14",
            operator="<",
            action="buy",
            strength=1.0,
            value=30,
        )
        repr_str = repr(rule)
        assert "rsi_14" in repr_str
        assert "30" in repr_str
        assert "buy" in repr_str

    def test_rule_repr_with_indicator(self):
        """Test rule string representation with indicator."""
        rule = Rule(
            indicator="close",
            operator=">",
            action="sell",
            strength=0.8,
            value_indicator="bb_upper_20_2",
        )
        repr_str = repr(rule)
        assert "close" in repr_str
        assert "bb_upper_20_2" in repr_str

    def test_rule_case_insensitive(self):
        """Test that rule indicators and actions are case-insensitive."""
        rule = Rule(
            indicator="RSI_14",
            operator="<",
            action="BUY",
            strength=1.0,
            value=30,
        )
        assert rule.indicator == "rsi_14"
        assert rule.action == "buy"


class TestStrategy:
    """Tests for Strategy class."""

    def test_strategy_creation(self):
        """Test creating a strategy."""
        rule = Rule(
            indicator="rsi_14",
            operator="<",
            action="buy",
            strength=1.0,
            value=30,
        )
        strategy = Strategy(
            name="TestStrategy",
            weight=1.0,
            combine="any",
            rules=[rule],
        )
        assert strategy.name == "TestStrategy"
        assert strategy.weight == 1.0
        assert strategy.combine == "any"
        assert len(strategy.rules) == 1

    def test_strategy_repr(self):
        """Test strategy string representation."""
        rule = Rule(
            indicator="rsi_14",
            operator="<",
            action="buy",
            strength=1.0,
            value=30,
        )
        strategy = Strategy(
            name="TestStrategy",
            weight=0.5,
            combine="all",
            rules=[rule],
        )
        repr_str = repr(strategy)
        assert "TestStrategy" in repr_str
        assert "0.5" in repr_str
        assert "all" in repr_str

    def test_get_required_indicators(self):
        """Test getting required indicators from strategy."""
        rules = [
            Rule(indicator="rsi_14", operator="<", action="buy", strength=1.0, value=30),
            Rule(
                indicator="close",
                operator=">",
                action="sell",
                strength=1.0,
                value_indicator="bb_upper_20_2",
            ),
        ]
        strategy = Strategy(
            name="Test", weight=1.0, combine="any", rules=rules
        )
        indicators = strategy.get_required_indicators()
        assert "rsi_14" in indicators
        assert "close" in indicators
        assert "bb_upper_20_2" in indicators


class TestValidateRule:
    """Tests for validate_rule function."""

    def test_valid_rule_with_value(self):
        """Test validating a valid rule with value."""
        rule_dict = {
            "indicator": "rsi_14",
            "operator": "<",
            "value": 30,
            "action": "buy",
            "strength": 1.0,
        }
        rule = validate_rule(rule_dict, "TestStrategy")
        assert rule.indicator == "rsi_14"
        assert rule.value == 30

    def test_valid_rule_with_indicator(self):
        """Test validating a valid rule with indicator comparison."""
        rule_dict = {
            "indicator": "close",
            "operator": ">",
            "value_indicator": "bb_upper_20_2",
            "action": "sell",
            "strength": 0.8,
        }
        rule = validate_rule(rule_dict, "TestStrategy")
        assert rule.value_indicator == "bb_upper_20_2"

    def test_missing_indicator(self):
        """Test that missing indicator raises error."""
        rule_dict = {
            "operator": "<",
            "value": 30,
            "action": "buy",
        }
        with pytest.raises(InvalidRuleError) as exc_info:
            validate_rule(rule_dict, "TestStrategy")
        assert "indicator" in str(exc_info.value)

    def test_missing_operator(self):
        """Test that missing operator raises error."""
        rule_dict = {
            "indicator": "rsi_14",
            "value": 30,
            "action": "buy",
        }
        with pytest.raises(InvalidRuleError) as exc_info:
            validate_rule(rule_dict, "TestStrategy")
        assert "operator" in str(exc_info.value)

    def test_invalid_operator(self):
        """Test that invalid operator raises error."""
        rule_dict = {
            "indicator": "rsi_14",
            "operator": "~",
            "value": 30,
            "action": "buy",
        }
        with pytest.raises(InvalidRuleError) as exc_info:
            validate_rule(rule_dict, "TestStrategy")
        assert "invalid operator" in str(exc_info.value)

    def test_invalid_action(self):
        """Test that invalid action raises error."""
        rule_dict = {
            "indicator": "rsi_14",
            "operator": "<",
            "value": 30,
            "action": "hold",
        }
        with pytest.raises(InvalidRuleError) as exc_info:
            validate_rule(rule_dict, "TestStrategy")
        assert "invalid action" in str(exc_info.value)

    def test_missing_value_and_indicator(self):
        """Test that missing both value and value_indicator raises error."""
        rule_dict = {
            "indicator": "rsi_14",
            "operator": "<",
            "action": "buy",
        }
        with pytest.raises(InvalidRuleError) as exc_info:
            validate_rule(rule_dict, "TestStrategy")
        assert "value" in str(exc_info.value)

    def test_both_value_and_indicator(self):
        """Test that having both value and value_indicator raises error."""
        rule_dict = {
            "indicator": "rsi_14",
            "operator": "<",
            "value": 30,
            "value_indicator": "sma_20",
            "action": "buy",
        }
        with pytest.raises(InvalidRuleError) as exc_info:
            validate_rule(rule_dict, "TestStrategy")
        assert "cannot specify both" in str(exc_info.value)

    def test_invalid_strength_negative(self):
        """Test that negative strength raises error."""
        rule_dict = {
            "indicator": "rsi_14",
            "operator": "<",
            "value": 30,
            "action": "buy",
            "strength": -0.5,
        }
        with pytest.raises(InvalidRuleError) as exc_info:
            validate_rule(rule_dict, "TestStrategy")
        assert "strength" in str(exc_info.value)

    def test_invalid_strength_too_high(self):
        """Test that strength > 1 raises error."""
        rule_dict = {
            "indicator": "rsi_14",
            "operator": "<",
            "value": 30,
            "action": "buy",
            "strength": 1.5,
        }
        with pytest.raises(InvalidRuleError) as exc_info:
            validate_rule(rule_dict, "TestStrategy")
        assert "strength" in str(exc_info.value)

    def test_default_strength(self):
        """Test that default strength is 1.0."""
        rule_dict = {
            "indicator": "rsi_14",
            "operator": "<",
            "value": 30,
            "action": "buy",
        }
        rule = validate_rule(rule_dict, "TestStrategy")
        assert rule.strength == 1.0

    def test_all_valid_operators(self):
        """Test all valid operators work."""
        for op in VALID_OPERATORS:
            rule_dict = {
                "indicator": "rsi_14",
                "operator": op,
                "value": 30,
                "action": "buy",
            }
            rule = validate_rule(rule_dict, "TestStrategy")
            assert rule.operator == op

    def test_all_valid_actions(self):
        """Test all valid actions work."""
        for action in VALID_ACTIONS:
            rule_dict = {
                "indicator": "rsi_14",
                "operator": "<",
                "value": 30,
                "action": action,
            }
            rule = validate_rule(rule_dict, "TestStrategy")
            assert rule.action == action


class TestValidateStrategy:
    """Tests for validate_strategy function."""

    def test_valid_strategy(self):
        """Test validating a valid strategy."""
        strategy_dict = {
            "name": "TestStrategy",
            "weight": 1.0,
            "combine": "any",
            "rules": [
                {
                    "indicator": "rsi_14",
                    "operator": "<",
                    "value": 30,
                    "action": "buy",
                }
            ],
        }
        strategy = validate_strategy(strategy_dict, "test.yaml")
        assert strategy.name == "TestStrategy"
        assert strategy.weight == 1.0
        assert strategy.combine == "any"

    def test_missing_name(self):
        """Test that missing name raises error."""
        strategy_dict = {
            "weight": 1.0,
            "combine": "any",
            "rules": [],
        }
        with pytest.raises(InvalidStrategyError) as exc_info:
            validate_strategy(strategy_dict, "test.yaml")
        assert "name" in str(exc_info.value)

    def test_invalid_weight_zero(self):
        """Test that zero weight raises error."""
        strategy_dict = {
            "name": "Test",
            "weight": 0,
            "combine": "any",
            "rules": [{"indicator": "rsi_14", "operator": "<", "value": 30, "action": "buy"}],
        }
        with pytest.raises(InvalidStrategyError) as exc_info:
            validate_strategy(strategy_dict, "test.yaml")
        assert "weight" in str(exc_info.value)

    def test_invalid_weight_negative(self):
        """Test that negative weight raises error."""
        strategy_dict = {
            "name": "Test",
            "weight": -0.5,
            "combine": "any",
            "rules": [{"indicator": "rsi_14", "operator": "<", "value": 30, "action": "buy"}],
        }
        with pytest.raises(InvalidStrategyError) as exc_info:
            validate_strategy(strategy_dict, "test.yaml")
        assert "weight" in str(exc_info.value)

    def test_invalid_combine_mode(self):
        """Test that invalid combine mode raises error."""
        strategy_dict = {
            "name": "Test",
            "weight": 1.0,
            "combine": "some",
            "rules": [{"indicator": "rsi_14", "operator": "<", "value": 30, "action": "buy"}],
        }
        with pytest.raises(InvalidStrategyError) as exc_info:
            validate_strategy(strategy_dict, "test.yaml")
        assert "combine" in str(exc_info.value)

    def test_empty_rules(self):
        """Test that empty rules raises error."""
        strategy_dict = {
            "name": "Test",
            "weight": 1.0,
            "combine": "any",
            "rules": [],
        }
        with pytest.raises(InvalidStrategyError) as exc_info:
            validate_strategy(strategy_dict, "test.yaml")
        assert "rule" in str(exc_info.value)

    def test_all_valid_combine_modes(self):
        """Test all valid combine modes work."""
        for mode in VALID_COMBINE_MODES:
            strategy_dict = {
                "name": "Test",
                "weight": 1.0,
                "combine": mode,
                "rules": [
                    {"indicator": "rsi_14", "operator": "<", "value": 30, "action": "buy"}
                ],
            }
            strategy = validate_strategy(strategy_dict, "test.yaml")
            assert strategy.combine == mode


class TestLoadStrategyFile:
    """Tests for load_strategy_file function."""

    def test_load_simple_strategy(self, simple_strategy_yaml):
        """Test loading a simple strategy file."""
        strategies = load_strategy_file(str(simple_strategy_yaml))
        assert len(strategies) == 1
        assert strategies[0].name == "TestStrategy"
        assert len(strategies[0].rules) == 2

    def test_load_multi_strategy(self, multi_strategy_yaml):
        """Test loading a file with multiple strategies."""
        strategies = load_strategy_file(str(multi_strategy_yaml))
        assert len(strategies) == 2
        assert strategies[0].name == "RSI_Strategy"
        assert strategies[1].name == "MACD_Strategy"

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading a non-existent file raises error."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_strategy_file(str(temp_dir / "nonexistent.yaml"))
        assert "nonexistent.yaml" in str(exc_info.value)

    def test_load_empty_file(self, empty_yaml):
        """Test loading an empty file raises error."""
        with pytest.raises(InvalidStrategyError) as exc_info:
            load_strategy_file(str(empty_yaml))
        assert "empty" in str(exc_info.value)

    def test_load_invalid_strategy(self, invalid_strategy_yaml):
        """Test loading an invalid strategy raises error."""
        with pytest.raises(InvalidStrategyError):
            load_strategy_file(str(invalid_strategy_yaml))

    def test_load_invalid_yaml_syntax(self, temp_dir):
        """Test loading invalid YAML syntax raises error."""
        file_path = temp_dir / "bad_yaml.yaml"
        file_path.write_text("strategies:\n  - name: [invalid")
        with pytest.raises(InvalidStrategyError) as exc_info:
            load_strategy_file(str(file_path))
        assert "YAML" in str(exc_info.value)

    def test_load_missing_strategies_key(self, temp_dir):
        """Test loading YAML without strategies key raises error."""
        file_path = temp_dir / "no_strategies.yaml"
        file_path.write_text("other_key: value")
        with pytest.raises(InvalidStrategyError) as exc_info:
            load_strategy_file(str(file_path))
        assert "strategies" in str(exc_info.value)


class TestLoadStrategies:
    """Tests for load_strategies function."""

    def test_load_single_file(self, simple_strategy_yaml):
        """Test loading a single strategy file."""
        strategies = load_strategies(str(simple_strategy_yaml))
        assert len(strategies) == 1

    def test_load_multiple_files(self, simple_strategy_yaml, multi_strategy_yaml):
        """Test loading multiple strategy files."""
        strategies = load_strategies([
            str(simple_strategy_yaml),
            str(multi_strategy_yaml),
        ])
        assert len(strategies) == 3  # 1 + 2


class TestGetAllRequiredIndicators:
    """Tests for get_all_required_indicators function."""

    def test_single_strategy(self):
        """Test getting indicators from single strategy."""
        rules = [
            Rule(indicator="rsi_14", operator="<", action="buy", value=30),
            Rule(indicator="close", operator=">", action="sell", value_indicator="bb_upper_20_2"),
        ]
        strategy = Strategy(name="Test", weight=1.0, combine="any", rules=rules)
        indicators = get_all_required_indicators([strategy])
        assert "rsi_14" in indicators
        assert "close" in indicators
        assert "bb_upper_20_2" in indicators

    def test_multiple_strategies(self):
        """Test getting indicators from multiple strategies."""
        strategy1 = Strategy(
            name="S1",
            weight=1.0,
            combine="any",
            rules=[Rule(indicator="rsi_14", operator="<", action="buy", value=30)],
        )
        strategy2 = Strategy(
            name="S2",
            weight=1.0,
            combine="any",
            rules=[Rule(indicator="macd_hist_12_26_9", operator=">", action="buy", value=0)],
        )
        indicators = get_all_required_indicators([strategy1, strategy2])
        assert "rsi_14" in indicators
        assert "macd_hist_12_26_9" in indicators


class TestNormalizeWeights:
    """Tests for normalize_weights function."""

    def test_normalize_weights(self):
        """Test weight normalization."""
        strategies = [
            Strategy(name="S1", weight=0.3, combine="any", rules=[]),
            Strategy(name="S2", weight=0.7, combine="any", rules=[]),
        ]
        normalized = normalize_weights(strategies)
        total = sum(s.weight for s in normalized)
        assert abs(total - 1.0) < 0.0001

    def test_normalize_unequal_weights(self):
        """Test normalizing unequal weights."""
        strategies = [
            Strategy(name="S1", weight=1.0, combine="any", rules=[]),
            Strategy(name="S2", weight=3.0, combine="any", rules=[]),
        ]
        normalized = normalize_weights(strategies)
        assert abs(normalized[0].weight - 0.25) < 0.0001
        assert abs(normalized[1].weight - 0.75) < 0.0001

    def test_normalize_zero_weights(self):
        """Test normalizing when all weights are zero assigns equal weights."""
        strategies = [
            Strategy(name="S1", weight=0, combine="any", rules=[]),
            Strategy(name="S2", weight=0, combine="any", rules=[]),
        ]
        # Force weights to zero (bypass validation)
        for s in strategies:
            s.weight = 0

        normalized = normalize_weights(strategies)
        assert normalized[0].weight == 0.5
        assert normalized[1].weight == 0.5
