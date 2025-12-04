"""
Strategy YAML loader and validation module.

This module handles loading strategy configurations from YAML files and
validating their structure.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from backtest.engine.exceptions import (
    FileNotFoundError,
    InvalidRuleError,
    InvalidStrategyError,
)


# Valid operators for rule comparisons
VALID_OPERATORS = {"<", "<=", ">", ">=", "==", "!="}

# Valid actions for rules
VALID_ACTIONS = {"buy", "sell"}

# Valid combine modes for strategies
VALID_COMBINE_MODES = {"all", "any"}


class Rule:
    """
    Represents a single trading rule.

    Attributes:
        indicator: Name of the indicator column to evaluate.
        operator: Comparison operator (<, <=, >, >=, ==, !=).
        value: Static threshold value (mutually exclusive with value_indicator).
        value_indicator: Indicator column to compare against.
        action: Action when rule triggers ("buy" or "sell").
        strength: Rule strength weight (0.0 to 1.0).
    """

    def __init__(
        self,
        indicator: str,
        operator: str,
        action: str,
        strength: float = 1.0,
        value: Optional[float] = None,
        value_indicator: Optional[str] = None,
    ) -> None:
        self.indicator = indicator.lower() if indicator else indicator
        self.operator = operator
        self.action = action.lower() if action else action
        self.strength = strength
        self.value = value
        self.value_indicator = (
            value_indicator.lower() if value_indicator else value_indicator
        )

    def __repr__(self) -> str:
        if self.value is not None:
            return f"Rule({self.indicator} {self.operator} {self.value}, {self.action}, strength={self.strength})"
        return f"Rule({self.indicator} {self.operator} {self.value_indicator}, {self.action}, strength={self.strength})"


class Strategy:
    """
    Represents a trading strategy with multiple rules.

    Attributes:
        name: Strategy name.
        weight: Strategy weight for multi-strategy aggregation.
        combine: Rule combination mode ("all" or "any").
        rules: List of Rule objects.
    """

    def __init__(
        self,
        name: str,
        weight: float,
        combine: str,
        rules: List[Rule],
    ) -> None:
        self.name = name
        self.weight = weight
        self.combine = combine.lower() if combine else combine
        self.rules = rules

    def __repr__(self) -> str:
        return f"Strategy({self.name}, weight={self.weight}, combine={self.combine}, rules={len(self.rules)})"

    def get_required_indicators(self) -> List[str]:
        """Get list of all indicators required by this strategy's rules."""
        indicators = set()
        for rule in self.rules:
            indicators.add(rule.indicator)
            if rule.value_indicator:
                indicators.add(rule.value_indicator)
        return list(indicators)


def validate_rule(rule_dict: Dict[str, Any], strategy_name: str) -> Rule:
    """
    Validate and create a Rule from a dictionary.

    Args:
        rule_dict: Dictionary containing rule configuration.
        strategy_name: Name of the parent strategy (for error messages).

    Returns:
        Validated Rule object.

    Raises:
        InvalidRuleError: If rule is invalid.
    """
    # Check required fields
    required_fields = {"indicator", "operator", "action"}
    missing = required_fields - set(rule_dict.keys())
    if missing:
        raise InvalidRuleError(
            f"in strategy '{strategy_name}'",
            f"missing required fields: {', '.join(missing)}",
        )

    indicator = rule_dict.get("indicator")
    operator = rule_dict.get("operator")
    action = rule_dict.get("action")
    strength = rule_dict.get("strength", 1.0)
    value = rule_dict.get("value")
    value_indicator = rule_dict.get("value_indicator")

    # Validate operator
    if operator not in VALID_OPERATORS:
        raise InvalidRuleError(
            f"in strategy '{strategy_name}'",
            f"invalid operator '{operator}', must be one of: {', '.join(VALID_OPERATORS)}",
        )

    # Validate action
    if action.lower() not in VALID_ACTIONS:
        raise InvalidRuleError(
            f"in strategy '{strategy_name}'",
            f"invalid action '{action}', must be one of: {', '.join(VALID_ACTIONS)}",
        )

    # Validate value or value_indicator (must have exactly one)
    if value is None and value_indicator is None:
        raise InvalidRuleError(
            f"in strategy '{strategy_name}'",
            "must specify either 'value' or 'value_indicator'",
        )
    if value is not None and value_indicator is not None:
        raise InvalidRuleError(
            f"in strategy '{strategy_name}'",
            "cannot specify both 'value' and 'value_indicator'",
        )

    # Validate strength
    if not isinstance(strength, (int, float)) or strength < 0 or strength > 1:
        raise InvalidRuleError(
            f"in strategy '{strategy_name}'",
            f"strength must be a number between 0 and 1, got {strength}",
        )

    return Rule(
        indicator=indicator,
        operator=operator,
        action=action,
        strength=float(strength),
        value=float(value) if value is not None else None,
        value_indicator=value_indicator,
    )


def validate_strategy(strategy_dict: Dict[str, Any], file_path: str) -> Strategy:
    """
    Validate and create a Strategy from a dictionary.

    Args:
        strategy_dict: Dictionary containing strategy configuration.
        file_path: Source file path (for error messages).

    Returns:
        Validated Strategy object.

    Raises:
        InvalidStrategyError: If strategy is invalid.
    """
    # Check required fields
    required_fields = {"name", "weight", "combine", "rules"}
    missing = required_fields - set(strategy_dict.keys())
    if missing:
        raise InvalidStrategyError(
            file_path, f"missing required fields: {', '.join(missing)}"
        )

    name = strategy_dict.get("name")
    weight = strategy_dict.get("weight")
    combine = strategy_dict.get("combine")
    rules_list = strategy_dict.get("rules", [])

    # Validate name
    if not name or not isinstance(name, str):
        raise InvalidStrategyError(file_path, "strategy name must be a non-empty string")

    # Validate weight
    if not isinstance(weight, (int, float)) or weight <= 0:
        raise InvalidStrategyError(
            file_path, f"weight must be a positive number, got {weight}"
        )

    # Validate combine mode
    if combine.lower() not in VALID_COMBINE_MODES:
        raise InvalidStrategyError(
            file_path,
            f"invalid combine mode '{combine}', must be one of: {', '.join(VALID_COMBINE_MODES)}",
        )

    # Validate rules
    if not rules_list or not isinstance(rules_list, list):
        raise InvalidStrategyError(
            file_path, "strategy must have at least one rule"
        )

    rules = [validate_rule(r, name) for r in rules_list]

    return Strategy(
        name=name,
        weight=float(weight),
        combine=combine,
        rules=rules,
    )


def load_strategy_file(file_path: str) -> List[Strategy]:
    """
    Load strategies from a YAML file.

    Args:
        file_path: Path to YAML strategy file.

    Returns:
        List of Strategy objects.

    Raises:
        FileNotFoundError: If file does not exist.
        InvalidStrategyError: If YAML is invalid or strategy structure is wrong.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    try:
        with open(path, "r") as f:
            content = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise InvalidStrategyError(file_path, f"YAML parsing error: {e}")

    if not content:
        raise InvalidStrategyError(file_path, "file is empty")

    if "strategies" not in content:
        raise InvalidStrategyError(file_path, "missing 'strategies' key")

    strategies_list = content["strategies"]
    if not isinstance(strategies_list, list):
        raise InvalidStrategyError(file_path, "'strategies' must be a list")

    if not strategies_list:
        raise InvalidStrategyError(file_path, "no strategies defined")

    return [validate_strategy(s, file_path) for s in strategies_list]


def load_strategies(file_paths: Union[str, List[str]]) -> List[Strategy]:
    """
    Load strategies from one or more YAML files.

    Args:
        file_paths: Single file path or list of file paths.

    Returns:
        List of all Strategy objects from all files.

    Raises:
        FileNotFoundError: If any file does not exist.
        InvalidStrategyError: If any YAML is invalid.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_strategies = []
    for file_path in file_paths:
        strategies = load_strategy_file(file_path)
        all_strategies.extend(strategies)

    return all_strategies


def get_all_required_indicators(strategies: List[Strategy]) -> List[str]:
    """
    Get all unique indicators required by a list of strategies.

    Args:
        strategies: List of Strategy objects.

    Returns:
        Sorted list of unique indicator names.
    """
    indicators = set()
    for strategy in strategies:
        indicators.update(strategy.get_required_indicators())
    return sorted(indicators)


def normalize_weights(strategies: List[Strategy]) -> List[Strategy]:
    """
    Normalize strategy weights to sum to 1.0.

    Creates new Strategy objects to avoid modifying the originals.

    Args:
        strategies: List of Strategy objects.

    Returns:
        New list of Strategy objects with normalized weights.
    """
    total_weight = sum(s.weight for s in strategies)

    normalized = []
    for s in strategies:
        if total_weight == 0:
            # Avoid division by zero, assign equal weights
            new_weight = 1.0 / len(strategies)
        else:
            new_weight = s.weight / total_weight

        # Create a new Strategy with the normalized weight
        normalized.append(Strategy(
            name=s.name,
            weight=new_weight,
            combine=s.combine,
            rules=s.rules,  # Rules are not modified, safe to share reference
        ))

    return normalized
