"""
Continuous scoring engine for backtesting.

This module provides continuous score generation from trading rules,
producing signal_raw ∈ [-1, +1] and signal_0_1 ∈ [0, 1] for each day.

The scoring engine does NOT execute trades - it only computes scores.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backtest.engine.activations import compute_rule_activation
from backtest.engine.constants import Action, CombineMode
from backtest.engine.rule_engine import (
    get_column_case_insensitive,
    validate_indicators_present,
)
from backtest.engine.strategy_loader import Rule, Strategy


@dataclass
class RuleScore:
    """Score result for a single rule."""

    rule_name: str
    activation: pd.Series
    score: pd.Series  # action_sign * strength * activation


@dataclass
class StrategyScore:
    """Score result for a single strategy."""

    strategy_name: str
    rule_scores: List[RuleScore]
    raw_score: pd.Series  # Sum of rule scores, clipped to [-1, +1]
    weight: float


@dataclass
class ScoringResult:
    """Complete scoring engine result."""

    # Per-strategy scores
    strategy_scores: List[StrategyScore]

    # Global aggregated scores
    signal_raw: pd.Series  # ∈ [-1, +1], strong sell → strong buy
    signal_0_1: pd.Series  # ∈ [0, 1], UI-friendly format

    # DataFrame with all scores
    scores_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def get_strategy_score(self, strategy_name: str) -> Optional[pd.Series]:
        """Get raw score for a specific strategy."""
        for ss in self.strategy_scores:
            if ss.strategy_name == strategy_name:
                return ss.raw_score
        return None


def evaluate_rule_score(
    df: pd.DataFrame,
    rule: Rule,
    rule_index: int = 0,
) -> RuleScore:
    """
    Evaluate a rule and compute continuous activation and score.

    Formula:
        action_sign = +1 for "buy", -1 for "sell"
        activation = compute_activation(rule, data) ∈ [0, 1]
        rule_score = action_sign * rule.strength * activation

    Args:
        df: DataFrame with indicator columns.
        rule: Rule to evaluate.
        rule_index: Index of the rule (for naming).

    Returns:
        RuleScore with activation and score series.
    """
    indicator_values = get_column_case_insensitive(df, rule.indicator)

    # Get comparison value
    if rule.value is not None:
        comparison_values = rule.value
        comparison_name = None
    else:
        comparison_values = get_column_case_insensitive(df, rule.value_indicator)
        comparison_name = rule.value_indicator

    # Compute continuous activation
    activation = compute_rule_activation(
        indicator_values=indicator_values,
        comparison_values=comparison_values,
        operator=rule.operator,
        indicator_name=rule.indicator,
        comparison_name=comparison_name,
        df=df,
    )

    # Compute rule score
    action_sign = 1.0 if rule.action == Action.BUY else -1.0
    score = action_sign * rule.strength * activation

    # Create rule name
    if rule.value is not None:
        rule_name = f"rule_{rule_index}_{rule.indicator}_{rule.operator}_{rule.value}"
    else:
        rule_name = f"rule_{rule_index}_{rule.indicator}_{rule.operator}_{rule.value_indicator}"

    return RuleScore(
        rule_name=rule_name,
        activation=activation,
        score=score,
    )


def evaluate_strategy_score(
    df: pd.DataFrame,
    strategy: Strategy,
) -> StrategyScore:
    """
    Evaluate all rules in a strategy and compute strategy-level score.

    For combine="any":
        strategy_score = sum(rule_scores), clipped to [-1, +1]

    For combine="all":
        strategy_activation = min(all rule activations for same action type)
        strategy_score = sum(rule_scores) * strategy_activation

    Args:
        df: DataFrame with indicator columns.
        strategy: Strategy to evaluate.

    Returns:
        StrategyScore with all rule scores and aggregated strategy score.
    """
    if not strategy.rules:
        return StrategyScore(
            strategy_name=strategy.name,
            rule_scores=[],
            raw_score=pd.Series(0.0, index=df.index),
            weight=strategy.weight,
        )

    # Evaluate all rules
    rule_scores = []
    for i, rule in enumerate(strategy.rules):
        rs = evaluate_rule_score(df, rule, rule_index=i)
        rule_scores.append(rs)

    # Aggregate based on combine mode
    if strategy.combine == CombineMode.ALL:
        # Separate buy and sell rules
        buy_activations = []
        sell_activations = []
        buy_scores = []
        sell_scores = []

        for i, rule in enumerate(strategy.rules):
            rs = rule_scores[i]
            if rule.action == Action.BUY:
                buy_activations.append(rs.activation)
                buy_scores.append(rs.score)
            else:
                sell_activations.append(rs.activation)
                sell_scores.append(rs.score)

        # For "all" mode: use minimum activation as gate
        raw_score = pd.Series(0.0, index=df.index)

        if buy_scores:
            buy_score_sum = pd.concat(buy_scores, axis=1).sum(axis=1)
            buy_activation_min = pd.concat(buy_activations, axis=1).min(axis=1)
            raw_score += buy_score_sum * buy_activation_min

        if sell_scores:
            sell_score_sum = pd.concat(sell_scores, axis=1).sum(axis=1)
            sell_activation_min = pd.concat(sell_activations, axis=1).min(axis=1)
            raw_score += sell_score_sum * sell_activation_min

    else:  # combine="any"
        # Simply sum all rule scores
        all_scores = [rs.score for rs in rule_scores]
        raw_score = pd.concat(all_scores, axis=1).sum(axis=1)

    # Clip to [-1, +1]
    raw_score = raw_score.clip(lower=-1.0, upper=1.0)

    return StrategyScore(
        strategy_name=strategy.name,
        rule_scores=rule_scores,
        raw_score=raw_score,
        weight=strategy.weight,
    )


def _zscore_normalize(series: pd.Series) -> pd.Series:
    """
    Apply Z-score normalization and map to [0, 1] using CDF.

    This spreads the distribution so that:
    - Bearish (z < -0.5): maps to ~0.0-0.3
    - Neutral (z ~ 0):    maps to ~0.4-0.6
    - Bullish (z > 0.5):  maps to ~0.7-1.0

    Uses the standard normal CDF (Phi function) to map z-scores to probabilities.
    """
    from scipy import stats

    mean = series.mean()
    std = series.std()

    if std < 1e-10:
        # No variance - return neutral
        return pd.Series(0.5, index=series.index)

    # Compute z-scores
    z_scores = (series - mean) / std

    # Map z-scores to [0, 1] using standard normal CDF
    # This naturally spreads the distribution:
    # z = -2 -> ~0.02, z = -1 -> ~0.16, z = 0 -> 0.5, z = 1 -> ~0.84, z = 2 -> ~0.98
    normalized = pd.Series(stats.norm.cdf(z_scores), index=series.index)

    return normalized


def aggregate_global_score(
    strategy_scores: List[StrategyScore],
    index: pd.Index,
    normalization: str = "none",
) -> Tuple[pd.Series, pd.Series]:
    """
    Aggregate strategy scores into global signal scores.

    Formula:
        global_raw_score = Σ (strategy_weight * strategy_score)
        global_raw_score = clip(global_raw_score, -1, +1)
        global_score_0_1 = transform(global_raw_score) based on normalization

    Args:
        strategy_scores: List of StrategyScore objects.
        index: DataFrame index for output series.
        normalization: Normalization method for signal_0_1:
            - "none": Simple linear transform (raw + 1) / 2
            - "minmax": Rescale to use full [0,1] range
            - "zscore": Z-score normalization with CDF mapping, spreads distribution:
                * Bearish: 0.0-0.3
                * Neutral: 0.4-0.6
                * Bullish: 0.7-1.0

    Returns:
        Tuple of (signal_raw, signal_0_1) series.
    """
    if not strategy_scores:
        signal_raw = pd.Series(0.0, index=index)
        signal_0_1 = pd.Series(0.5, index=index)
        return signal_raw, signal_0_1

    # Weight normalization (should already be done, but ensure)
    total_weight = sum(ss.weight for ss in strategy_scores)
    if total_weight == 0:
        total_weight = 1.0

    # Weighted sum
    signal_raw = pd.Series(0.0, index=index)
    for ss in strategy_scores:
        normalized_weight = ss.weight / total_weight
        signal_raw += normalized_weight * ss.raw_score

    # Clip to [-1, +1]
    signal_raw = signal_raw.clip(lower=-1.0, upper=1.0)

    # Apply normalization
    if normalization == "zscore":
        # Z-score normalization with CDF mapping
        signal_0_1 = _zscore_normalize(signal_raw)
    elif normalization == "minmax":
        # Simple min-max rescaling
        signal_0_1 = (signal_raw + 1.0) / 2.0
        min_val = signal_0_1.min()
        max_val = signal_0_1.max()
        if max_val > min_val:
            signal_0_1 = (signal_0_1 - min_val) / (max_val - min_val)
    else:
        # Default: simple linear transform
        signal_0_1 = (signal_raw + 1.0) / 2.0

    return signal_raw, signal_0_1


def build_scores_dataframe(
    df: pd.DataFrame,
    strategy_scores: List[StrategyScore],
    signal_raw: pd.Series,
    signal_0_1: pd.Series,
    include_rule_scores: bool = False,
) -> pd.DataFrame:
    """
    Build a merged DataFrame with indicators and all scores.

    Args:
        df: Original DataFrame with indicators.
        strategy_scores: List of StrategyScore objects.
        signal_raw: Global raw score series.
        signal_0_1: Global 0-1 score series.
        include_rule_scores: Whether to include individual rule scores.

    Returns:
        DataFrame with original columns plus score columns.
    """
    result = df.copy()

    # Add strategy scores
    for ss in strategy_scores:
        col_name = f"score_{ss.strategy_name.replace(' ', '_')}"
        result[col_name] = ss.raw_score

        if include_rule_scores:
            for rs in ss.rule_scores:
                rule_col = f"{col_name}_{rs.rule_name}"
                result[rule_col] = rs.score
                activation_col = f"{col_name}_{rs.rule_name}_activation"
                result[activation_col] = rs.activation

    # Add global scores
    result["signal_raw"] = signal_raw
    result["signal_0_1"] = signal_0_1

    return result


def run_scoring_engine(
    df: pd.DataFrame,
    strategies: List[Strategy],
    include_rule_scores: bool = False,
    normalization: str = "none",
) -> ScoringResult:
    """
    Run the continuous scoring engine.

    This is the main entry point for score mode. It evaluates all strategies
    and computes continuous scores without executing trades.

    Args:
        df: DataFrame with indicator columns.
        strategies: List of Strategy objects (should have normalized weights).
        include_rule_scores: Whether to include individual rule scores in output.
        normalization: Normalization method for signal_0_1:
            - "none": Simple linear transform (default)
            - "minmax": Rescale to use full [0,1] range
            - "zscore": Z-score with CDF mapping, spreads distribution:
                * Bearish: 0.0-0.3
                * Neutral: 0.4-0.6
                * Bullish: 0.7-1.0

    Returns:
        ScoringResult with all scores and merged DataFrame.

    Raises:
        ValueError: If DataFrame is empty.
    """
    # Validate DataFrame is not empty
    if df.empty:
        raise ValueError("Cannot run scoring engine on empty DataFrame")

    # Validate indicators
    validate_indicators_present(df, strategies)

    # Evaluate each strategy
    strategy_scores = []
    for strategy in strategies:
        ss = evaluate_strategy_score(df, strategy)
        strategy_scores.append(ss)

    # Aggregate global score
    signal_raw, signal_0_1 = aggregate_global_score(
        strategy_scores, df.index, normalization=normalization
    )

    # Build output DataFrame
    scores_df = build_scores_dataframe(
        df=df,
        strategy_scores=strategy_scores,
        signal_raw=signal_raw,
        signal_0_1=signal_0_1,
        include_rule_scores=include_rule_scores,
    )

    return ScoringResult(
        strategy_scores=strategy_scores,
        signal_raw=signal_raw,
        signal_0_1=signal_0_1,
        scores_df=scores_df,
    )
