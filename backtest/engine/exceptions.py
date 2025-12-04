"""
Custom exceptions for the backtest engine module.

This module defines all custom exceptions used throughout the backtest engine.
"""

from typing import Any, List, Optional


class BacktestError(Exception):
    """Base exception for all backtest-related errors."""

    pass


class StrategyError(BacktestError):
    """Exception raised when strategy loading or validation fails."""

    pass


class InvalidStrategyError(StrategyError):
    """Exception raised when strategy YAML is invalid."""

    def __init__(self, file_path: str, details: str = "") -> None:
        self.file_path = file_path
        message = f"Invalid strategy file: {file_path}"
        if details:
            message += f": {details}"
        super().__init__(message)


class MissingIndicatorError(BacktestError):
    """Exception raised when required indicators are missing from data."""

    def __init__(self, missing_indicators: List[str]) -> None:
        self.missing_indicators = missing_indicators
        message = f"Missing indicators in data: {', '.join(missing_indicators)}"
        super().__init__(message)


class InvalidRuleError(StrategyError):
    """Exception raised when a rule definition is invalid."""

    def __init__(self, rule_details: str, reason: str = "") -> None:
        self.rule_details = rule_details
        message = f"Invalid rule: {rule_details}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class DataError(BacktestError):
    """Exception raised when data loading or validation fails."""

    pass


class InsufficientDataError(DataError):
    """Exception raised when there's not enough data for backtesting."""

    def __init__(self, rows: int, required: int = 1) -> None:
        self.rows = rows
        self.required = required
        super().__init__(
            f"Insufficient data: got {rows} rows, need at least {required}"
        )


class FileNotFoundError(BacktestError):
    """Exception raised when a required file is not found."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        super().__init__(f"File not found: {file_path}")


class PortfolioError(BacktestError):
    """Exception raised when portfolio simulation fails."""

    pass


class InvalidParameterError(BacktestError):
    """Exception raised when an invalid parameter is provided."""

    def __init__(self, param_name: str, value: Any, reason: str = "") -> None:
        self.param_name = param_name
        self.value = value
        message = f"Invalid parameter '{param_name}': {value}"
        if reason:
            message += f": {reason}"
        super().__init__(message)
