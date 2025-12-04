"""
Custom exceptions for the stock indicator module.

This module defines all custom exceptions used throughout the backtest package.
"""

from typing import List


class IndicatorError(Exception):
    """Base exception for all indicator-related errors."""

    pass


class LoaderError(IndicatorError):
    """Exception raised when loading CSV data fails."""

    pass


class ValidationError(IndicatorError):
    """Exception raised when data validation fails."""

    pass


class MissingColumnError(ValidationError):
    """Exception raised when required columns are missing."""

    def __init__(self, missing_columns: List[str]) -> None:
        self.missing_columns = missing_columns
        message = f"Missing required columns: {', '.join(missing_columns)}"
        super().__init__(message)


class InvalidDataTypeError(ValidationError):
    """Exception raised when data types cannot be converted."""

    def __init__(self, column: str, details: str = "") -> None:
        self.column = column
        message = f"Invalid data type in column '{column}'"
        if details:
            message += f": {details}"
        super().__init__(message)


class DuplicateDateError(ValidationError):
    """Exception raised when duplicate dates are found."""

    def __init__(self, duplicate_dates: list) -> None:
        self.duplicate_dates = duplicate_dates
        message = f"Duplicate dates found: {duplicate_dates[:5]}"
        if len(duplicate_dates) > 5:
            message += f" (and {len(duplicate_dates) - 5} more)"
        super().__init__(message)


class NonMonotonicDateError(ValidationError):
    """Exception raised when dates are not in chronological order."""

    def __init__(self, details: str = "") -> None:
        message = "Dates are not in monotonically increasing order"
        if details:
            message += f": {details}"
        super().__init__(message)


class EmptyDataError(ValidationError):
    """Exception raised when the data is empty."""

    def __init__(self) -> None:
        super().__init__("Data is empty (no rows)")


class FileNotFoundError(LoaderError):
    """Exception raised when the CSV file is not found."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        super().__init__(f"File not found: {file_path}")


class EmptyFileError(LoaderError):
    """Exception raised when the CSV file is empty."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        super().__init__(f"File is empty: {file_path}")
