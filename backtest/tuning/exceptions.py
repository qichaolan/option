"""
Custom exceptions for the parameter tuning module.
"""

from typing import List, Optional


class TuningError(Exception):
    """Base exception for all tuning-related errors."""

    pass


class ParamConfigError(TuningError):
    """Exception raised when parameter config is invalid."""

    def __init__(self, file_path: str, details: str = "") -> None:
        self.file_path = file_path
        message = f"Invalid parameter config: {file_path}"
        if details:
            message += f": {details}"
        super().__init__(message)


class InvalidPathError(TuningError):
    """Exception raised when a YAML path is invalid."""

    def __init__(self, path: str, reason: str = "") -> None:
        self.path = path
        message = f"Invalid YAML path: {path}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class ParameterRangeError(TuningError):
    """Exception raised when parameter range is invalid."""

    def __init__(self, param_name: str, reason: str = "") -> None:
        self.param_name = param_name
        message = f"Invalid parameter range for '{param_name}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)
