"""
Parameter configuration parsing and validation.

This module handles loading and validating parameter search configurations.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from backtest.tuning.exceptions import ParamConfigError, ParameterRangeError


@dataclass
class ParameterSpec:
    """Specification for a single parameter to tune."""

    name: str
    path: str
    start: float
    end: float
    step: float

    def __post_init__(self) -> None:
        """Validate parameter specification."""
        if self.step <= 0:
            raise ParameterRangeError(self.name, "step must be positive")
        if self.start > self.end:
            raise ParameterRangeError(
                self.name, f"start ({self.start}) must be <= end ({self.end})"
            )

    def get_values(self) -> List[float]:
        """
        Generate list of values to search.

        Returns:
            List of parameter values from start to end with given step.
        """
        # Use numpy arange for proper float handling, then round to avoid
        # floating point precision issues
        values = np.arange(self.start, self.end + self.step / 2, self.step)
        # Round to reasonable precision
        decimals = max(0, -int(np.floor(np.log10(abs(self.step)))) + 1) if self.step != 0 else 2
        values = np.round(values, decimals)
        return values.tolist()

    def __repr__(self) -> str:
        return f"ParameterSpec({self.name}: {self.start} to {self.end} step {self.step})"


@dataclass
class ParamConfig:
    """Complete parameter configuration for grid search."""

    parameters: List[ParameterSpec]

    def get_search_space_size(self) -> int:
        """Calculate total number of combinations to search."""
        size = 1
        for param in self.parameters:
            size *= len(param.get_values())
        return size

    def __repr__(self) -> str:
        return f"ParamConfig({len(self.parameters)} params, {self.get_search_space_size()} combinations)"


def validate_param_spec(spec: ParameterSpec) -> None:
    """
    Validate a ParameterSpec object.

    Args:
        spec: ParameterSpec to validate.

    Raises:
        ParamConfigError: If name or path is invalid.
        ParameterRangeError: If range values are invalid.
    """
    if not spec.name or not isinstance(spec.name, str):
        raise ParamConfigError(spec.name or "(empty)", "name must be a non-empty string")

    if not spec.path or not isinstance(spec.path, str):
        raise ParamConfigError(spec.name, "path must be a non-empty string")

    if spec.step <= 0:
        raise ParameterRangeError(spec.name, "step must be positive")

    if spec.start > spec.end:
        raise ParameterRangeError(
            spec.name, f"start ({spec.start}) must be <= end ({spec.end})"
        )


def validate_param_dict(param_dict: Dict[str, Any], index: int) -> ParameterSpec:
    """
    Validate and create a ParameterSpec from a dictionary.

    Args:
        param_dict: Dictionary containing parameter configuration.
        index: Index of parameter in list (for error messages).

    Returns:
        Validated ParameterSpec.

    Raises:
        ParamConfigError: If parameter is invalid.
    """
    required_fields = {"name", "path", "start", "end", "step"}
    missing = required_fields - set(param_dict.keys())
    if missing:
        raise ParamConfigError(
            f"param[{index}]",
            f"missing required fields: {', '.join(missing)}",
        )

    name = param_dict["name"]
    path = param_dict["path"]
    start = param_dict["start"]
    end = param_dict["end"]
    step = param_dict["step"]

    # Validate types
    if not isinstance(name, str) or not name:
        raise ParamConfigError(f"param[{index}]", "name must be a non-empty string")

    if not isinstance(path, str) or not path:
        raise ParamConfigError(f"param[{index}]", "path must be a non-empty string")

    # Validate numeric types
    for field_name, value in [("start", start), ("end", end), ("step", step)]:
        if not isinstance(value, (int, float)):
            raise ParamConfigError(
                f"param[{index}]",
                f"{field_name} must be a number, got {type(value).__name__}",
            )

    return ParameterSpec(
        name=name,
        path=path,
        start=float(start),
        end=float(end),
        step=float(step),
    )


def load_param_config(file_path: str) -> ParamConfig:
    """
    Load parameter configuration from YAML file.

    Args:
        file_path: Path to YAML configuration file.

    Returns:
        ParamConfig with all parameters.

    Raises:
        ParamConfigError: If file is invalid.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Parameter config file not found: {file_path}")

    try:
        with open(path, "r") as f:
            content = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ParamConfigError(file_path, f"YAML parsing error: {e}")

    if not content:
        raise ParamConfigError(file_path, "file is empty")

    # Accept both "params" and "parameters" keys for flexibility
    if "params" in content:
        params_list = content["params"]
    elif "parameters" in content:
        params_list = content["parameters"]
    else:
        raise ParamConfigError(file_path, "missing 'parameters' key")
    if not isinstance(params_list, list):
        raise ParamConfigError(file_path, "'params' must be a list")

    # Empty list is allowed
    if params_list is None:
        params_list = []

    parameters = []
    for i, param_dict in enumerate(params_list):
        param = validate_param_dict(param_dict, i)
        parameters.append(param)

    return ParamConfig(parameters=parameters)
