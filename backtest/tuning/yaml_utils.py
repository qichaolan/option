"""
YAML manipulation utilities.

This module provides utilities for modifying YAML structures by path.
"""

import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from backtest.tuning.exceptions import InvalidPathError


# Pattern to match array index: name[index]
ARRAY_INDEX_PATTERN = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\]$")


def parse_path_segment(segment: str) -> tuple:
    """
    Parse a single path segment.

    Args:
        segment: Path segment like "strategies" or "rules[0]"

    Returns:
        Tuple of (key_name, index) where index is None for non-array access.

    Raises:
        InvalidPathError: If segment format is invalid.
    """
    # Check for array index pattern
    match = ARRAY_INDEX_PATTERN.match(segment)
    if match:
        key_name = match.group(1)
        index = int(match.group(2))
        return key_name, index

    # Simple key access
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", segment):
        return segment, None

    raise InvalidPathError(segment, "invalid segment format")


def get_by_path(data: Dict[str, Any], path: str) -> Any:
    """
    Get a value from nested dict/list structure by path.

    Args:
        data: Root dictionary.
        path: Dot-separated path like "strategies[0].rules[1].value"

    Returns:
        Value at the specified path.

    Raises:
        InvalidPathError: If path is invalid or doesn't exist.
    """
    segments = path.split(".")
    current = data

    for segment in segments:
        key_name, index = parse_path_segment(segment)

        # Access the key
        if not isinstance(current, dict):
            raise InvalidPathError(path, f"expected dict at '{key_name}', got {type(current).__name__}")

        if key_name not in current:
            raise InvalidPathError(path, f"key '{key_name}' not found")

        current = current[key_name]

        # Access array index if specified
        if index is not None:
            if not isinstance(current, list):
                raise InvalidPathError(path, f"expected list at '{key_name}', got {type(current).__name__}")

            if index >= len(current):
                raise InvalidPathError(
                    path,
                    f"index {index} out of range for '{key_name}' (length {len(current)})",
                )

            current = current[index]

    return current


def set_by_path(data: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a value in nested dict/list structure by path.

    Args:
        data: Root dictionary (modified in place).
        path: Dot-separated path like "strategies[0].rules[1].value"
        value: Value to set.

    Raises:
        InvalidPathError: If path is invalid or parent doesn't exist.
    """
    segments = path.split(".")

    # Navigate to parent of target
    current = data
    for segment in segments[:-1]:
        key_name, index = parse_path_segment(segment)

        if not isinstance(current, dict):
            raise InvalidPathError(path, f"expected dict at '{key_name}'")

        if key_name not in current:
            raise InvalidPathError(path, f"key '{key_name}' not found")

        current = current[key_name]

        if index is not None:
            if not isinstance(current, list):
                raise InvalidPathError(path, f"expected list at '{key_name}'")

            if index >= len(current):
                raise InvalidPathError(path, f"index {index} out of range")

            current = current[index]

    # Set the final value
    final_segment = segments[-1]
    key_name, index = parse_path_segment(final_segment)

    if not isinstance(current, dict):
        raise InvalidPathError(path, f"expected dict at parent of '{key_name}'")

    if index is not None:
        # Setting value in an array element
        if key_name not in current:
            raise InvalidPathError(path, f"key '{key_name}' not found")

        if not isinstance(current[key_name], list):
            raise InvalidPathError(path, f"expected list at '{key_name}'")

        if index >= len(current[key_name]):
            raise InvalidPathError(path, f"index {index} out of range")

        current[key_name][index] = value
    else:
        # Setting value directly
        current[key_name] = value


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML file into dictionary.

    Args:
        file_path: Path to YAML file.

    Returns:
        Dictionary with YAML contents.

    Raises:
        FileNotFoundError: If file doesn't exist.
        yaml.YAMLError: If YAML is invalid.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def clone_and_modify(
    base_yaml: Dict[str, Any],
    modifications: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a deep copy of YAML structure and apply modifications.

    Args:
        base_yaml: Original YAML dictionary.
        modifications: Dict mapping paths to new values.

    Returns:
        Modified copy of YAML structure.
    """
    modified = copy.deepcopy(base_yaml)

    for path, value in modifications.items():
        set_by_path(modified, path, value)

    return modified


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary to YAML file.

    Args:
        data: Dictionary to save.
        file_path: Output file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
