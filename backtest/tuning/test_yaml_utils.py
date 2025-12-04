"""
Tests for YAML manipulation utilities.

This module tests:
- Path parsing and navigation
- Getting/setting values by path
- YAML cloning and modification
- File I/O operations
"""

import tempfile
from pathlib import Path

import pytest

from backtest.tuning.exceptions import InvalidPathError
from backtest.tuning.yaml_utils import (
    clone_and_modify,
    get_by_path,
    load_yaml,
    parse_path_segment,
    save_yaml,
    set_by_path,
)


class TestParsePathSegment:
    """Tests for parse_path_segment function."""

    def test_simple_key(self):
        """Test parsing simple key."""
        key, index = parse_path_segment("strategies")
        assert key == "strategies"
        assert index is None

    def test_array_index(self):
        """Test parsing array index."""
        key, index = parse_path_segment("rules[0]")
        assert key == "rules"
        assert index == 0

    def test_array_large_index(self):
        """Test parsing large array index."""
        key, index = parse_path_segment("items[123]")
        assert key == "items"
        assert index == 123

    def test_key_with_underscore(self):
        """Test parsing key with underscores."""
        key, index = parse_path_segment("my_key_name")
        assert key == "my_key_name"
        assert index is None

    def test_key_with_numbers(self):
        """Test parsing key with numbers."""
        key, index = parse_path_segment("rule2")
        assert key == "rule2"
        assert index is None

    def test_invalid_format_starts_with_number(self):
        """Test parsing key starting with number."""
        with pytest.raises(InvalidPathError):
            parse_path_segment("2rule")

    def test_invalid_format_special_chars(self):
        """Test parsing key with special characters."""
        with pytest.raises(InvalidPathError):
            parse_path_segment("my-key")

    def test_invalid_array_format(self):
        """Test parsing invalid array format."""
        with pytest.raises(InvalidPathError):
            parse_path_segment("rules[]")


class TestGetByPath:
    """Tests for get_by_path function."""

    def test_simple_path(self):
        """Test getting value by simple path."""
        data = {"name": "test", "value": 42}
        assert get_by_path(data, "name") == "test"
        assert get_by_path(data, "value") == 42

    def test_nested_path(self):
        """Test getting value by nested path."""
        data = {
            "level1": {
                "level2": {
                    "value": "nested"
                }
            }
        }
        assert get_by_path(data, "level1.level2.value") == "nested"

    def test_array_access(self):
        """Test getting value from array."""
        data = {
            "items": ["first", "second", "third"]
        }
        assert get_by_path(data, "items[0]") == "first"
        assert get_by_path(data, "items[1]") == "second"
        assert get_by_path(data, "items[2]") == "third"

    def test_nested_array_access(self):
        """Test getting value from nested array."""
        data = {
            "strategies": [
                {"rules": [{"value": 10}, {"value": 20}]},
                {"rules": [{"value": 30}, {"value": 40}]},
            ]
        }
        assert get_by_path(data, "strategies[0].rules[0].value") == 10
        assert get_by_path(data, "strategies[0].rules[1].value") == 20
        assert get_by_path(data, "strategies[1].rules[0].value") == 30
        assert get_by_path(data, "strategies[1].rules[1].value") == 40

    def test_key_not_found(self):
        """Test error when key not found."""
        data = {"name": "test"}
        with pytest.raises(InvalidPathError) as exc_info:
            get_by_path(data, "missing")
        assert "not found" in str(exc_info.value).lower()

    def test_index_out_of_range(self):
        """Test error when index out of range."""
        data = {"items": [1, 2, 3]}
        with pytest.raises(InvalidPathError) as exc_info:
            get_by_path(data, "items[10]")
        assert "out of range" in str(exc_info.value).lower()

    def test_expected_dict_got_list(self):
        """Test error when expecting dict but got list."""
        data = {"items": [1, 2, 3]}
        with pytest.raises(InvalidPathError) as exc_info:
            get_by_path(data, "items.name")
        assert "expected dict" in str(exc_info.value).lower()

    def test_expected_list_got_dict(self):
        """Test error when expecting list but got dict."""
        data = {"item": {"name": "test"}}
        with pytest.raises(InvalidPathError) as exc_info:
            get_by_path(data, "item[0]")
        assert "expected list" in str(exc_info.value).lower()


class TestSetByPath:
    """Tests for set_by_path function."""

    def test_simple_set(self):
        """Test setting value by simple path."""
        data = {"name": "old"}
        set_by_path(data, "name", "new")
        assert data["name"] == "new"

    def test_nested_set(self):
        """Test setting value by nested path."""
        data = {"level1": {"level2": {"value": "old"}}}
        set_by_path(data, "level1.level2.value", "new")
        assert data["level1"]["level2"]["value"] == "new"

    def test_array_set(self):
        """Test setting value in array."""
        data = {"items": [1, 2, 3]}
        set_by_path(data, "items[1]", 99)
        assert data["items"] == [1, 99, 3]

    def test_nested_array_set(self):
        """Test setting value in nested array."""
        data = {
            "strategies": [
                {"rules": [{"value": 10}]}
            ]
        }
        set_by_path(data, "strategies[0].rules[0].value", 42)
        assert data["strategies"][0]["rules"][0]["value"] == 42

    def test_set_creates_key(self):
        """Test setting creates new key if not exists."""
        data = {"existing": "value"}
        set_by_path(data, "new_key", "new_value")
        assert data["new_key"] == "new_value"

    def test_key_not_found_in_path(self):
        """Test error when intermediate key not found."""
        data = {"level1": {}}
        with pytest.raises(InvalidPathError):
            set_by_path(data, "level1.level2.value", "test")

    def test_index_out_of_range(self):
        """Test error when index out of range."""
        data = {"items": [1, 2]}
        with pytest.raises(InvalidPathError):
            set_by_path(data, "items[5]", 99)


class TestCloneAndModify:
    """Tests for clone_and_modify function."""

    def test_clone_with_single_modification(self):
        """Test cloning with single modification."""
        original = {"value": 10, "other": "unchanged"}
        modified = clone_and_modify(original, {"value": 20})

        assert original["value"] == 10  # Original unchanged
        assert modified["value"] == 20
        assert modified["other"] == "unchanged"

    def test_clone_with_multiple_modifications(self):
        """Test cloning with multiple modifications."""
        original = {
            "strategies": [
                {"rules": [{"value": 10}, {"value": 20}]}
            ]
        }
        modifications = {
            "strategies[0].rules[0].value": 15,
            "strategies[0].rules[1].value": 25,
        }
        modified = clone_and_modify(original, modifications)

        # Original unchanged
        assert original["strategies"][0]["rules"][0]["value"] == 10
        assert original["strategies"][0]["rules"][1]["value"] == 20

        # Modified
        assert modified["strategies"][0]["rules"][0]["value"] == 15
        assert modified["strategies"][0]["rules"][1]["value"] == 25

    def test_clone_is_deep_copy(self):
        """Test that clone is truly a deep copy."""
        original = {"nested": {"deep": {"value": [1, 2, 3]}}}
        modified = clone_and_modify(original, {})

        # Modify the clone
        modified["nested"]["deep"]["value"].append(4)

        # Original should be unchanged
        assert original["nested"]["deep"]["value"] == [1, 2, 3]

    def test_clone_with_empty_modifications(self):
        """Test cloning with no modifications."""
        original = {"value": 10}
        modified = clone_and_modify(original, {})

        assert modified == original
        assert modified is not original


class TestLoadYaml:
    """Tests for load_yaml function."""

    def test_load_simple_yaml(self):
        """Test loading simple YAML file."""
        yaml_content = """
name: test
value: 42
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            data = load_yaml(f.name)
            assert data["name"] == "test"
            assert data["value"] == 42

            Path(f.name).unlink()

    def test_load_complex_yaml(self):
        """Test loading complex YAML file."""
        yaml_content = """
strategies:
  - name: "Strategy1"
    rules:
      - indicator: "rsi_14"
        operator: "<"
        value: 30
        action: "buy"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            data = load_yaml(f.name)
            assert data["strategies"][0]["name"] == "Strategy1"
            assert data["strategies"][0]["rules"][0]["value"] == 30

            Path(f.name).unlink()

    def test_load_file_not_found(self):
        """Test error when file not found."""
        with pytest.raises(FileNotFoundError):
            load_yaml("/nonexistent/path/file.yaml")


class TestSaveYaml:
    """Tests for save_yaml function."""

    def test_save_simple_yaml(self):
        """Test saving simple YAML data."""
        data = {"name": "test", "value": 42}

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "output.yaml"
            save_yaml(data, str(file_path))

            assert file_path.exists()
            loaded = load_yaml(str(file_path))
            assert loaded == data

    def test_save_creates_parent_dirs(self):
        """Test that save creates parent directories."""
        data = {"name": "test"}

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "nested" / "dir" / "output.yaml"
            save_yaml(data, str(file_path))

            assert file_path.exists()

    def test_save_preserves_order(self):
        """Test that save preserves key order."""
        data = {"z_key": 1, "a_key": 2, "m_key": 3}

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "output.yaml"
            save_yaml(data, str(file_path))

            # Read raw content
            content = file_path.read_text()
            z_pos = content.find("z_key")
            a_pos = content.find("a_key")
            m_pos = content.find("m_key")

            # Original order should be preserved (sort_keys=False)
            assert z_pos < a_pos < m_pos


class TestIntegration:
    """Integration tests for YAML utilities."""

    def test_full_workflow(self):
        """Test complete workflow: load, modify, save, reload."""
        yaml_content = """
strategies:
  - name: "RSI_Strategy"
    rules:
      - indicator: "rsi_14"
        operator: "<"
        value: 30
        action: "buy"
      - indicator: "rsi_14"
        operator: ">"
        value: 70
        action: "sell"
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save original
            original_path = Path(temp_dir) / "original.yaml"
            original_path.write_text(yaml_content)

            # Load and modify
            data = load_yaml(str(original_path))
            modifications = {
                "strategies[0].rules[0].value": 25,
                "strategies[0].rules[1].value": 75,
            }
            modified = clone_and_modify(data, modifications)

            # Save modified
            modified_path = Path(temp_dir) / "modified.yaml"
            save_yaml(modified, str(modified_path))

            # Reload and verify
            reloaded = load_yaml(str(modified_path))
            assert reloaded["strategies"][0]["rules"][0]["value"] == 25
            assert reloaded["strategies"][0]["rules"][1]["value"] == 75

    def test_strategy_yaml_modification(self):
        """Test realistic strategy YAML modification."""
        data = {
            "strategies": [
                {
                    "name": "MediumTerm_Trend",
                    "weight": 1.0,
                    "rules": [
                        {
                            "indicator": "sma_20",
                            "operator": ">",
                            "value_indicator": "sma_50",
                            "action": "buy",
                        },
                        {
                            "indicator": "rsi_14",
                            "operator": "<",
                            "value": 30,
                            "action": "buy",
                        },
                    ],
                }
            ]
        }

        # Modify RSI threshold
        modifications = {"strategies[0].rules[1].value": 35}
        modified = clone_and_modify(data, modifications)

        assert modified["strategies"][0]["rules"][1]["value"] == 35
        assert data["strategies"][0]["rules"][1]["value"] == 30  # Original unchanged
