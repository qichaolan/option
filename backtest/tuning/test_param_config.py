"""
Tests for parameter configuration parsing.

This module tests:
- ParameterSpec validation and value generation
- ParamConfig loading and validation
- Edge cases and error handling
"""

import tempfile
from pathlib import Path

import pytest

from backtest.tuning.exceptions import ParamConfigError, ParameterRangeError
from backtest.tuning.param_config import (
    ParamConfig,
    ParameterSpec,
    load_param_config,
    validate_param_spec,
)


class TestParameterSpec:
    """Tests for ParameterSpec dataclass."""

    def test_basic_spec_creation(self):
        """Test creating a basic parameter spec."""
        spec = ParameterSpec(
            name="rsi_oversold",
            path="strategies[0].rules[0].value",
            start=20.0,
            end=40.0,
            step=5.0,
        )

        assert spec.name == "rsi_oversold"
        assert spec.path == "strategies[0].rules[0].value"
        assert spec.start == 20.0
        assert spec.end == 40.0
        assert spec.step == 5.0

    def test_get_values_basic(self):
        """Test generating values from spec."""
        spec = ParameterSpec(
            name="test",
            path="test.path",
            start=10.0,
            end=30.0,
            step=10.0,
        )

        values = spec.get_values()
        assert values == [10.0, 20.0, 30.0]

    def test_get_values_fractional_step(self):
        """Test generating values with fractional step."""
        spec = ParameterSpec(
            name="test",
            path="test.path",
            start=0.0,
            end=1.0,
            step=0.25,
        )

        values = spec.get_values()
        assert values == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_get_values_single_value(self):
        """Test when start equals end."""
        spec = ParameterSpec(
            name="test",
            path="test.path",
            start=50.0,
            end=50.0,
            step=10.0,
        )

        values = spec.get_values()
        assert values == [50.0]

    def test_get_values_rounding(self):
        """Test floating point rounding in values."""
        spec = ParameterSpec(
            name="test",
            path="test.path",
            start=0.1,
            end=0.3,
            step=0.1,
        )

        values = spec.get_values()
        # Should be properly rounded
        assert len(values) == 3
        assert abs(values[0] - 0.1) < 0.0001
        assert abs(values[1] - 0.2) < 0.0001
        assert abs(values[2] - 0.3) < 0.0001

    def test_get_values_many_decimals(self):
        """Test value generation with many decimal places."""
        spec = ParameterSpec(
            name="test",
            path="test.path",
            start=0.001,
            end=0.005,
            step=0.001,
        )

        values = spec.get_values()
        assert len(values) == 5


class TestValidateParamSpec:
    """Tests for validate_param_spec function."""

    def test_valid_spec(self):
        """Test validation of valid spec."""
        spec = ParameterSpec(
            name="test",
            path="strategies[0].rules[0].value",
            start=20.0,
            end=40.0,
            step=5.0,
        )

        # Should not raise - dataclass validates on creation
        validate_param_spec(spec)

    def test_empty_name(self):
        """Test validation rejects empty name."""
        # Create spec object bypassing __post_init__ to test validate_param_spec
        spec = object.__new__(ParameterSpec)
        spec.name = ""
        spec.path = "test.path"
        spec.start = 20.0
        spec.end = 40.0
        spec.step = 5.0

        with pytest.raises(ParamConfigError) as exc_info:
            validate_param_spec(spec)
        assert "name" in str(exc_info.value).lower()

    def test_empty_path(self):
        """Test validation rejects empty path."""
        spec = object.__new__(ParameterSpec)
        spec.name = "test"
        spec.path = ""
        spec.start = 20.0
        spec.end = 40.0
        spec.step = 5.0

        with pytest.raises(ParamConfigError) as exc_info:
            validate_param_spec(spec)
        assert "path" in str(exc_info.value).lower()

    def test_start_greater_than_end(self):
        """Test validation rejects start > end at creation."""
        with pytest.raises(ParameterRangeError) as exc_info:
            ParameterSpec(
                name="test",
                path="test.path",
                start=50.0,
                end=20.0,
                step=5.0,
            )
        assert "start" in str(exc_info.value).lower() or "end" in str(exc_info.value).lower()

    def test_zero_step(self):
        """Test validation rejects zero step at creation."""
        with pytest.raises(ParameterRangeError) as exc_info:
            ParameterSpec(
                name="test",
                path="test.path",
                start=20.0,
                end=40.0,
                step=0.0,
            )
        assert "step" in str(exc_info.value).lower()

    def test_negative_step(self):
        """Test validation rejects negative step at creation."""
        with pytest.raises(ParameterRangeError) as exc_info:
            ParameterSpec(
                name="test",
                path="test.path",
                start=20.0,
                end=40.0,
                step=-5.0,
            )
        assert "step" in str(exc_info.value).lower()


class TestParamConfig:
    """Tests for ParamConfig dataclass."""

    def test_basic_config(self):
        """Test creating basic config."""
        specs = [
            ParameterSpec("p1", "path1", 10.0, 20.0, 5.0),
            ParameterSpec("p2", "path2", 50.0, 70.0, 10.0),
        ]
        config = ParamConfig(parameters=specs)

        assert len(config.parameters) == 2

    def test_get_search_space_size(self):
        """Test calculating search space size."""
        specs = [
            ParameterSpec("p1", "path1", 10.0, 20.0, 5.0),  # 3 values
            ParameterSpec("p2", "path2", 50.0, 70.0, 10.0),  # 3 values
        ]
        config = ParamConfig(parameters=specs)

        assert config.get_search_space_size() == 9  # 3 * 3

    def test_get_search_space_size_empty(self):
        """Test search space size with no parameters."""
        config = ParamConfig(parameters=[])
        assert config.get_search_space_size() == 1  # One empty combination

    def test_get_search_space_size_single_param(self):
        """Test search space size with single parameter."""
        specs = [
            ParameterSpec("p1", "path1", 0.0, 100.0, 25.0),  # 5 values
        ]
        config = ParamConfig(parameters=specs)

        assert config.get_search_space_size() == 5


class TestLoadParamConfig:
    """Tests for load_param_config function."""

    def test_load_valid_config(self):
        """Test loading valid configuration file."""
        config_yaml = """
parameters:
  - name: rsi_oversold
    path: strategies[0].rules[0].value
    start: 20
    end: 40
    step: 5

  - name: rsi_overbought
    path: strategies[0].rules[1].value
    start: 60
    end: 80
    step: 5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()

            config = load_param_config(f.name)

            assert len(config.parameters) == 2
            assert config.parameters[0].name == "rsi_oversold"
            assert config.parameters[0].start == 20.0
            assert config.parameters[1].name == "rsi_overbought"

            Path(f.name).unlink()

    def test_load_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_param_config("/nonexistent/path/config.yaml")

    def test_load_empty_parameters(self):
        """Test loading config with empty parameters list."""
        config_yaml = """
parameters: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()

            config = load_param_config(f.name)
            assert len(config.parameters) == 0

            Path(f.name).unlink()

    def test_load_missing_parameters_key(self):
        """Test loading config without parameters key."""
        config_yaml = """
other_key: value
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()

            with pytest.raises(ParamConfigError) as exc_info:
                load_param_config(f.name)
            assert "parameters" in str(exc_info.value).lower()

            Path(f.name).unlink()

    def test_load_missing_required_fields(self):
        """Test loading config with missing required fields."""
        config_yaml = """
parameters:
  - name: test
    path: test.path
    # missing start, end, step
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()

            with pytest.raises(ParamConfigError):
                load_param_config(f.name)

            Path(f.name).unlink()

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML file."""
        config_yaml = "invalid: yaml: content: [}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()

            with pytest.raises(ParamConfigError):
                load_param_config(f.name)

            Path(f.name).unlink()

    def test_load_config_with_floats(self):
        """Test loading config with float values."""
        config_yaml = """
parameters:
  - name: threshold
    path: strategies[0].rules[0].value
    start: 0.5
    end: 1.5
    step: 0.25
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()

            config = load_param_config(f.name)

            assert config.parameters[0].start == 0.5
            assert config.parameters[0].end == 1.5
            assert config.parameters[0].step == 0.25

            Path(f.name).unlink()

    def test_load_config_validates_ranges(self):
        """Test that loading validates parameter ranges."""
        config_yaml = """
parameters:
  - name: invalid
    path: test.path
    start: 100
    end: 50
    step: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()

            with pytest.raises(ParameterRangeError):
                load_param_config(f.name)

            Path(f.name).unlink()


class TestSearchSpaceCalculation:
    """Tests for search space size calculations."""

    def test_large_search_space(self):
        """Test calculating large search space."""
        specs = [
            ParameterSpec("p1", "path1", 0.0, 100.0, 10.0),  # 11 values
            ParameterSpec("p2", "path2", 0.0, 100.0, 10.0),  # 11 values
            ParameterSpec("p3", "path3", 0.0, 100.0, 10.0),  # 11 values
        ]
        config = ParamConfig(parameters=specs)

        # 11 * 11 * 11 = 1331
        assert config.get_search_space_size() == 1331

    def test_mixed_range_sizes(self):
        """Test with different range sizes."""
        specs = [
            ParameterSpec("p1", "path1", 0.0, 10.0, 5.0),  # 3 values
            ParameterSpec("p2", "path2", 0.0, 100.0, 25.0),  # 5 values
        ]
        config = ParamConfig(parameters=specs)

        # 3 * 5 = 15
        assert config.get_search_space_size() == 15
