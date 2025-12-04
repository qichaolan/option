"""
Pytest fixtures for backtest engine tests.
"""

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_indicator_data() -> pd.DataFrame:
    """Create sample indicator data for testing."""
    np.random.seed(42)
    n_days = 100

    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")
    close = 100 + np.cumsum(np.random.randn(n_days) * 2)

    df = pd.DataFrame({
        "Date": dates,
        "Open": close - np.random.rand(n_days),
        "High": close + np.random.rand(n_days) * 2,
        "Low": close - np.random.rand(n_days) * 2,
        "Close": close,
        "Volume": np.random.randint(1000000, 10000000, n_days),
        "rsi_14": np.random.uniform(20, 80, n_days),
        "macd_12_26_9": np.random.randn(n_days) * 2,
        "macd_signal_12_26_9": np.random.randn(n_days) * 2,
        "macd_hist_12_26_9": np.random.randn(n_days),
        "mfi_14": np.random.uniform(20, 80, n_days),
        "bb_upper_20_2": close + 5,
        "bb_lower_20_2": close - 5,
        "vol_sma_20": np.random.randint(5000000, 8000000, n_days),
        "sma_20": close - 1,
        "ema_9": close + 0.5,
    })

    return df


@pytest.fixture
def sample_indicator_csv(temp_dir, sample_indicator_data) -> Path:
    """Create a sample indicator CSV file."""
    file_path = temp_dir / "test_data.csv"
    sample_indicator_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def simple_strategy_yaml(temp_dir) -> Path:
    """Create a simple strategy YAML file."""
    content = """strategies:
  - name: "TestStrategy"
    weight: 1.0
    combine: "any"
    rules:
      - indicator: "rsi_14"
        operator: "<"
        value: 30
        action: "buy"
        strength: 1.0
      - indicator: "rsi_14"
        operator: ">"
        value: 70
        action: "sell"
        strength: 1.0
"""
    file_path = temp_dir / "simple_strategy.yaml"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def multi_strategy_yaml(temp_dir) -> Path:
    """Create a multi-strategy YAML file."""
    content = """strategies:
  - name: "RSI_Strategy"
    weight: 0.5
    combine: "any"
    rules:
      - indicator: "rsi_14"
        operator: "<"
        value: 30
        action: "buy"
        strength: 1.0
      - indicator: "rsi_14"
        operator: ">"
        value: 70
        action: "sell"
        strength: 1.0

  - name: "MACD_Strategy"
    weight: 0.5
    combine: "all"
    rules:
      - indicator: "macd_hist_12_26_9"
        operator: ">"
        value: 0
        action: "buy"
        strength: 0.8
      - indicator: "macd_12_26_9"
        operator: ">"
        value_indicator: "macd_signal_12_26_9"
        action: "buy"
        strength: 0.8
"""
    file_path = temp_dir / "multi_strategy.yaml"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def indicator_comparison_strategy(temp_dir) -> Path:
    """Create a strategy that compares indicators."""
    content = """strategies:
  - name: "Indicator_Comparison"
    weight: 1.0
    combine: "any"
    rules:
      - indicator: "close"
        operator: "<"
        value_indicator: "bb_lower_20_2"
        action: "buy"
        strength: 1.0
      - indicator: "close"
        operator: ">"
        value_indicator: "bb_upper_20_2"
        action: "sell"
        strength: 1.0
"""
    file_path = temp_dir / "indicator_comparison.yaml"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def invalid_strategy_yaml(temp_dir) -> Path:
    """Create an invalid strategy YAML file."""
    content = """strategies:
  - name: "InvalidStrategy"
    weight: 1.0
    combine: "any"
"""
    file_path = temp_dir / "invalid_strategy.yaml"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def empty_yaml(temp_dir) -> Path:
    """Create an empty YAML file."""
    file_path = temp_dir / "empty.yaml"
    file_path.write_text("")
    return file_path


@pytest.fixture
def trending_up_data() -> pd.DataFrame:
    """Create data with clear upward trend for predictable signals."""
    n_days = 50
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")

    # Create trending data with low RSI at start, high at end
    rsi = np.linspace(25, 75, n_days)  # RSI goes from 25 to 75
    close = np.linspace(100, 150, n_days)  # Price trends up

    return pd.DataFrame({
        "Date": dates,
        "Open": close - 0.5,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": [1000000] * n_days,
        "rsi_14": rsi,
        "macd_hist_12_26_9": np.linspace(-2, 2, n_days),
        "macd_12_26_9": np.linspace(-1, 1, n_days),
        "macd_signal_12_26_9": np.zeros(n_days),
    })


@pytest.fixture
def oscillating_data() -> pd.DataFrame:
    """Create data with oscillating RSI for multiple buy/sell signals."""
    n_days = 100
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")

    # Create oscillating RSI that crosses thresholds
    rsi = 50 + 30 * np.sin(np.linspace(0, 6 * np.pi, n_days))
    close = 100 + 10 * np.sin(np.linspace(0, 6 * np.pi, n_days))

    return pd.DataFrame({
        "Date": dates,
        "Open": close - 0.5,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": [1000000] * n_days,
        "rsi_14": rsi,
    })


@pytest.fixture
def all_strategies_yaml(temp_dir) -> Path:
    """Create a strategy file with all operator types."""
    content = """strategies:
  - name: "AllOperators"
    weight: 1.0
    combine: "any"
    rules:
      - indicator: "rsi_14"
        operator: "<"
        value: 30
        action: "buy"
        strength: 1.0
      - indicator: "rsi_14"
        operator: "<="
        value: 25
        action: "buy"
        strength: 0.8
      - indicator: "rsi_14"
        operator: ">"
        value: 70
        action: "sell"
        strength: 1.0
      - indicator: "rsi_14"
        operator: ">="
        value: 75
        action: "sell"
        strength: 0.8
      - indicator: "rsi_14"
        operator: "=="
        value: 50
        action: "buy"
        strength: 0.5
      - indicator: "rsi_14"
        operator: "!="
        value: 50
        action: "sell"
        strength: 0.3
"""
    file_path = temp_dir / "all_operators.yaml"
    file_path.write_text(content)
    return file_path
