"""
Unit tests for activation functions.

Tests all activation functions in the activations module for:
- Correct output range [0, 1]
- Proper behavior at boundary conditions
- Data-driven normalizing scale computation
- Indicator-specific activations (BB, RSI, MFI, MACD, etc.)
"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine.activations import (
    clip_activation,
    compute_normalizing_scale,
    activation_less_than,
    activation_greater_than,
    activation_less_than_indicator,
    activation_greater_than_indicator,
    activation_bb_lower,
    activation_bb_upper,
    activation_rsi_oversold,
    activation_rsi_overbought,
    activation_mfi_oversold,
    activation_mfi_overbought,
    activation_volume_spike,
    activation_sma_trend_bullish,
    activation_sma_trend_bearish,
    activation_macd_histogram_positive,
    activation_macd_histogram_negative,
    activation_macd_crossover_bullish,
    activation_macd_crossover_bearish,
    compute_rule_activation,
)


class TestClipActivation:
    """Tests for clip_activation function."""

    def test_values_in_range(self):
        """Values in [0, 1] should remain unchanged."""
        values = pd.Series([0.0, 0.5, 1.0])
        result = clip_activation(values)
        pd.testing.assert_series_equal(result, values)

    def test_values_below_zero(self):
        """Values below 0 should be clipped to 0."""
        values = pd.Series([-0.5, -1.0, -100.0])
        result = clip_activation(values)
        assert (result == 0.0).all()

    def test_values_above_one(self):
        """Values above 1 should be clipped to 1."""
        values = pd.Series([1.5, 2.0, 100.0])
        result = clip_activation(values)
        assert (result == 1.0).all()

    def test_mixed_values(self):
        """Mixed values should be clipped appropriately."""
        values = pd.Series([-1.0, 0.0, 0.5, 1.0, 2.0])
        expected = pd.Series([0.0, 0.0, 0.5, 1.0, 1.0])
        result = clip_activation(values)
        pd.testing.assert_series_equal(result, expected)


class TestComputeNormalizingScale:
    """Tests for compute_normalizing_scale function."""

    def test_std_method(self):
        """Standard deviation method should use indicator std."""
        indicator = pd.Series([100.0, 110.0, 90.0, 105.0, 95.0])
        scale = compute_normalizing_scale(indicator, 100.0, method="std")
        # Should be constant since no lookback
        assert (scale > 0).all()
        assert len(scale.unique()) == 1

    def test_range_method(self):
        """Range method should use (max - min) / 4."""
        indicator = pd.Series([100.0, 120.0, 80.0])
        scale = compute_normalizing_scale(indicator, 100.0, method="range")
        expected = (120.0 - 80.0) / 4  # 10
        assert abs(scale.iloc[0] - expected) < 0.01

    def test_threshold_pct_method(self):
        """Threshold percentage method should use 20% of threshold."""
        indicator = pd.Series([100.0, 110.0, 90.0])
        scale = compute_normalizing_scale(indicator, 100.0, method="threshold_pct")
        expected = 100.0 * 0.2  # 20
        assert abs(scale.iloc[0] - expected) < 0.01

    def test_minimum_scale(self):
        """Scale should never be below 0.001."""
        indicator = pd.Series([100.0, 100.0, 100.0])  # No variation
        scale = compute_normalizing_scale(indicator, 100.0)
        assert (scale >= 0.001).all()

    def test_rolling_lookback(self):
        """Rolling lookback should compute rolling scale."""
        indicator = pd.Series([100.0, 110.0, 90.0, 105.0, 95.0, 120.0])
        scale = compute_normalizing_scale(indicator, 100.0, method="std", lookback=3)
        # Should vary over time
        assert len(scale.dropna().unique()) > 1


class TestActivationLessThan:
    """Tests for activation_less_than function."""

    def test_exactly_at_threshold(self):
        """At threshold, activation should be 0."""
        indicator = pd.Series([30.0, 30.0, 30.0])
        result = activation_less_than(indicator, 30.0)
        assert (result == 0.0).all()

    def test_above_threshold(self):
        """Above threshold, activation should be 0."""
        indicator = pd.Series([35.0, 40.0, 50.0])
        result = activation_less_than(indicator, 30.0)
        assert (result == 0.0).all()

    def test_below_threshold(self):
        """Below threshold, activation should be > 0."""
        indicator = pd.Series([25.0, 20.0, 15.0])
        result = activation_less_than(indicator, 30.0)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_deeper_gives_higher_activation(self):
        """Deeper below threshold should give higher activation."""
        # Use a wider range to avoid saturation, with more variance
        # The std of this series is larger, so small differences won't saturate
        indicator = pd.Series([29.9, 29.5, 29.0, 27.0, 24.0, 20.0])
        result = activation_less_than(indicator, 30.0)
        # Activation should increase as we go deeper (for non-saturated values)
        # Earlier values closer to threshold should have lower activation
        assert result.iloc[0] < result.iloc[-1]
        # All activations should be positive since all values are below threshold
        assert (result > 0.0).all()


class TestActivationGreaterThan:
    """Tests for activation_greater_than function."""

    def test_exactly_at_threshold(self):
        """At threshold, activation should be 0."""
        indicator = pd.Series([70.0, 70.0, 70.0])
        result = activation_greater_than(indicator, 70.0)
        assert (result == 0.0).all()

    def test_below_threshold(self):
        """Below threshold, activation should be 0."""
        indicator = pd.Series([65.0, 60.0, 50.0])
        result = activation_greater_than(indicator, 70.0)
        assert (result == 0.0).all()

    def test_above_threshold(self):
        """Above threshold, activation should be > 0."""
        indicator = pd.Series([75.0, 80.0, 85.0])
        result = activation_greater_than(indicator, 70.0)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()


class TestActivationIndicatorComparison:
    """Tests for indicator-to-indicator comparison activations."""

    def test_less_than_indicator_satisfied(self):
        """Activation when indicator < comparison_indicator."""
        indicator = pd.Series([95.0, 90.0, 85.0])
        comparison = pd.Series([100.0, 100.0, 100.0])
        result = activation_less_than_indicator(indicator, comparison)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_less_than_indicator_not_satisfied(self):
        """No activation when indicator >= comparison_indicator."""
        indicator = pd.Series([105.0, 110.0, 115.0])
        comparison = pd.Series([100.0, 100.0, 100.0])
        result = activation_less_than_indicator(indicator, comparison)
        assert (result == 0.0).all()

    def test_greater_than_indicator_satisfied(self):
        """Activation when indicator > comparison_indicator."""
        indicator = pd.Series([105.0, 110.0, 115.0])
        comparison = pd.Series([100.0, 100.0, 100.0])
        result = activation_greater_than_indicator(indicator, comparison)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()


class TestBollingerBandActivation:
    """Tests for Bollinger Band activation functions."""

    def test_bb_lower_below_band(self):
        """Activation when close is below lower band."""
        close = pd.Series([95.0, 90.0, 85.0])
        bb_lower = pd.Series([100.0, 100.0, 100.0])
        bb_middle = pd.Series([110.0, 110.0, 110.0])
        result = activation_bb_lower(close, bb_lower, bb_middle)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_bb_lower_above_band(self):
        """No activation when close is above lower band."""
        close = pd.Series([105.0, 110.0, 115.0])
        bb_lower = pd.Series([100.0, 100.0, 100.0])
        result = activation_bb_lower(close, bb_lower)
        assert (result == 0.0).all()

    def test_bb_upper_above_band(self):
        """Activation when close is above upper band."""
        close = pd.Series([125.0, 130.0, 135.0])
        bb_upper = pd.Series([120.0, 120.0, 120.0])
        result = activation_bb_upper(close, bb_upper)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_bb_upper_below_band(self):
        """No activation when close is below upper band."""
        close = pd.Series([115.0, 110.0, 105.0])
        bb_upper = pd.Series([120.0, 120.0, 120.0])
        result = activation_bb_upper(close, bb_upper)
        assert (result == 0.0).all()


class TestRSIActivation:
    """Tests for RSI activation functions."""

    def test_rsi_oversold_below_threshold(self):
        """Activation when RSI is below oversold threshold."""
        rsi = pd.Series([25.0, 20.0, 15.0])
        result = activation_rsi_oversold(rsi, threshold=30.0)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_rsi_oversold_at_zero(self):
        """Full activation when RSI is at 0."""
        rsi = pd.Series([0.0])
        result = activation_rsi_oversold(rsi, threshold=30.0)
        assert result.iloc[0] == 1.0

    def test_rsi_oversold_above_threshold(self):
        """No activation when RSI is above oversold threshold."""
        rsi = pd.Series([35.0, 50.0, 70.0])
        result = activation_rsi_oversold(rsi, threshold=30.0)
        assert (result == 0.0).all()

    def test_rsi_overbought_above_threshold(self):
        """Activation when RSI is above overbought threshold."""
        rsi = pd.Series([75.0, 80.0, 85.0])
        result = activation_rsi_overbought(rsi, threshold=70.0)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_rsi_overbought_at_100(self):
        """Full activation when RSI is at 100."""
        rsi = pd.Series([100.0])
        result = activation_rsi_overbought(rsi, threshold=70.0)
        assert result.iloc[0] == 1.0

    def test_rsi_overbought_below_threshold(self):
        """No activation when RSI is below overbought threshold."""
        rsi = pd.Series([65.0, 50.0, 30.0])
        result = activation_rsi_overbought(rsi, threshold=70.0)
        assert (result == 0.0).all()


class TestMFIActivation:
    """Tests for MFI activation functions."""

    def test_mfi_oversold_below_threshold(self):
        """Activation when MFI is below oversold threshold."""
        mfi = pd.Series([15.0, 10.0, 5.0])
        result = activation_mfi_oversold(mfi, threshold=20.0)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_mfi_oversold_above_threshold(self):
        """No activation when MFI is above oversold threshold."""
        mfi = pd.Series([25.0, 50.0, 70.0])
        result = activation_mfi_oversold(mfi, threshold=20.0)
        assert (result == 0.0).all()

    def test_mfi_overbought_above_threshold(self):
        """Activation when MFI is above overbought threshold."""
        mfi = pd.Series([85.0, 90.0, 95.0])
        result = activation_mfi_overbought(mfi, threshold=80.0)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_mfi_overbought_below_threshold(self):
        """No activation when MFI is below overbought threshold."""
        mfi = pd.Series([75.0, 50.0, 30.0])
        result = activation_mfi_overbought(mfi, threshold=80.0)
        assert (result == 0.0).all()


class TestVolumeActivation:
    """Tests for volume spike activation function."""

    def test_volume_above_average(self):
        """Activation when volume exceeds average."""
        volume = pd.Series([150000.0, 200000.0, 250000.0])
        vol_sma = pd.Series([100000.0, 100000.0, 100000.0])
        result = activation_volume_spike(volume, vol_sma, multiplier=2.0)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_volume_at_multiplier(self):
        """Full activation when volume is at multiplier times average."""
        volume = pd.Series([200000.0])
        vol_sma = pd.Series([100000.0])
        result = activation_volume_spike(volume, vol_sma, multiplier=2.0)
        assert result.iloc[0] == 1.0

    def test_volume_below_average(self):
        """No activation when volume is below average."""
        volume = pd.Series([80000.0, 50000.0, 30000.0])
        vol_sma = pd.Series([100000.0, 100000.0, 100000.0])
        result = activation_volume_spike(volume, vol_sma)
        assert (result == 0.0).all()


class TestSMATrendActivation:
    """Tests for SMA/EMA trend activation functions."""

    def test_bullish_price_above_sma(self):
        """Activation when price is above SMA."""
        close = pd.Series([105.0, 110.0, 115.0])
        sma = pd.Series([100.0, 100.0, 100.0])
        result = activation_sma_trend_bullish(close, sma)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_bullish_price_below_sma(self):
        """No activation when price is below SMA."""
        close = pd.Series([95.0, 90.0, 85.0])
        sma = pd.Series([100.0, 100.0, 100.0])
        result = activation_sma_trend_bullish(close, sma)
        assert (result == 0.0).all()

    def test_bullish_with_slope(self):
        """Combined activation with rising SMA."""
        close = pd.Series([105.0, 110.0, 115.0])
        sma = pd.Series([100.0, 101.0, 102.0])
        sma_lag = pd.Series([99.0, 100.0, 101.0])
        result = activation_sma_trend_bullish(close, sma, sma_lag)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_bearish_price_below_sma(self):
        """Activation when price is below SMA."""
        close = pd.Series([95.0, 90.0, 85.0])
        sma = pd.Series([100.0, 100.0, 100.0])
        result = activation_sma_trend_bearish(close, sma)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()


class TestMACDActivation:
    """Tests for MACD activation functions."""

    def test_histogram_positive(self):
        """Activation for positive MACD histogram."""
        macd_hist = pd.Series([0.5, 1.0, 1.5])
        result = activation_macd_histogram_positive(macd_hist)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_histogram_negative_no_positive_activation(self):
        """No positive activation for negative histogram."""
        macd_hist = pd.Series([-0.5, -1.0, -1.5])
        result = activation_macd_histogram_positive(macd_hist)
        assert (result == 0.0).all()

    def test_histogram_negative(self):
        """Activation for negative MACD histogram."""
        macd_hist = pd.Series([-0.5, -1.0, -1.5])
        result = activation_macd_histogram_negative(macd_hist)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_bullish_crossover(self):
        """Activation when MACD > signal."""
        macd = pd.Series([1.0, 1.5, 2.0])
        signal = pd.Series([0.5, 0.5, 0.5])
        result = activation_macd_crossover_bullish(macd, signal)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_bearish_crossover(self):
        """Activation when MACD < signal."""
        macd = pd.Series([0.0, -0.5, -1.0])
        signal = pd.Series([0.5, 0.5, 0.5])
        result = activation_macd_crossover_bearish(macd, signal)
        assert (result > 0.0).all()
        assert (result <= 1.0).all()


class TestComputeRuleActivation:
    """Tests for the generic rule activation dispatcher."""

    def test_rsi_detection(self):
        """Should detect RSI indicator and use specialized activation."""
        indicator = pd.Series([25.0, 20.0, 15.0])
        result = compute_rule_activation(
            indicator_values=indicator,
            comparison_values=30.0,
            operator="<",
            indicator_name="rsi_14",
        )
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_mfi_detection(self):
        """Should detect MFI indicator and use specialized activation."""
        indicator = pd.Series([85.0, 90.0, 95.0])
        result = compute_rule_activation(
            indicator_values=indicator,
            comparison_values=80.0,
            operator=">",
            indicator_name="mfi_14",
        )
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_generic_less_than(self):
        """Should use generic less_than for unknown indicators."""
        indicator = pd.Series([25.0, 20.0, 15.0])
        result = compute_rule_activation(
            indicator_values=indicator,
            comparison_values=30.0,
            operator="<",
            indicator_name="some_indicator",
        )
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_generic_greater_than(self):
        """Should use generic greater_than for unknown indicators."""
        indicator = pd.Series([35.0, 40.0, 45.0])
        result = compute_rule_activation(
            indicator_values=indicator,
            comparison_values=30.0,
            operator=">",
            indicator_name="some_indicator",
        )
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_indicator_comparison(self):
        """Should handle indicator-to-indicator comparison."""
        indicator = pd.Series([95.0, 90.0, 85.0])
        comparison = pd.Series([100.0, 100.0, 100.0])
        result = compute_rule_activation(
            indicator_values=indicator,
            comparison_values=comparison,
            operator="<",
            indicator_name="close",
            comparison_name="sma_200",
        )
        assert (result > 0.0).all()
        assert (result <= 1.0).all()

    def test_output_always_in_range(self):
        """Output should always be in [0, 1] regardless of input."""
        # Test with extreme values
        indicator = pd.Series([-1000.0, 0.0, 1000.0])
        result = compute_rule_activation(
            indicator_values=indicator,
            comparison_values=0.0,
            operator="<",
            indicator_name="test",
        )
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()


class TestActivationRangeInvariants:
    """Tests to ensure activation functions always return values in [0, 1]."""

    @pytest.mark.parametrize("values", [
        [float('inf'), float('-inf'), float('nan')],
        [1e10, -1e10, 0],
        [0.0001, -0.0001, 0],
    ])
    def test_extreme_values_rsi(self, values):
        """RSI activation should handle extreme values."""
        rsi = pd.Series(values)
        result = activation_rsi_oversold(rsi, 30.0)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    @pytest.mark.parametrize("threshold", [0, 30, 50, 70, 100])
    def test_various_thresholds_rsi(self, threshold):
        """RSI activation should work with various thresholds."""
        rsi = pd.Series([10.0, 30.0, 50.0, 70.0, 90.0])
        result = activation_rsi_oversold(rsi, float(threshold))
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()
