"""
Tests for the calculations module.

This module tests all technical indicator calculations including:
- Trend/Momentum: SMA, EMA, MACD, RSI, MFI
- Volatility: ATR, Historical Volatility
- Volume: OBV, Volume SMA
- Market Structure: Pivot Points, Bollinger Bands

Tests verify:
- Correct mathematical calculations
- NA handling for insufficient lookback
- Edge cases (all gains, all losses, zero volume, etc.)
"""

import numpy as np
import pandas as pd
import pytest

from backtest.indicators.calculations import (
    TRADING_DAYS_PER_YEAR,
    add_all_indicators,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_historical_volatility,
    calculate_macd,
    calculate_mfi,
    calculate_obv,
    calculate_pivot_high,
    calculate_pivot_low,
    calculate_rsi,
    calculate_sma,
    calculate_true_range,
    calculate_volume_sma,
)


class TestCalculateSMA:
    """Tests for calculate_sma function."""

    def test_sma_basic(self):
        """Test basic SMA calculation."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_sma(series, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)  # (1+2+3)/3
        assert result.iloc[3] == pytest.approx(3.0)  # (2+3+4)/3
        assert result.iloc[4] == pytest.approx(4.0)  # (3+4+5)/3

    def test_sma_insufficient_data(self):
        """Test SMA with insufficient data returns NaN."""
        series = pd.Series([1.0, 2.0])
        result = calculate_sma(series, 5)
        assert result.isna().all()

    def test_sma_exact_period(self):
        """Test SMA when data length equals period."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = calculate_sma(series, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)

    def test_sma_period_1(self):
        """Test SMA with period 1 returns original series."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = calculate_sma(series, 1)
        assert (result == series).all()

    def test_sma_nan_handling(self):
        """Test SMA handles NaN values correctly."""
        series = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        result = calculate_sma(series, 3)
        # NaN in window should make result NaN
        assert pd.isna(result.iloc[2])


class TestCalculateEMA:
    """Tests for calculate_ema function."""

    def test_ema_basic(self):
        """Test basic EMA calculation."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_ema(series, 3)
        # EMA starts from first value
        assert result.iloc[0] == 1.0
        # EMA should gradually approach the price
        assert result.iloc[-1] > result.iloc[0]

    def test_ema_converges(self):
        """Test that EMA converges to constant value for constant input."""
        series = pd.Series([5.0] * 100)
        result = calculate_ema(series, 10)
        assert result.iloc[-1] == pytest.approx(5.0)

    def test_ema_period_1(self):
        """Test EMA with period 1 returns original series."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = calculate_ema(series, 1)
        assert (result == series).all()

    def test_ema_formula(self):
        """Test EMA formula: alpha = 2/(period+1)."""
        series = pd.Series([1.0, 2.0])
        result = calculate_ema(series, 9)
        # alpha = 2/10 = 0.2
        # EMA[1] = 0.2 * 2.0 + 0.8 * 1.0 = 1.2
        assert result.iloc[1] == pytest.approx(1.2)


class TestCalculateMACD:
    """Tests for calculate_macd function."""

    def test_macd_basic(self, sample_ohlcv_data):
        """Test basic MACD calculation."""
        close = sample_ohlcv_data["Close"]
        macd, signal, hist = calculate_macd(close)
        assert len(macd) == len(close)
        assert len(signal) == len(close)
        assert len(hist) == len(close)

    def test_macd_histogram_equals_diff(self, sample_ohlcv_data):
        """Test that histogram = MACD - signal."""
        close = sample_ohlcv_data["Close"]
        macd, signal, hist = calculate_macd(close)
        expected_hist = macd - signal
        assert np.allclose(hist.values, expected_hist.values, rtol=1e-10)

    def test_macd_custom_periods(self, sample_ohlcv_data):
        """Test MACD with custom periods."""
        close = sample_ohlcv_data["Close"]
        macd, signal, hist = calculate_macd(close, 5, 10, 3)
        assert len(macd) == len(close)

    def test_macd_converges_for_constant(self):
        """Test MACD converges to 0 for constant price."""
        close = pd.Series([100.0] * 100)
        macd, signal, hist = calculate_macd(close)
        # For constant price, MACD should approach 0
        assert abs(macd.iloc[-1]) < 0.01


class TestCalculateRSI:
    """Tests for calculate_rsi function."""

    def test_rsi_range(self, sample_ohlcv_data):
        """Test that RSI is between 0 and 100."""
        close = sample_ohlcv_data["Close"]
        rsi = calculate_rsi(close)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_all_gains(self):
        """Test RSI with all positive changes."""
        # Use more data points for RSI to converge to extreme
        close = pd.Series([100.0 + i for i in range(100)])
        rsi = calculate_rsi(close)
        valid_rsi = rsi.dropna()
        # Should be very high (all gains) - with enough data points
        assert valid_rsi.iloc[-1] > 90

    def test_rsi_all_losses(self):
        """Test RSI with all negative changes."""
        close = pd.Series([100.0 - i for i in range(50)])
        rsi = calculate_rsi(close)
        # Should be close to 0 (all losses)
        assert rsi.iloc[-1] < 5

    def test_rsi_period(self):
        """Test RSI with custom period."""
        close = pd.Series([100 + np.sin(i / 10) * 5 for i in range(100)])
        rsi = calculate_rsi(close, period=7)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        close = pd.Series([100.0, 101.0, 102.0])
        rsi = calculate_rsi(close, period=14)
        # First 14 values should be NaN
        assert pd.isna(rsi.iloc[0])


class TestCalculateMFI:
    """Tests for calculate_mfi function."""

    def test_mfi_range(self, sample_ohlcv_data):
        """Test that MFI is between 0 and 100."""
        mfi = calculate_mfi(
            sample_ohlcv_data["High"],
            sample_ohlcv_data["Low"],
            sample_ohlcv_data["Close"],
            sample_ohlcv_data["Volume"],
        )
        valid_mfi = mfi.dropna()
        assert (valid_mfi >= 0).all()
        assert (valid_mfi <= 100).all()

    def test_mfi_all_positive_flow(self):
        """Test MFI with all positive money flow."""
        n = 100
        high = pd.Series([100.0 + i for i in range(n)])
        low = high - 1
        close = high - 0.5
        volume = pd.Series([1000000] * n)
        mfi = calculate_mfi(high, low, close, volume)
        valid_mfi = mfi.dropna()
        # Should be very high (all positive flow)
        assert valid_mfi.iloc[-1] > 90

    def test_mfi_insufficient_data(self, minimal_ohlcv_data):
        """Test MFI with minimal data."""
        mfi = calculate_mfi(
            minimal_ohlcv_data["High"],
            minimal_ohlcv_data["Low"],
            minimal_ohlcv_data["Close"],
            minimal_ohlcv_data["Volume"],
        )
        # First 14 values should be NaN
        assert pd.isna(mfi.iloc[0])


class TestCalculateTrueRange:
    """Tests for calculate_true_range function."""

    def test_tr_basic(self):
        """Test basic True Range calculation."""
        high = pd.Series([102.0, 104.0])
        low = pd.Series([100.0, 101.0])
        close = pd.Series([101.0, 103.0])
        tr = calculate_true_range(high, low, close)
        # First TR uses high-low
        assert tr.iloc[0] == pytest.approx(2.0)
        # Second TR: max(104-101, |104-101|, |101-101|) = 3
        assert tr.iloc[1] == pytest.approx(3.0)

    def test_tr_gap_up(self):
        """Test True Range with gap up."""
        high = pd.Series([102.0, 110.0])
        low = pd.Series([100.0, 108.0])
        close = pd.Series([101.0, 109.0])
        tr = calculate_true_range(high, low, close)
        # Gap up: |110-101| = 9 > 110-108 = 2
        assert tr.iloc[1] == pytest.approx(9.0)

    def test_tr_gap_down(self):
        """Test True Range with gap down."""
        high = pd.Series([102.0, 95.0])
        low = pd.Series([100.0, 93.0])
        close = pd.Series([101.0, 94.0])
        tr = calculate_true_range(high, low, close)
        # Gap down: |93-101| = 8 > 95-93 = 2
        assert tr.iloc[1] == pytest.approx(8.0)


class TestCalculateATR:
    """Tests for calculate_atr function."""

    def test_atr_basic(self, sample_ohlcv_data):
        """Test basic ATR calculation."""
        atr = calculate_atr(
            sample_ohlcv_data["High"],
            sample_ohlcv_data["Low"],
            sample_ohlcv_data["Close"],
        )
        assert len(atr) == len(sample_ohlcv_data)
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()

    def test_atr_insufficient_data(self, minimal_ohlcv_data):
        """Test ATR with insufficient data."""
        atr = calculate_atr(
            minimal_ohlcv_data["High"],
            minimal_ohlcv_data["Low"],
            minimal_ohlcv_data["Close"],
        )
        # First 14 values should be NaN
        assert pd.isna(atr.iloc[0])


class TestCalculateHistoricalVolatility:
    """Tests for calculate_historical_volatility function."""

    def test_hv_basic(self, sample_ohlcv_data):
        """Test basic HV calculation."""
        hv = calculate_historical_volatility(sample_ohlcv_data["Close"])
        assert len(hv) == len(sample_ohlcv_data)
        valid_hv = hv.dropna()
        assert (valid_hv >= 0).all()

    def test_hv_annualized(self, sample_ohlcv_data):
        """Test that HV is annualized by default."""
        hv_ann = calculate_historical_volatility(
            sample_ohlcv_data["Close"], annualize=True
        )
        hv_not_ann = calculate_historical_volatility(
            sample_ohlcv_data["Close"], annualize=False
        )
        # Annualized should be larger by sqrt(252) factor
        ratio = hv_ann.dropna().iloc[-1] / hv_not_ann.dropna().iloc[-1]
        assert ratio == pytest.approx(np.sqrt(TRADING_DAYS_PER_YEAR), rel=0.01)

    def test_hv_constant_price(self):
        """Test HV for constant price is 0."""
        close = pd.Series([100.0] * 100)
        hv = calculate_historical_volatility(close)
        assert hv.dropna().iloc[-1] == 0.0

    def test_hv_period(self, sample_ohlcv_data):
        """Test HV with custom period."""
        hv = calculate_historical_volatility(sample_ohlcv_data["Close"], period=10)
        # Should have more valid values with shorter period
        assert hv.dropna().count() >= len(sample_ohlcv_data) - 10


class TestCalculateOBV:
    """Tests for calculate_obv function."""

    def test_obv_basic(self):
        """Test basic OBV calculation."""
        close = pd.Series([100.0, 101.0, 100.0, 102.0])
        volume = pd.Series([1000.0, 2000.0, 1500.0, 2500.0])
        obv = calculate_obv(close, volume)
        # Day 1: +1000 (first day, direction set to +1)
        # Day 2: +2000 (up)
        # Day 3: -1500 (down)
        # Day 4: +2500 (up)
        expected = [1000.0, 3000.0, 1500.0, 4000.0]
        assert obv.iloc[0] == pytest.approx(expected[0])
        assert obv.iloc[1] == pytest.approx(expected[1])
        assert obv.iloc[2] == pytest.approx(expected[2])
        assert obv.iloc[3] == pytest.approx(expected[3])

    def test_obv_all_up(self):
        """Test OBV with all up days."""
        close = pd.Series([100.0 + i for i in range(5)])
        volume = pd.Series([1000.0] * 5)
        obv = calculate_obv(close, volume)
        # All days up: cumulative sum = 5 * 1000 = 5000
        assert obv.iloc[-1] == pytest.approx(5000.0)

    def test_obv_all_down(self):
        """Test OBV with all down days."""
        close = pd.Series([100.0 - i for i in range(5)])
        volume = pd.Series([1000.0] * 5)
        obv = calculate_obv(close, volume)
        # First day: +1000 (direction set to +1)
        # Days 2-5: -1000 each (down)
        # Total: 1000 - 4*1000 = -3000
        assert obv.iloc[-1] == pytest.approx(-3000.0)

    def test_obv_unchanged_price(self):
        """Test OBV when price unchanged."""
        close = pd.Series([100.0, 100.0, 100.0])
        volume = pd.Series([1000.0, 2000.0, 3000.0])
        obv = calculate_obv(close, volume)
        # First day: +1000 (direction set to +1)
        # Day 2: unchanged (direction = 0), adds 0
        # Day 3: unchanged (direction = 0), adds 0
        assert obv.iloc[0] == pytest.approx(1000.0)
        assert obv.iloc[1] == pytest.approx(1000.0)
        assert obv.iloc[2] == pytest.approx(1000.0)


class TestCalculateVolumeSMA:
    """Tests for calculate_volume_sma function."""

    def test_vol_sma_basic(self, sample_ohlcv_data):
        """Test basic volume SMA calculation."""
        vol_sma = calculate_volume_sma(sample_ohlcv_data["Volume"])
        assert len(vol_sma) == len(sample_ohlcv_data)
        valid = vol_sma.dropna()
        assert (valid > 0).all()

    def test_vol_sma_custom_period(self, sample_ohlcv_data):
        """Test volume SMA with custom period."""
        vol_sma = calculate_volume_sma(sample_ohlcv_data["Volume"], period=10)
        # Should have more valid values with shorter period
        assert vol_sma.dropna().count() >= len(sample_ohlcv_data) - 10


class TestCalculatePivotHigh:
    """Tests for calculate_pivot_high function."""

    def test_pivot_high_basic(self):
        """Test basic pivot high detection."""
        high = pd.Series([10, 11, 12, 11, 10, 11, 13, 12, 11, 10])
        pivot = calculate_pivot_high(high, 2, 2)
        # Position 2 (value 12) should be pivot high
        assert pivot.iloc[2] == 12.0
        # Position 6 (value 13) should be pivot high
        assert pivot.iloc[6] == 13.0
        # Other positions should be NaN
        assert pd.isna(pivot.iloc[0])
        assert pd.isna(pivot.iloc[4])

    def test_pivot_high_edge(self):
        """Test pivot high at edges (should be NaN)."""
        high = pd.Series([15, 12, 11, 10, 11, 12, 13, 14, 15, 20])
        pivot = calculate_pivot_high(high, 3, 3)
        # First and last 3 positions should be NaN (edge)
        assert pd.isna(pivot.iloc[0])
        assert pd.isna(pivot.iloc[-1])

    def test_pivot_high_no_pivots(self):
        """Test when no pivots exist (monotonic)."""
        high = pd.Series([i for i in range(20)])  # Always increasing
        pivot = calculate_pivot_high(high, 3, 3)
        assert pivot.dropna().count() == 0


class TestCalculatePivotLow:
    """Tests for calculate_pivot_low function."""

    def test_pivot_low_basic(self):
        """Test basic pivot low detection."""
        low = pd.Series([12, 11, 10, 11, 12, 11, 9, 10, 11, 12])
        pivot = calculate_pivot_low(low, 2, 2)
        # Position 2 (value 10) should be pivot low
        assert pivot.iloc[2] == 10.0
        # Position 6 (value 9) should be pivot low
        assert pivot.iloc[6] == 9.0
        # Other positions should be NaN
        assert pd.isna(pivot.iloc[0])
        assert pd.isna(pivot.iloc[4])

    def test_pivot_low_edge(self):
        """Test pivot low at edges (should be NaN)."""
        low = pd.Series([5, 8, 9, 10, 9, 8, 7, 6, 5, 1])
        pivot = calculate_pivot_low(low, 3, 3)
        # First and last 3 positions should be NaN (edge)
        assert pd.isna(pivot.iloc[0])
        assert pd.isna(pivot.iloc[-1])

    def test_pivot_low_no_pivots(self):
        """Test when no pivots exist (monotonic)."""
        low = pd.Series([20 - i for i in range(20)])  # Always decreasing
        pivot = calculate_pivot_low(low, 3, 3)
        assert pivot.dropna().count() == 0


class TestCalculateBollingerBands:
    """Tests for calculate_bollinger_bands function."""

    def test_bb_basic(self, sample_ohlcv_data):
        """Test basic Bollinger Bands calculation."""
        mid, upper, lower = calculate_bollinger_bands(sample_ohlcv_data["Close"])
        assert len(mid) == len(sample_ohlcv_data)
        # Upper > mid > lower
        valid_idx = mid.dropna().index
        assert (upper.loc[valid_idx] > mid.loc[valid_idx]).all()
        assert (mid.loc[valid_idx] > lower.loc[valid_idx]).all()

    def test_bb_mid_equals_sma(self, sample_ohlcv_data):
        """Test that middle band equals SMA."""
        mid, _, _ = calculate_bollinger_bands(sample_ohlcv_data["Close"], period=20)
        sma = calculate_sma(sample_ohlcv_data["Close"], 20)
        valid_idx = mid.dropna().index
        assert np.allclose(mid.loc[valid_idx], sma.loc[valid_idx])

    def test_bb_std_dev_multiplier(self, sample_ohlcv_data):
        """Test Bollinger Bands with different std dev multiplier."""
        close = sample_ohlcv_data["Close"]
        mid1, upper1, lower1 = calculate_bollinger_bands(close, std_dev=2.0)
        mid2, upper2, lower2 = calculate_bollinger_bands(close, std_dev=3.0)
        # Wider bands with higher std dev
        valid_idx = mid1.dropna().index
        assert ((upper2 - lower2) > (upper1 - lower1)).loc[valid_idx].all()

    def test_bb_constant_price(self):
        """Test Bollinger Bands with constant price."""
        close = pd.Series([100.0] * 50)
        mid, upper, lower = calculate_bollinger_bands(close)
        valid_idx = mid.dropna().index
        # All bands should equal price (std = 0)
        assert (mid.loc[valid_idx] == 100.0).all()
        assert (upper.loc[valid_idx] == 100.0).all()
        assert (lower.loc[valid_idx] == 100.0).all()


class TestAddAllIndicators:
    """Tests for add_all_indicators function (integration)."""

    def test_all_indicators_added(self, sample_ohlcv_data):
        """Test that all required indicators are added."""
        result = add_all_indicators(sample_ohlcv_data)

        expected_indicators = [
            "sma_5", "sma_9", "sma_20", "sma_50", "sma_200",
            "ema_9", "ema_21", "ema_50",
            "macd_12_26_9", "macd_signal_12_26_9", "macd_hist_12_26_9",
            "rsi_14", "mfi_14",
            "atr_14", "hv_20",
            "obv", "vol_sma_20",
            "pivot_high_3", "pivot_low_3",
            "bb_mid_20_2", "bb_upper_20_2", "bb_lower_20_2",
        ]

        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing indicator: {indicator}"

    def test_original_columns_preserved(self, sample_ohlcv_data):
        """Test that original OHLCV columns are preserved."""
        result = add_all_indicators(sample_ohlcv_data)
        for col in ["Date", "Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns
            assert (result[col] == sample_ohlcv_data[col]).all()

    def test_does_not_modify_input(self, sample_ohlcv_data):
        """Test that input DataFrame is not modified."""
        original_columns = list(sample_ohlcv_data.columns)
        add_all_indicators(sample_ohlcv_data)
        assert list(sample_ohlcv_data.columns) == original_columns

    def test_handles_minimal_data(self, minimal_ohlcv_data):
        """Test with minimal data (10 rows)."""
        result = add_all_indicators(minimal_ohlcv_data)
        assert len(result) == 10
        # Some indicators will be NaN due to insufficient lookback
        assert "sma_5" in result.columns

    def test_large_dataset_performance(self, large_dataset):
        """Test performance with large dataset."""
        from backtest.indicators.loader import load_and_prepare

        df = load_and_prepare(str(large_dataset))
        result = add_all_indicators(df)
        assert len(result) == 5000
        # Verify indicators are calculated
        assert not result["sma_200"].dropna().empty

    def test_indicator_values_reasonable(self, sample_ohlcv_data):
        """Test that indicator values are in reasonable ranges."""
        result = add_all_indicators(sample_ohlcv_data)

        # RSI should be 0-100
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

        # MFI should be 0-100
        mfi = result["mfi_14"].dropna()
        assert (mfi >= 0).all() and (mfi <= 100).all()

        # ATR should be positive
        atr = result["atr_14"].dropna()
        assert (atr >= 0).all()

        # BB upper > lower
        bb_upper = result["bb_upper_20_2"].dropna()
        bb_lower = result["bb_lower_20_2"].dropna()
        assert (bb_upper > bb_lower).all()
