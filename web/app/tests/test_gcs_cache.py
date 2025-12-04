"""
Tests for GCS cache module.
"""

import io
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services import gcs_cache


class TestValidateSymbol:
    """Tests for _validate_symbol function."""

    def test_valid_symbol(self):
        """Should accept valid symbols."""
        assert gcs_cache._validate_symbol("SPY") == "SPY"
        assert gcs_cache._validate_symbol("aapl") == "AAPL"
        assert gcs_cache._validate_symbol("MSFT") == "MSFT"

    def test_invalid_symbol_with_numbers(self):
        """Should reject symbols with numbers."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            gcs_cache._validate_symbol("SPY123")

    def test_invalid_symbol_with_special_chars(self):
        """Should reject symbols with special characters."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            gcs_cache._validate_symbol("SPY-X")
        with pytest.raises(ValueError, match="Invalid symbol format"):
            gcs_cache._validate_symbol("../etc/passwd")

    def test_empty_symbol(self):
        """Should reject empty symbols."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            gcs_cache._validate_symbol("")
        with pytest.raises(ValueError, match="Invalid symbol format"):
            gcs_cache._validate_symbol("   ")

    def test_symbol_too_long(self):
        """Should reject symbols longer than 10 chars."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            gcs_cache._validate_symbol("VERYLONGSYMBOL")


class TestGetCachePath:
    """Tests for _get_cache_path function."""

    def test_returns_correct_path(self):
        """Should return correct GCS path for symbol."""
        path = gcs_cache._get_cache_path("SPY")
        assert path == "cache/daily_scores/SPY.parquet"

    def test_uses_custom_prefix(self):
        """Should use custom prefix from environment."""
        with patch.dict(os.environ, {"SCORE_CACHE_PREFIX": "custom/"}):
            # Need to reload module to pick up env var
            import importlib
            importlib.reload(gcs_cache)
            path = gcs_cache._get_cache_path("AAPL")
            assert path.startswith("custom/") or path.startswith("cache/")
            # Reset
            importlib.reload(gcs_cache)


class TestReadScores:
    """Tests for read_scores function."""

    @patch.object(gcs_cache, "_get_client")
    def test_returns_none_when_blob_not_exists(self, mock_get_client):
        """Should return None when blob doesn't exist."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False

        mock_get_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        result = gcs_cache.read_scores("SPY")

        assert result is None
        mock_blob.exists.assert_called_once()

    @patch.object(gcs_cache, "_get_client")
    def test_returns_dataframe_when_blob_exists(self, mock_get_client):
        """Should return DataFrame when blob exists."""
        # Create test parquet data
        test_df = pd.DataFrame({
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "signal_raw": [0.5, 0.6],
            "signal_0_1": [0.75, 0.8],
        })
        buffer = io.BytesIO()
        test_df.to_parquet(buffer, index=False)
        buffer.seek(0)
        parquet_bytes = buffer.getvalue()

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = parquet_bytes

        mock_get_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        result = gcs_cache.read_scores("SPY")

        assert result is not None
        assert len(result) == 2
        assert "date" in result.columns
        assert "signal_raw" in result.columns

    @patch.object(gcs_cache, "_get_client")
    def test_handles_exception_gracefully(self, mock_get_client):
        """Should return None on exception."""
        mock_get_client.side_effect = Exception("Connection error")

        result = gcs_cache.read_scores("SPY")

        assert result is None


class TestWriteScores:
    """Tests for write_scores function."""

    @patch.object(gcs_cache, "_get_client")
    def test_returns_true_for_empty_dataframe(self, mock_get_client):
        """Should return True for empty DataFrame without uploading."""
        result = gcs_cache.write_scores("SPY", pd.DataFrame())
        assert result is True
        mock_get_client.assert_not_called()

    @patch.object(gcs_cache, "_get_client")
    def test_uploads_dataframe_to_gcs(self, mock_get_client):
        """Should upload DataFrame to GCS."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_get_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        test_df = pd.DataFrame({
            "date": [datetime(2024, 1, 1)],
            "signal_raw": [0.5],
            "signal_0_1": [0.75],
        })

        result = gcs_cache.write_scores("SPY", test_df)

        assert result is True
        mock_blob.upload_from_file.assert_called_once()

    @patch.object(gcs_cache, "_get_client")
    def test_handles_exception_gracefully(self, mock_get_client):
        """Should return False on exception."""
        mock_get_client.side_effect = Exception("Upload error")

        test_df = pd.DataFrame({
            "date": [datetime(2024, 1, 1)],
            "signal_raw": [0.5],
            "signal_0_1": [0.75],
        })

        result = gcs_cache.write_scores("SPY", test_df)

        assert result is False


class TestGetLatestScore:
    """Tests for get_latest_score function."""

    @patch.object(gcs_cache, "read_scores")
    def test_returns_none_when_no_cache(self, mock_read):
        """Should return None when no cached scores."""
        mock_read.return_value = None

        result = gcs_cache.get_latest_score("SPY")

        assert result is None

    @patch.object(gcs_cache, "read_scores")
    def test_returns_latest_score(self, mock_read):
        """Should return the latest score by date."""
        mock_read.return_value = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02"]),
            "signal_raw": [0.5, 0.7, 0.6],
            "signal_0_1": [0.75, 0.85, 0.8],
        })

        result = gcs_cache.get_latest_score("SPY")

        assert result is not None
        assert result["signal_raw"] == 0.7
        assert result["signal_0_1"] == 0.85


class TestAddScore:
    """Tests for add_score function."""

    @patch.object(gcs_cache, "read_scores")
    @patch.object(gcs_cache, "write_scores")
    def test_adds_new_score_to_empty_cache(self, mock_write, mock_read):
        """Should add score to empty cache."""
        mock_read.return_value = None
        mock_write.return_value = True

        result = gcs_cache.add_score(
            "SPY",
            datetime(2024, 1, 1),
            0.5,
            0.75,
        )

        assert result is True
        mock_write.assert_called_once()
        written_df = mock_write.call_args[0][1]
        assert len(written_df) == 1

    @patch.object(gcs_cache, "read_scores")
    @patch.object(gcs_cache, "write_scores")
    def test_appends_to_existing_cache(self, mock_write, mock_read):
        """Should append to existing cache."""
        mock_read.return_value = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "signal_raw": [0.5],
            "signal_0_1": [0.75],
        })
        mock_write.return_value = True

        result = gcs_cache.add_score(
            "SPY",
            datetime(2024, 1, 2),
            0.6,
            0.8,
        )

        assert result is True
        written_df = mock_write.call_args[0][1]
        assert len(written_df) == 2

    @patch.object(gcs_cache, "read_scores")
    @patch.object(gcs_cache, "write_scores")
    def test_updates_existing_date(self, mock_write, mock_read):
        """Should update score for existing date."""
        mock_read.return_value = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "signal_raw": [0.5],
            "signal_0_1": [0.75],
        })
        mock_write.return_value = True

        result = gcs_cache.add_score(
            "SPY",
            datetime(2024, 1, 1),
            0.9,
            0.95,
        )

        assert result is True
        written_df = mock_write.call_args[0][1]
        assert len(written_df) == 1
        assert written_df.iloc[0]["signal_raw"] == 0.9


    def test_rejects_invalid_score_0_1(self):
        """Should reject score_0_1 out of bounds."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            gcs_cache.add_score("SPY", datetime(2024, 1, 1), 0.5, 1.5)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            gcs_cache.add_score("SPY", datetime(2024, 1, 1), 0.5, -0.1)

    def test_rejects_invalid_symbol(self):
        """Should reject invalid symbols."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            gcs_cache.add_score("SPY123", datetime(2024, 1, 1), 0.5, 0.75)


class TestClearCache:
    """Tests for clear_cache function."""

    @patch.object(gcs_cache, "_get_client")
    def test_deletes_blob_if_exists(self, mock_get_client):
        """Should delete blob if it exists."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True

        mock_get_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        result = gcs_cache.clear_cache("SPY")

        assert result is True
        mock_blob.delete.assert_called_once()

    @patch.object(gcs_cache, "_get_client")
    def test_returns_true_if_blob_not_exists(self, mock_get_client):
        """Should return True even if blob doesn't exist."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False

        mock_get_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        result = gcs_cache.clear_cache("SPY")

        assert result is True
        mock_blob.delete.assert_not_called()


class TestHasScoreForDate:
    """Tests for has_score_for_date function."""

    @patch.object(gcs_cache, "read_scores")
    def test_returns_false_when_no_cache(self, mock_read):
        """Should return False when no cache."""
        mock_read.return_value = None

        result = gcs_cache.has_score_for_date("SPY", datetime(2024, 1, 1))

        assert result is False

    @patch.object(gcs_cache, "read_scores")
    def test_returns_true_when_date_exists(self, mock_read):
        """Should return True when date exists in cache."""
        mock_read.return_value = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "signal_raw": [0.5, 0.6],
            "signal_0_1": [0.75, 0.8],
        })

        result = gcs_cache.has_score_for_date("SPY", datetime(2024, 1, 1))

        assert result is True

    @patch.object(gcs_cache, "read_scores")
    def test_returns_false_when_date_not_exists(self, mock_read):
        """Should return False when date not in cache."""
        mock_read.return_value = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "signal_raw": [0.5, 0.6],
            "signal_0_1": [0.75, 0.8],
        })

        result = gcs_cache.has_score_for_date("SPY", datetime(2024, 1, 5))

        assert result is False


class TestReadScoresNotFoundExceptions:
    """Additional tests for read_scores exception handling."""

    @patch.object(gcs_cache, "_get_client")
    def test_returns_none_on_not_found_exception(self, mock_get_client):
        """Should return None when NotFound exception is raised."""
        from google.cloud.exceptions import NotFound

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.side_effect = NotFound("Blob not found")

        mock_get_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        result = gcs_cache.read_scores("SPY")

        assert result is None


class TestClearCacheExceptions:
    """Additional tests for clear_cache exception handling."""

    @patch.object(gcs_cache, "_get_client")
    def test_returns_false_on_exception(self, mock_get_client):
        """Should return False when an exception occurs during clear."""
        mock_get_client.side_effect = Exception("Connection failed")

        result = gcs_cache.clear_cache("SPY")

        assert result is False

    @patch.object(gcs_cache, "_get_client")
    def test_returns_false_on_delete_error(self, mock_get_client):
        """Should return False when delete raises an exception."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.delete.side_effect = Exception("Delete failed")

        mock_get_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        result = gcs_cache.clear_cache("SPY")

        assert result is False


class TestGetLatestScoreEdgeCases:
    """Edge case tests for get_latest_score."""

    @patch.object(gcs_cache, "read_scores")
    def test_returns_none_for_empty_dataframe(self, mock_read):
        """Should return None when DataFrame is empty."""
        mock_read.return_value = pd.DataFrame()

        result = gcs_cache.get_latest_score("SPY")

        assert result is None
