"""
Exceptions for the daily scorer module.
"""


class DailyScorerError(Exception):
    """Base exception for daily scorer errors."""

    pass


class DownloadError(DailyScorerError):
    """Error downloading stock data."""

    pass


class CacheError(DailyScorerError):
    """Error with score cache operations."""

    pass


class InsufficientDataError(DailyScorerError):
    """Not enough data to compute indicators/scores."""

    pass


class ConfigurationError(DailyScorerError):
    """Invalid configuration."""

    pass
