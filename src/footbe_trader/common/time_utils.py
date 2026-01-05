"""Time utilities for consistent timestamp handling."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Get current UTC datetime.

    Returns:
        Current datetime in UTC timezone.
    """
    return datetime.now(UTC)


def to_timestamp(dt: datetime) -> int:
    """Convert datetime to Unix timestamp (seconds).

    Args:
        dt: Datetime to convert.

    Returns:
        Unix timestamp in seconds.
    """
    return int(dt.timestamp())


def from_timestamp(ts: int) -> datetime:
    """Convert Unix timestamp to UTC datetime.

    Args:
        ts: Unix timestamp in seconds.

    Returns:
        Datetime in UTC timezone.
    """
    return datetime.fromtimestamp(ts, tz=UTC)


def format_iso(dt: datetime) -> str:
    """Format datetime as ISO 8601 string.

    Args:
        dt: Datetime to format.

    Returns:
        ISO 8601 formatted string.
    """
    return dt.isoformat()


def parse_iso(s: str) -> datetime:
    """Parse ISO 8601 string to datetime.

    Args:
        s: ISO 8601 formatted string.

    Returns:
        Parsed datetime.
    """
    return datetime.fromisoformat(s)
