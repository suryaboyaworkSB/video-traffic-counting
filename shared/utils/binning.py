"""
Time-bin helpers for traffic count aggregation.

Most traffic studies report counts in 15-minute (or 5/15/60-minute) bins.
"""

from datetime import datetime, timedelta


def floor_to_bin(ts: datetime, bin_minutes: int = 15) -> datetime:
    """Floor a timestamp to the start of its time bin."""
    minute = (ts.minute // bin_minutes) * bin_minutes
    return ts.replace(minute=minute, second=0, microsecond=0)


def bin_label(ts: datetime, bin_minutes: int = 15) -> str:
    """Human-friendly label like '08:00-08:15'."""
    start = floor_to_bin(ts, bin_minutes)
    end = start + timedelta(minutes=bin_minutes)
    return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
