"""Network integration test for ``PolygonFlatFilesSource`` against live Polygon S3 flat files.

This test hits the real Polygon flat-files S3 bucket (``flatfiles`` at
``https://files.polygon.io``) via the boto3 ``_Boto3FlatFilesClient`` and is therefore:

* gated on the real ``POLYGON_S3_*`` credentials — skipped entirely when they are absent
  (the operator has not sourced them), so CI stays green without credentials;
* marked ``@pytest.mark.network`` — also deselected by the default ``addopts``
  (``-m 'not network'``); run explicitly with ``uv run pytest -m network`` once the keys
  are set;
* sized to a single real recent day-aggs object;
* asserting *shape and sanity*, never exact values — live data drifts.

It is the flat-files API-reachability probe: the moment the keys land, it flips from
skipped to live and validates that the synthetic-fixture contracts in
``tests/unit/test_polygon_flatfiles.py`` match Polygon's real S3 payload
(EQUITIES_INGEST.md §4.5).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from alphaforge.core.time import Timeframe, now_ms
from alphaforge.data.schemas import Dataset, validate_table
from alphaforge.data.sources.polygon_flatfiles import PolygonFlatFilesSource

pytestmark = pytest.mark.network

_HAVE_CREDS = all(
    os.environ.get(var)
    for var in (
        "POLYGON_S3_ENDPOINT",
        "POLYGON_S3_ACCESS_KEY_ID",
        "POLYGON_S3_SECRET_ACCESS_KEY",
        "POLYGON_S3_BUCKET",
    )
)
requires_creds = pytest.mark.skipif(not _HAVE_CREDS, reason="POLYGON_S3_* not set")

D1_MS = Timeframe.D1.ms


def _recent_weekday() -> datetime:
    """A weekday roughly two weeks back (very likely a real US session with a day-aggs file)."""
    d = datetime.now(tz=UTC) - timedelta(days=14)
    while d.weekday() >= 5:  # Sat=5, Sun=6 -> step back to Friday
        d -= timedelta(days=1)
    return d


@requires_creds
def test_fetch_one_real_day_shape() -> None:
    """Fetch ONE real recent day and assert shape/sanity (never exact values)."""
    src = PolygonFlatFilesSource()  # constructed from POLYGON_S3_* env
    d = _recent_weekday().date()
    tbl = src.fetch_day(d, now=now_ms())

    validate_table(tbl, Dataset.OHLCV_1D)
    # A full US session has thousands of names — assert it is a real panel, not a stub.
    assert tbl.num_rows > 1_000
    # Every bar is labeled by the session's UTC day open.
    day_open = int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)
    ts_opens = set(tbl.column("ts_open").cast(pa.int64()).to_pylist())
    assert ts_opens == {day_open}
    # Sanity: positive prices and volumes.
    assert min(tbl.column("close").to_pylist()) > 0.0
    assert min(tbl.column("volume").to_pylist()) >= 0.0


@requires_creds
def test_available_sessions_lists_recent_days() -> None:
    """available_sessions returns real session dates within a recent two-week span."""
    src = PolygonFlatFilesSource()
    end = datetime.now(tz=UTC) - timedelta(days=2)
    start = end - timedelta(days=14)
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime(end.year, end.month, end.day, tzinfo=UTC).timestamp() * 1000)
    sessions = src.available_sessions(start=start_ms, until=end_ms)
    # ~2 weeks -> at least a handful of trading days, all weekdays, ascending + unique.
    assert len(sessions) >= 5
    assert sessions == sorted(set(sessions))
    assert all(s.weekday() < 5 for s in sessions)
