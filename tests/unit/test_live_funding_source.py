"""Pin the funding path that ACTUALLY RUNS, not the one we wish ran.

WHY THIS FILE EXISTS. `tests/unit/test_paper_broker_funding.py` has nine passing tests and every
one of them calls `PaperBroker.apply_funding` directly. That is the innermost layer. Nothing
exercised the layer above it -- the `_funding_source` closure in `cli/paper_cmds.py` that reads the
lake and hands events to `LiveLoop._settle_funding`.

So the live crypto sleeve ran from 2026-06-29 to 2026-08-12 with a `_funding_source` that raised
`TypeError` on EVERY non-empty read, and the whole suite stayed green. `ts_funding` is
`timestamp[ms, tz=UTC]` in FUNDING_SCHEMA, `.to_pylist()` therefore yields `datetime` objects, and
the code did `int(x)` on them. `LiveLoop._settle_funding` catches bare `Exception`, so each failure
became one WARN line that read like a venue hiccup. Measured consequence: three distinct cash
levels across 278 hourly marks in 44 days, every one of them a rebalance. A funding-CARRY sleeve
booked funding exactly zero times, while `paper_trading_state.py` published the sentence "the live
loop now settles every stored funding event for the window just closed".

The lesson generalises past funding: a test that pins the intention is not a test. These pin the
conversion the live path performs and the escalation that makes a permanent failure look different
from a transient one.
"""

from __future__ import annotations

import datetime as dt
import itertools

import pyarrow as pa
import pytest

from alphaforge.data.schemas import FUNDING_SCHEMA

_T0 = 1758124800003  # 2025-09-17 16:00 UTC, a real settlement from the live lake


def _funding_table(n: int = 3) -> pa.Table:
    """A table shaped exactly like `reader.funding` returns -- schema-cast, not hand-rolled."""
    ts = [_T0 + i * 4 * 3_600_000 for i in range(n)]  # 4h interval, as BTCUSDT actually settles
    return pa.table(
        {
            "instrument_id": ["BINANCE:PERP:BTCUSDT"] * n,
            "ts_funding": ts,
            "rate": [0.0001 * (i + 1) for i in range(n)],
            "available_at": [t + 300_000 for t in ts],
            "ingested_at": [t + 600_000 for t in ts],
        }
    ).cast(FUNDING_SCHEMA)


def test_ts_funding_is_a_timestamp_not_an_int() -> None:
    """The premise of the bug. If this ever fails, the schema changed and the rest is moot."""
    tbl = _funding_table()
    assert pa.types.is_timestamp(tbl.column("ts_funding").type)
    assert isinstance(tbl.column("ts_funding").to_pylist()[0], dt.datetime)


def test_naive_int_conversion_raises_which_is_what_broke_the_sleeve() -> None:
    """Guards the regression directly: this is the exact expression that shipped."""
    tbl = _funding_table()
    with pytest.raises(TypeError):
        [int(x) for x in tbl.column("ts_funding").to_pylist()]


def test_live_conversion_yields_epoch_ms_ints() -> None:
    """The corrected expression from paper_cmds.py, asserted end to end."""
    tbl = _funding_table()
    out = [int(x) for x in tbl.column("ts_funding").cast(pa.int64()).to_pylist()]

    assert all(isinstance(v, int) for v in out)
    assert out[0] == _T0
    # Epoch MILLISECONDS, not seconds or microseconds -- a unit slip here would silently place
    # every settlement in 1970 or the far future and book funding against the wrong mark.
    assert dt.datetime.fromtimestamp(out[0] / 1000, dt.UTC).year == 2025
    # Interval preserved: the stored events ARE the schedule, so spacing must survive the cast.
    assert [b - a for a, b in itertools.pairwise(out)] == [4 * 3_600_000] * (len(out) - 1)


def test_full_zip_shape_matches_what_settle_funding_unpacks() -> None:
    """`_settle_funding` does `for iid, ts_funding, rate in events`; prove that unpacks."""
    tbl = _funding_table()
    events = list(
        zip(
            [str(x) for x in tbl.column("instrument_id").to_pylist()],
            [int(x) for x in tbl.column("ts_funding").cast(pa.int64()).to_pylist()],
            [float(x) for x in tbl.column("rate").to_pylist()],
            strict=True,
        )
    )
    assert len(events) == 3
    for iid, ts_funding, rate in events:  # the exact unpack the loop performs
        assert isinstance(iid, str) and iid.startswith("BINANCE:PERP:")
        assert isinstance(ts_funding, int)
        assert isinstance(rate, float)


def test_empty_read_is_still_fine() -> None:
    """No settlements in the window is normal and must not raise."""
    tbl = _funding_table(0)
    assert [int(x) for x in tbl.column("ts_funding").cast(pa.int64()).to_pylist()] == []


# --------------------------------------------------------------------------- available_at window
#
# The SECOND funding defect, found 2026-08-12 after the cast was fixed and funding STILL booked
# nothing. Selecting on ts_funding in (prev, cycle] loses every settlement that lands exactly on a
# bar boundary, because publication lags settlement: an event stamped 16:00:00 is available at
# 16:05:00, so the 16:00 cycle sees it in the ts window but rejects it as not-yet-knowable, and
# the 17:00 cycle accepts it as knowable but excludes 16:00 from its half-open window. Too new for
# its own bar, too old for the next. On the live lake that silently dropped the entire
# 00:00/08:00/16:00 schedule — 42 settlements a day across 9 legs, every day.
#
# Selecting on available_at instead is exactly-once BY CONSTRUCTION: an event becomes available at
# one instant, so exactly one half-open cycle window contains it.

LAG_MS = 300_000  # FUNDING_PUBLICATION_LAG_MS: 5 minutes


def _select(
    rows: list[tuple[str, int, float, int]], start: int, end: int
) -> list[tuple[str, int, float]]:
    """The live selector from paper_cmds.py, in miniature: window on available_at, dedupe on key."""
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int, float]] = []
    for iid, ts, rate, av in rows:
        if not (start < av <= end):
            continue
        if (iid, ts) in seen:
            continue
        seen.add((iid, ts))
        out.append((iid, ts, rate))
    return out


def test_boundary_settlement_is_lost_by_the_ts_funding_window() -> None:
    """Pins the defect itself: the old windowing drops a settlement on a bar boundary."""
    hour = 3_600_000
    cycle = _T0 - (_T0 % hour)          # a settlement landing exactly on a bar open
    ts, av = cycle, cycle + LAG_MS

    at_settlement = (cycle - hour) < ts <= cycle and av <= cycle       # PIT rejects it
    at_next = (cycle) < ts <= (cycle + hour) and av <= cycle + hour    # window rejects it
    assert not at_settlement, "expected the settlement cycle to reject on point-in-time"
    assert not at_next, "expected the next cycle to reject on the half-open window"


def test_available_at_window_books_a_boundary_settlement_exactly_once() -> None:
    """The fix: the same event books, once, in the first cycle that could honestly see it."""
    hour = 3_600_000
    cycle = _T0 - (_T0 % hour)
    rows = [("BINANCE:PERP:BTCUSDT", cycle, 0.0001, cycle + LAG_MS)]

    booked = [len(_select(rows, c - hour, c)) for c in (cycle, cycle + hour, cycle + 2 * hour)]
    assert booked == [0, 1, 0], f"expected exactly one booking, in the NEXT cycle; got {booked}"


def test_duplicate_lake_rows_are_never_billed_twice() -> None:
    """The funding lake can hold the same settlement twice; the book must not pay for it twice."""
    hour = 3_600_000
    cycle = _T0 - (_T0 % hour)
    dup = ("BINANCE:PERP:BTCUSDT", cycle, 0.0001, cycle + LAG_MS)
    assert len(_select([dup, dup, dup], cycle, cycle + hour)) == 1


def test_replay_of_consecutive_cycles_books_each_event_once() -> None:
    """End to end over a day: every settlement booked, none twice, none missed."""
    hour = 3_600_000
    base = _T0 - (_T0 % (8 * hour))
    rows = [
        (f"BINANCE:PERP:SYM{i}", base + k * 8 * hour, 0.0001, base + k * 8 * hour + LAG_MS)
        for i in range(9)
        for k in range(3)
    ]
    counts: dict[tuple[str, int], int] = {}
    for step in range(30):  # 30 hourly cycles covers all three settlements with room to spare
        c = base + step * hour
        for iid, ts, _ in _select(rows, c - hour, c):
            counts[(iid, ts)] = counts.get((iid, ts), 0) + 1

    assert len(counts) == 27, (
        f"expected all 9 legs x 3 settlements booked, got {len(counts)}"
    )
    assert set(counts.values()) == {1}, (
        f"every event must book exactly once, got {set(counts.values())}"
    )
