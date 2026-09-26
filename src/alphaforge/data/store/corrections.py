"""Receipted corrections for stored splits that contradict the raw prices.

Every lake stores the vendor split factor, new shares per old (``data/schemas.py``), and the
adjusted-close kernel divides pre-ex bars by it. A stored row that does not follow that
convention writes a fake move of the full ratio into every adjusted series crossing it. The raw
ex-date move is the evidence: a factor ``r`` prints about ``-log(r)``. This module judges each
split against the raw bars (:func:`classify_split`), records the decisions in one content-hashed
file beside the lake (``<lake>/_corrections/corporate_actions.json``), and applies them
(:func:`apply_split_corrections`) inside :meth:`PITDataReader.corporate_actions`, so no consumer
can read an uncorrected row by forgetting a step.

A correction matches only the exact stored row it was decided on (instrument, ex-date, stored
ratio). If the vendor later restates the row, the correction stops matching instead of flipping
a fixed value back. ``available_at`` is never changed: the point-in-time gate stays the vendor's
declaration date.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow as pa

CORRECTIONS_RELPATH: Final = Path("_corrections") / "corporate_actions.json"
SCHEMA: Final = "canli.corporate-action-corrections.v1"

#: A split this small (|log r| < 0.10) cannot be told apart from an ordinary daily move.
MIN_LOG_RATIO: Final = 0.10
#: The ex-date move matches a convention when it is within 25% of log(r), plus 0.05 for the
#: day's own move. The same tolerance as scripts/audit_split_adjustment_direction.py.
REL_TOL: Final = 0.25
ABS_TOL: Final = 0.05
#: Sessions either side of the stored ex-date searched for the move a misdated split made.
WINDOW: Final = 5
#: Every session in the window moving less than this, and the split is a phantom.
PHANTOM_MAX_MOVE: Final = 0.10

ACTIONS: Final = ("invert", "redate", "invert_redate", "void")


@dataclass(frozen=True)
class SplitCorrection:
    instrument_id: str
    stored_ex_ms: int
    stored_ratio: float
    action: str
    ratio: float
    ex_ms: int
    verdict: str


def classify_split(
    closes: pd.Series,
    ex_ms: int,
    ratio: float,
    *,
    other_splits: Sequence[tuple[int, float]] = (),
) -> dict[str, Any]:
    """Judge one stored split against the instrument's raw daily closes.

    ``closes`` is indexed by session open (epoch ms), ascending. The raw move on a session is
    log(close / previous close), the same measure the direction audit uses. ``other_splits``
    are the instrument's other stored splits as ``(ex_ms, ratio)``: a misdated row whose move
    belongs to a split already stored on that session is a DUPLICATE and is voided, because
    moving it there would apply the same split twice.
    """
    base: dict[str, Any] = {"action": None, "ratio": ratio, "ex_ms": ex_ms}
    if not math.isfinite(ratio) or ratio <= 0.0:
        return {**base, "verdict": "INVALID"}
    log_r = math.log(ratio)
    if abs(log_r) < MIN_LOG_RATIO:
        return {**base, "verdict": "SMALL"}
    series = closes.dropna().sort_index()
    index = [int(v) for v in series.index]
    values = [float(v) for v in series.to_numpy()]
    pos = next((i for i, ts in enumerate(index) if ts >= ex_ms), None)
    if pos is None or pos == 0:
        return {**base, "verdict": "UNDETERMINED"}

    def move(i: int) -> float:
        return math.log(values[i] / values[i - 1])

    tol = REL_TOL * abs(log_r) + ABS_TOL
    on_date = move(pos)
    evidence = {"ex_date_move": on_date}
    if abs(on_date + log_r) <= tol:
        return {**base, "verdict": "VENDOR_CONVENTION", **evidence}
    if abs(on_date - log_r) <= tol:
        return {
            **base,
            "verdict": "RECIPROCAL",
            "action": "invert",
            "ratio": 1.0 / ratio,
            **evidence,
        }
    near = range(max(1, pos - WINDOW), min(len(values), pos + WINDOW + 1))
    vendor = [i for i in near if i != pos and abs(move(i) + log_r) <= tol]
    reciprocal = [i for i in near if i != pos and abs(move(i) - log_r) <= tol]

    def stored_on(session: int, factor: float) -> bool:
        return any(
            ts == session and math.isclose(r, factor, rel_tol=1e-6) for ts, r in other_splits
        )

    target = vendor[0] if len(vendor) == 1 and not reciprocal else None
    target_recip = reciprocal[0] if len(reciprocal) == 1 and not vendor else None
    if (target is not None and stored_on(index[target], ratio)) or (
        target_recip is not None and stored_on(index[target_recip], 1.0 / ratio)
    ):
        return {**base, "verdict": "DUPLICATE", "action": "void", **evidence}
    if len(vendor) == 1 and not reciprocal:
        return {
            **base,
            "verdict": "MISDATED",
            "action": "redate",
            "ex_ms": index[vendor[0]],
            "matched_move": move(vendor[0]),
            **evidence,
        }
    if len(reciprocal) == 1 and not vendor:
        return {
            **base,
            "verdict": "RECIPROCAL_MISDATED",
            "action": "invert_redate",
            "ratio": 1.0 / ratio,
            "ex_ms": index[reciprocal[0]],
            "matched_move": move(reciprocal[0]),
            **evidence,
        }
    largest = max(abs(move(i)) for i in near)
    if largest < PHANTOM_MAX_MOVE:
        return {
            **base,
            "verdict": "PHANTOM",
            "action": "void",
            "largest_move_in_window": largest,
            **evidence,
        }
    return {**base, "verdict": "UNRESOLVED", "largest_move_in_window": largest, **evidence}


def _content_hash(body: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in body.items() if k != "content_hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def write_split_corrections(
    lake_root: Path, corrections: Sequence[SplitCorrection], *, evidence: Mapping[str, Any]
) -> Path:
    """Write the lake's correction file atomically; returns its path."""
    for c in corrections:
        if c.action not in ACTIONS or not (math.isfinite(c.ratio) and c.ratio > 0):
            raise ValueError(f"invalid correction {c}")
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "evidence": dict(evidence),
        "corrections": [asdict(c) for c in corrections],
    }
    body["content_hash"] = _content_hash(body)
    path = Path(lake_root) / CORRECTIONS_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".corrections-", suffix=".json")
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(body, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def load_split_corrections(lake_root: Path) -> tuple[tuple[SplitCorrection, ...], str | None]:
    """The lake's corrections and the file's content hash; ``((), None)`` without a file.

    A file whose recorded hash does not match its content raises: a hand edit must not change
    what every consumer reads without leaving a trace.
    """
    path = Path(lake_root) / CORRECTIONS_RELPATH
    if not path.is_file():
        return (), None
    body = json.loads(path.read_text())
    if body.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unexpected schema {body.get('schema')!r}")
    recorded = body.get("content_hash")
    if recorded != _content_hash(body):
        raise ValueError(f"{path}: content hash does not match the file")
    corrections = tuple(SplitCorrection(**row) for row in body["corrections"])
    return corrections, str(recorded)


def _epoch_ms(column: pa.ChunkedArray) -> list[int]:
    return [int(v) for v in column.cast(pa.int64()).to_pylist()]


def apply_split_corrections(table: pa.Table, corrections: Sequence[SplitCorrection]) -> pa.Table:
    """Apply ``corrections`` to a corporate-actions table; rows they do not name pass through."""
    if not corrections or table.num_rows == 0:
        return table
    by_key = {(c.instrument_id, c.stored_ex_ms): c for c in corrections}
    ids = table.column("instrument_id").to_pylist()
    types = table.column("action_type").to_pylist()
    ex = _epoch_ms(table.column("ex_date"))
    ratios = table.column("ratio").to_pylist()
    keep: list[bool] = []
    new_ex = list(ex)
    new_ratio = list(ratios)
    for i in range(table.num_rows):
        c = by_key.get((ids[i], ex[i])) if types[i] == "split" else None
        if (
            c is None
            or ratios[i] is None
            or not math.isclose(float(ratios[i]), c.stored_ratio, rel_tol=1e-9)
        ):
            keep.append(True)
            continue
        if c.action == "void":
            keep.append(False)
            continue
        keep.append(True)
        new_ratio[i] = c.ratio
        new_ex[i] = c.ex_ms
    ex_type = table.schema.field("ex_date").type
    out = table.set_column(
        table.schema.get_field_index("ex_date"),
        table.schema.field("ex_date"),
        pa.array(new_ex, type=pa.int64()).cast(ex_type),
    ).set_column(
        table.schema.get_field_index("ratio"),
        table.schema.field("ratio"),
        pa.array(new_ratio, type=table.schema.field("ratio").type),
    )
    return out.filter(pa.array(keep))
