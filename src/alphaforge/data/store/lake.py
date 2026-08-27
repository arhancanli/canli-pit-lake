"""Lake layout — partition paths, glob patterns, and the tmp-file protocol.

:class:`LakePaths` is the single source of truth for *where* lake files live; it
performs path arithmetic plus read-only directory listing, never data I/O. Both the
writer and the reader consume it, so the partition layout (delegated to
:func:`alphaforge.data.schemas.lake_relpath`) is defined exactly once:

    <root>/<dataset>/instrument_id=<id>/year=<YYYY>/data.parquet

Crash-safety protocol (dataDesign.md §2.2): a leaf partition is always rewritten
whole — first to a temporary file in the SAME directory (name prefix
:data:`TMP_PREFIX`, so it sits on the same filesystem), then atomically promoted with
``os.replace``. Readers therefore never observe a torn file: they only ever open
``data.parquet``, and tmp files are invisible to them by naming convention. Leftover
tmp files from a crashed writer are inert garbage, removed by
:meth:`alphaforge.data.store.writer.LakeWriter.clean_tmp`.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Final

from alphaforge.data.schemas import Dataset, lake_relpath

__all__ = [
    "TMP_PREFIX",
    "LakePaths",
]

TMP_PREFIX: Final[str] = ".tmp-"
"""Filename prefix marking an in-flight (not yet atomically promoted) partition file.

The leading dot keeps tmp files out of ``data.parquet`` globs and directory listings;
the writer creates them in the destination directory so ``os.replace`` is an
atomic same-filesystem rename (POSIX guarantee).
"""

_LEAF_FILENAME: Final[str] = "data.parquet"
_YEAR_DIR_RE: Final[re.Pattern[str]] = re.compile(r"^year=(\d{4})$")


class LakePaths:
    """Pure path arithmetic plus directory listing for the Parquet lake.

    All paths are derived from a single ``root`` directory (typically
    ``settings.paths.lake_dir``). The class never creates, writes, or deletes
    files — the writer owns mutation, DuckDB (via the reader) owns querying.
    """

    def __init__(self, root: Path) -> None:
        """``root`` is the lake root directory; it need not exist yet (listing
        methods return empty results for a missing root)."""
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The lake root directory (as given; not required to exist)."""
        return self._root

    def partition_path(self, dataset: Dataset, instrument_id: str, year: int) -> Path:
        """Absolute path of the single leaf file for ``(dataset, instrument_id, year)``.

        Layout is delegated to :func:`alphaforge.data.schemas.lake_relpath`, which
        also validates ``instrument_id`` (no ``/``, ``=``, NUL) and ``year``
        (4-digit), raising :class:`ValueError` on violation.
        """
        return self._root / lake_relpath(dataset, instrument_id, year)

    def glob(self, dataset: Dataset) -> str:
        """DuckDB-compatible glob matching every leaf file of ``dataset``.

        Matches only ``data.parquet`` leaves under hive-style
        ``instrument_id=*/year=*`` directories — in-flight :data:`TMP_PREFIX`
        files can never match. Suitable for ``read_parquet('<glob>',
        hive_partitioning=true)`` ad-hoc queries; the PIT reader prefers explicit
        file lists for partition pruning.
        """
        return str(self._root / dataset.value / "instrument_id=*" / "year=*" / _LEAF_FILENAME)

    def instrument_ids(self, dataset: Dataset) -> list[str]:
        """Sorted instrument ids physically present in the lake for ``dataset``.

        Derived from the hive partition directories (``instrument_id=<id>``) — the
        ids that are actually stored, not what some registry believes should be
        stored. Returns ``[]`` for a missing dataset directory (or lake root).
        Listing only; no file is opened.
        """
        dataset_dir = self._root / dataset.value
        if not dataset_dir.is_dir():
            return []
        prefix = "instrument_id="
        return sorted(
            entry.name.removeprefix(prefix)
            for entry in dataset_dir.iterdir()
            if entry.is_dir() and entry.name.startswith(prefix)
        )

    def years_for(self, dataset: Dataset, instrument_id: str) -> list[int]:
        """Sorted years for which a leaf file exists for ``(dataset, instrument_id)``.

        Only directories matching ``year=NNNN`` that actually contain
        ``data.parquet`` count; tmp files and stray entries are ignored. Returns
        ``[]`` if the instrument (or the whole lake) has no data yet. Raises
        :class:`ValueError` on a malformed ``instrument_id`` (same validation as
        :meth:`partition_path`).
        """
        instrument_dir = self.partition_path(dataset, instrument_id, 1970).parent.parent
        if not instrument_dir.is_dir():
            return []
        years: list[int] = []
        for entry in instrument_dir.iterdir():
            match = _YEAR_DIR_RE.match(entry.name)
            if match is None or not entry.is_dir():
                continue
            if (entry / _LEAF_FILENAME).is_file():
                years.append(int(match.group(1)))
        return sorted(years)

    def partition_paths(
        self,
        dataset: Dataset,
        instrument_id: str,
        *,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> list[Path]:
        """Existing leaf files for ``(dataset, instrument_id)``, year-ascending.

        ``year_min``/``year_max`` are inclusive bounds used for partition pruning
        (``None`` = unbounded). This is the reader's file-listing primitive: it
        returns only files that exist, so missing partitions simply contribute
        nothing.
        """
        return [
            self.partition_path(dataset, instrument_id, year)
            for year in self.years_for(dataset, instrument_id)
            if (year_min is None or year >= year_min) and (year_max is None or year <= year_max)
        ]

    def tmp_path(self, target: Path) -> Path:
        """A fresh, collision-free tmp path in the SAME directory as ``target``.

        Same-directory placement guarantees ``os.replace(tmp, target)`` is an
        atomic rename on POSIX (no cross-filesystem copy); the uuid component
        makes concurrent or repeated writes collision-free.
        """
        return target.parent / f"{TMP_PREFIX}{uuid.uuid4().hex}-{target.name}"

    @staticmethod
    def is_tmp(path: Path) -> bool:
        """True iff ``path`` follows the in-flight tmp naming convention."""
        return path.name.startswith(TMP_PREFIX)

    def tmp_files(self) -> list[Path]:
        """All leftover in-flight tmp files anywhere under the lake root, sorted.

        Non-empty output indicates a writer crashed (or failed) between writing a
        tmp file and promoting it; the files are inert (readers never open them)
        and are deleted by ``LakeWriter.clean_tmp()``.
        """
        if not self._root.is_dir():
            return []
        return sorted(p for p in self._root.rglob(f"{TMP_PREFIX}*") if p.is_file())
