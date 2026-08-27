"""Rotation + retention tests for alphaforge.core.logging (offline, tmp_path only).

Covers the SIZE+TIME rollover policy and the strict age-based prune helper added
in Phase 8 (buildabilityCritique.md section 4: structlog JSONL is the only
unbounded growth; rotation + ~30-day retention are mandatory). Nothing here
touches the real network or ./var -- every handler writes under ``tmp_path``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pytest
import structlog

from alphaforge.core.logging import (
    LOG_FILE_NAME,
    MAX_BYTES_DEFAULT,
    get_logger,
    prune_logs,
    setup_logging,
)

_AF_HANDLER_NAMES = ("alphaforge.console", "alphaforge.file")


@pytest.fixture(autouse=True)
def _clean_logging_state() -> Iterator[None]:
    """Strip AlphaForge handlers and reset structlog after each test so global
    logging state never leaks between tests (mirrors test_logging.py)."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() in _AF_HANDLER_NAMES:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(logging.WARNING)
    structlog.reset_defaults()


def _file_handler() -> TimedRotatingFileHandler:
    handlers = [
        h
        for h in logging.getLogger().handlers
        if h.get_name() == "alphaforge.file" and isinstance(h, TimedRotatingFileHandler)
    ]
    assert len(handlers) == 1
    return handlers[0]


def _rolled_files(log_dir: Path) -> list[Path]:
    """Rolled siblings (``alphaforge.jsonl.*``), excluding the live file."""
    live = log_dir / LOG_FILE_NAME
    return sorted(p for p in log_dir.glob(f"{LOG_FILE_NAME}.*") if p != live)


class TestSizeRollover:
    def test_writes_jsonl_and_default_is_time_based(self, tmp_path: Path) -> None:
        """A normal setup still writes parseable JSONL and keeps the daily/30d contract."""
        setup_logging(tmp_path)
        get_logger("alphaforge.test").info("rotation_smoke", k="v")
        text = (tmp_path / LOG_FILE_NAME).read_text(encoding="utf-8")
        rec = json.loads(text.splitlines()[0])
        assert rec["event"] == "rotation_smoke"
        assert rec["k"] == "v"
        fh = _file_handler()
        assert fh.when.upper() == "MIDNIGHT"
        assert fh.utc is True
        assert fh.backupCount == 30

    def test_default_max_bytes_is_64_mib(self) -> None:
        assert MAX_BYTES_DEFAULT == 64 * 1024 * 1024

    def test_rollover_triggers_at_size_threshold(self, tmp_path: Path) -> None:
        """Enough volume past a tiny cap produces rolled files within a single day."""
        setup_logging(tmp_path, max_bytes=2048)  # tiny cap forces intra-day rolls
        log = get_logger("alphaforge.test")
        payload = "x" * 256
        for i in range(200):
            log.info("burst", i=i, blob=payload)
        rolled = _rolled_files(tmp_path)
        assert rolled, "size-bounded rotation produced no rolled files"
        # The live file stays under the cap after the latest roll.
        assert (tmp_path / LOG_FILE_NAME).stat().st_size <= 2048

    def test_max_bytes_zero_disables_size_rollover(self, tmp_path: Path) -> None:
        """With the size trigger off, a burst stays in ONE file (pure daily policy)."""
        setup_logging(tmp_path, max_bytes=0)
        log = get_logger("alphaforge.test")
        for i in range(200):
            log.info("burst", i=i, blob="y" * 256)
        assert _rolled_files(tmp_path) == []
        assert (tmp_path / LOG_FILE_NAME).stat().st_size > 2048


class TestPruneLogs:
    def test_prunes_files_older_than_keep_days(self, tmp_path: Path) -> None:
        """Old rolled files are deleted; the live file and recent rolls survive."""
        log_dir = tmp_path
        live = log_dir / LOG_FILE_NAME
        live.write_text("{}\n", encoding="utf-8")
        old = log_dir / f"{LOG_FILE_NAME}.2026-01-01"
        recent = log_dir / f"{LOG_FILE_NAME}.2026-06-13"
        old.write_text("old\n", encoding="utf-8")
        recent.write_text("recent\n", encoding="utf-8")

        now = time.time()
        # Age the old file 40 days back, the recent one 1 day back.
        import os

        os.utime(old, (now - 40 * 86_400, now - 40 * 86_400))
        os.utime(recent, (now - 1 * 86_400, now - 1 * 86_400))

        removed = prune_logs(log_dir, keep_days=30, now=now)
        assert removed == 1
        assert not old.exists()
        assert recent.exists()
        assert live.exists()

    def test_never_prunes_live_file(self, tmp_path: Path) -> None:
        """Even with keep_days=0 the live file is untouched; rolled files all go."""
        live = tmp_path / LOG_FILE_NAME
        live.write_text("{}\n", encoding="utf-8")
        rolled = tmp_path / f"{LOG_FILE_NAME}.2026-01-01"
        rolled.write_text("x\n", encoding="utf-8")
        import os

        past = time.time() - 5 * 86_400
        os.utime(rolled, (past, past))

        removed = prune_logs(tmp_path, keep_days=0, now=time.time())
        assert removed == 1
        assert live.exists()
        assert not rolled.exists()

    def test_missing_dir_returns_zero(self, tmp_path: Path) -> None:
        assert prune_logs(tmp_path / "nope", keep_days=30, now=time.time()) == 0

    def test_negative_keep_days_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="keep_days must be >= 0"):
            prune_logs(tmp_path, keep_days=-1)

    def test_recent_rolled_file_kept(self, tmp_path: Path) -> None:
        """A rolled file inside the window is retained."""
        rolled = tmp_path / f"{LOG_FILE_NAME}.2026-06-10"
        rolled.write_text("x\n", encoding="utf-8")
        now = time.time()
        import os

        os.utime(rolled, (now - 2 * 86_400, now - 2 * 86_400))
        assert prune_logs(tmp_path, keep_days=30, now=now) == 0
        assert rolled.exists()
