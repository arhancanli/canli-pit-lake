"""Tests for alphaforge.core.logging: JSONL sink correctness, idempotency,
level filtering, foreign stdlib routing, and rotation/retention settings."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

import pytest
import structlog

from alphaforge.core.errors import ConfigError
from alphaforge.core.logging import LOG_FILE_NAME, get_logger, setup_logging

_AF_HANDLER_NAMES = ("alphaforge.console", "alphaforge.file")


@pytest.fixture(autouse=True)
def _clean_logging_state() -> Iterator[None]:
    """Remove AlphaForge handlers and reset structlog after each test so the
    global logging state never leaks between tests."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() in _AF_HANDLER_NAMES:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(logging.WARNING)
    structlog.reset_defaults()


def _af_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if h.get_name() in _AF_HANDLER_NAMES]


def _read_jsonl(log_dir: Path) -> list[dict[str, Any]]:
    text = (log_dir / LOG_FILE_NAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


class TestJsonlSink:
    def test_line_written_and_parseable(self, tmp_path: Path) -> None:
        setup_logging(tmp_path)
        get_logger("alphaforge.test").info("ingest_complete", symbol="BTCUSDT", rows=720)
        lines = _read_jsonl(tmp_path)
        assert len(lines) == 1
        rec = lines[0]
        assert rec["event"] == "ingest_complete"
        assert rec["symbol"] == "BTCUSDT"
        assert rec["rows"] == 720
        assert rec["level"] == "info"
        assert rec["logger"] == "alphaforge.test"

    def test_ts_is_iso_utc(self, tmp_path: Path) -> None:
        setup_logging(tmp_path)
        get_logger("alphaforge.test").info("tick")
        rec = _read_jsonl(tmp_path)[0]
        ts = datetime.fromisoformat(rec["ts"])
        assert ts.tzinfo is not None
        assert ts.utcoffset() == UTC.utcoffset(None)

    def test_bound_context_appears(self, tmp_path: Path) -> None:
        setup_logging(tmp_path)
        log = get_logger("alphaforge.test").bind(run_id="r-001", component="ingest")
        log.info("upsert")
        rec = _read_jsonl(tmp_path)[0]
        assert rec["run_id"] == "r-001"
        assert rec["component"] == "ingest"

    def test_foreign_stdlib_logger_routed(self, tmp_path: Path) -> None:
        setup_logging(tmp_path)
        logging.getLogger("urllib3.foreign").warning("retrying %s", "GET /klines")
        recs = [r for r in _read_jsonl(tmp_path) if r["logger"] == "urllib3.foreign"]
        assert len(recs) == 1
        assert recs[0]["event"] == "retrying GET /klines"
        assert recs[0]["level"] == "warning"

    def test_json_file_false_writes_nothing(self, tmp_path: Path) -> None:
        setup_logging(tmp_path, json_file=False)
        get_logger("alphaforge.test").info("console_only")
        assert not (tmp_path / LOG_FILE_NAME).exists()
        assert len(_af_handlers()) == 1


class TestIdempotency:
    def test_no_duplicate_handlers(self, tmp_path: Path) -> None:
        setup_logging(tmp_path)
        setup_logging(tmp_path)
        assert len(_af_handlers()) == 2  # exactly one console + one file

    def test_no_duplicate_lines_after_reconfigure(self, tmp_path: Path) -> None:
        setup_logging(tmp_path)
        setup_logging(tmp_path)
        get_logger("alphaforge.test").info("once")
        assert len(_read_jsonl(tmp_path)) == 1

    def test_reconfigure_can_change_level(self, tmp_path: Path) -> None:
        setup_logging(tmp_path, level="WARNING")
        setup_logging(tmp_path, level="DEBUG")
        get_logger("alphaforge.test").debug("now_visible")
        assert len(_read_jsonl(tmp_path)) == 1


class TestLevels:
    def test_level_filters_below_threshold(self, tmp_path: Path) -> None:
        setup_logging(tmp_path, level="WARNING")
        log = get_logger("alphaforge.test")
        log.info("suppressed")
        log.warning("emitted")
        lines = _read_jsonl(tmp_path)
        assert [r["event"] for r in lines] == ["emitted"]

    def test_level_case_insensitive(self, tmp_path: Path) -> None:
        setup_logging(tmp_path, level="debug")
        get_logger("alphaforge.test").debug("visible")
        assert len(_read_jsonl(tmp_path)) == 1

    def test_invalid_level_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown log level"):
            setup_logging(tmp_path, level="LOUD")


class TestRotation:
    def test_daily_utc_rotation_with_30_day_retention(self, tmp_path: Path) -> None:
        setup_logging(tmp_path)
        file_handlers = [h for h in _af_handlers() if isinstance(h, TimedRotatingFileHandler)]
        assert len(file_handlers) == 1
        fh = file_handlers[0]
        assert fh.when.upper() == "MIDNIGHT"
        assert fh.utc is True
        assert fh.backupCount == 30
        assert Path(fh.baseFilename) == tmp_path / LOG_FILE_NAME

    def test_log_dir_created(self, tmp_path: Path) -> None:
        target = tmp_path / "var" / "log"
        setup_logging(target)
        get_logger("alphaforge.test").info("created")
        assert (target / LOG_FILE_NAME).is_file()
