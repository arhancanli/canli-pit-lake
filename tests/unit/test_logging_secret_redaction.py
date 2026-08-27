"""Regression: credentials must NEVER reach a log file or the console.

httpx logs every request at INFO including the full URL with its query string. The
corporate-actions ingest pages a keyed Polygon REST endpoint once per instrument, so a single
full pass over ~18k instruments wrote the LIVE api key to disk ~1.5 million times (found
2026-08-02; the log dir is gitignored so it never left the machine, but the key was rotated
and the leak is fixed at the source).

The fix redacts in the SHARED processor chain, which is what makes it robust: it covers both
native structlog events AND foreign stdlib records (httpx, boto, urllib3) through
``foreign_pre_chain``, so no call site has to remember to be careful. These tests pin the
RUNNING path — they drive the real ``setup_logging`` and assert on the bytes actually written.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from alphaforge.core.logging import LOG_FILE_NAME, get_logger, setup_logging

_KEY = "V678RGw3lvpxfhBMzH01lvClpog9Am_p"  # shaped like a real key; not a live credential
_URL = f"https://api.polygon.io/v3/reference/splits?ticker=AAPL&apiKey={_KEY}"


def _read_log(log_dir: Path) -> str:
    return (log_dir / LOG_FILE_NAME).read_text(encoding="utf-8")


def test_native_event_is_redacted(tmp_path: Path) -> None:
    setup_logging(level="INFO", json_file=True, log_dir=tmp_path)
    get_logger(__name__).info("http_request", url=_URL)
    for h in logging.getLogger().handlers:
        h.flush()
    body = _read_log(tmp_path)
    assert _KEY not in body, "the live key reached the log file"
    assert "apiKey=<REDACTED>" in body


def test_foreign_stdlib_record_is_redacted(tmp_path: Path) -> None:
    """The real leak path: httpx logs through stdlib, not structlog."""
    setup_logging(level="INFO", json_file=True, log_dir=tmp_path)
    logging.getLogger("httpx").info('HTTP Request: GET %s "HTTP/1.1 200 OK"', _URL)
    for h in logging.getLogger().handlers:
        h.flush()
    body = _read_log(tmp_path)
    assert _KEY not in body, "a foreign (httpx) record leaked the key — foreign_pre_chain gap"
    assert "<REDACTED>" in body


def test_other_secret_param_names_are_covered(tmp_path: Path) -> None:
    setup_logging(level="INFO", json_file=True, log_dir=tmp_path)
    log = get_logger(__name__)
    for param in ("api_key", "token", "access_token", "signature", "secret"):
        log.info("call", url=f"https://x.test/a?{param}={_KEY}")
    for h in logging.getLogger().handlers:
        h.flush()
    body = _read_log(tmp_path)
    assert _KEY not in body


def test_non_secret_content_is_untouched(tmp_path: Path) -> None:
    """Redaction must not corrupt ordinary logging (it runs on every event)."""
    setup_logging(level="INFO", json_file=True, log_dir=tmp_path)
    get_logger(__name__).info("cycle", instrument_id="XUSE:CASH:AAPLUSD", qty=100, note="a=b")
    for h in logging.getLogger().handlers:
        h.flush()
    rec = json.loads([ln for ln in _read_log(tmp_path).splitlines() if ln.strip()][-1])
    assert rec["instrument_id"] == "XUSE:CASH:AAPLUSD"
    assert rec["qty"] == 100
    assert rec["note"] == "a=b"  # short value, not secret-shaped -> untouched
