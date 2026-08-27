"""Structured logging for AlphaForge (structlog over stdlib logging).

Two sinks on the root logger, so both structlog-native and foreign stdlib
records (ccxt, urllib3, ...) flow through identical processors:

- console: human-readable dev renderer on stderr
- JSONL file: ``<log_dir>/alphaforge.jsonl``, rotated by a SIZE+TIME policy with
  30 rotated files retained.

Rotation policy (buildabilityCritique.md section 4: structlog JSONL is the only
unbounded growth in the system, so rotation + ~30-day retention are mandatory
from Phase 1 and re-affirmed by the Phase-8 soak gate):

- TIME: the file rolls over at every UTC midnight (one file per calendar day),
  which is what gives the ~30-day retention window: ``backupCount`` = 30 rolled
  files plus the live file is roughly a 30-day trailing window of one-per-day
  logs.
- SIZE: the SAME file ALSO rolls over the instant it crosses ``max_bytes`` within
  a day. A 24/7 paper loop logging per-cycle/per-symbol lines is bounded per day,
  but a tight crash-restart loop or a verbose ccxt failure storm can emit a burst
  that would otherwise let a SINGLE day's file grow without bound until the next
  midnight roll -- the size cap turns that unbounded intra-day growth into at most
  one extra rolled file. Default 64 MiB.

Because rotation is size-aware, ``backupCount`` alone no longer guarantees a
strict 30-day window (a high-traffic day can consume several of the 30 backup
slots). :func:`prune_logs` provides the strict time-based complement: it deletes
rolled files whose on-disk mtime is older than ``keep_days`` days, so retention is
bounded by BOTH a file count (the handler's ``backupCount``) and an age
(``prune_logs``) -- whichever bites first. The live file is never pruned.

Every line carries ``ts`` (ISO-8601 UTC), ``level``, ``logger``, ``event``
plus whatever key-values the caller bound. ``setup_logging`` is idempotent:
handlers are tagged by name and replaced, never duplicated, so re-running it
(tests, notebook re-execution) is safe.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from collections.abc import MutableMapping
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Final

import structlog
from structlog.typing import Processor

from alphaforge.core.errors import ConfigError

__all__ = [
    "LOG_FILE_NAME",
    "MAX_BYTES_DEFAULT",
    "get_logger",
    "prune_logs",
    "setup_logging",
]

LOG_FILE_NAME = "alphaforge.jsonl"
_CONSOLE_HANDLER_NAME = "alphaforge.console"
_FILE_HANDLER_NAME = "alphaforge.file"
_RETENTION_DAYS = 30
MAX_BYTES_DEFAULT = 64 * 1024 * 1024
"""Default per-file size cap (64 MiB) before an intra-day size rollover fires."""

# Shared by the structlog-native pipeline and the foreign_pre_chain so stdlib
# records get the same ts/level/logger keys. format_exc_info renders
# exceptions to a string "exception" key (identical in console and JSONL).
#: Query-string parameters whose VALUES are secrets and must never reach a log file or the
#: console. httpx logs every request at INFO including its full URL, so an ingest that pages a
#: keyed REST endpoint writes the credential once per call — a full corporate-actions pass over
#: ~18k instruments wrote the live Polygon key to disk ~1.5M times before this was added
#: (2026-08-02; logs are gitignored so it never left the machine, but the key was rotated).
#: Redaction happens in the SHARED chain so it covers BOTH native structlog events and foreign
#: stdlib records (httpx/boto/urllib3) via ``foreign_pre_chain`` — i.e. the secret is scrubbed
#: before any handler formats it, rather than relying on each call site to be careful.
_SECRET_QUERY_PARAMS: Final[tuple[str, ...]] = (
    "apiKey", "api_key", "apikey", "token", "access_token", "key", "signature", "secret",
)
_SECRET_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(" + "|".join(_SECRET_QUERY_PARAMS) + r")=[^&\s\"']{6,}"
)


def _redact_secrets(
    _logger: object,
    _name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Scrub secret-bearing query params from every string value in the event.

    Fail-open by design: logging must never raise. A value that is not a string is left alone.
    """
    for k, v in event_dict.items():
        if isinstance(v, str) and "=" in v:
            redacted = _SECRET_RE.sub(r"\1=<REDACTED>", v)
            if redacted != v:
                event_dict[k] = redacted
    return event_dict


_SHARED_PROCESSORS: tuple[Processor, ...] = (
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.UnicodeDecoder(),
    _redact_secrets,  # LAST: scrub after every other processor has populated the event
)


class _SizedTimedRotatingFileHandler(TimedRotatingFileHandler):
    """``TimedRotatingFileHandler`` that ALSO rolls over on a size threshold.

    The base handler rolls strictly at the time boundary (UTC midnight here).
    This subclass additionally rolls the instant the live file would cross
    ``max_bytes``, so a single high-traffic day cannot grow one file without
    bound until the next midnight. ``max_bytes <= 0`` disables the size check and
    the handler behaves exactly like its base class (pure time rotation).

    It remains a genuine :class:`TimedRotatingFileHandler` (``when='MIDNIGHT'``,
    ``utc=True``, ``backupCount`` honored) -- callers that introspect ``when`` /
    ``utc`` / ``backupCount`` / ``baseFilename`` see the same values; only the
    rollover TRIGGER is widened from time-only to time-or-size.
    """

    def __init__(self, *args: object, max_bytes: int, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._max_bytes = max_bytes

    def shouldRollover(self, record: logging.LogRecord) -> int:
        """Return non-zero if EITHER the time boundary OR the size cap is hit.

        Time check is delegated to the base handler (it owns the UTC-midnight
        ``rolloverAt`` schedule). The size check mirrors stdlib
        ``RotatingFileHandler``: it measures the would-be post-write file size by
        seeking to end and adding the formatted record length, so the cap is a
        ceiling that triggers BEFORE the byte that would exceed it is written.
        """
        if super().shouldRollover(record):
            return 1
        if self._max_bytes > 0 and self.stream is not None:
            msg = f"{self.format(record)}\n"
            self.stream.seek(0, os.SEEK_END)
            if self.stream.tell() + len(msg.encode(self.encoding or "utf-8")) >= self._max_bytes:
                return 1
        return 0

    def rotation_filename(self, default_name: str) -> str:
        """Disambiguate the rolled-file name so an intra-day SIZE roll never clobbers.

        The base ``TimedRotatingFileHandler`` names each rolled file by its date
        suffix (``alphaforge.jsonl.2026-06-14``). On a pure daily schedule that is
        unique, but several SIZE rollovers within the SAME day would all resolve to
        the identical date name and overwrite one another -- losing every chunk but
        the last. When the computed name already exists we append the smallest free
        ``.N`` counter (``...2026-06-14.1``, ``.2``, ...), so each intra-day chunk
        is preserved as its own file while the date-per-day retention story holds.
        """
        name = super().rotation_filename(default_name)
        if not os.path.exists(name):
            return name
        counter = 1
        while os.path.exists(f"{name}.{counter}"):
            counter += 1
        return f"{name}.{counter}"


def setup_logging(
    log_dir: Path,
    level: str = "INFO",
    json_file: bool = True,
    *,
    max_bytes: int = MAX_BYTES_DEFAULT,
) -> None:
    """Configure structlog + stdlib root handlers. Call once at process start.

    ``log_dir`` is created if missing. ``level`` is a stdlib level name
    (case-insensitive: DEBUG/INFO/WARNING/ERROR/CRITICAL) applied to the root
    logger; unknown names raise ConfigError. ``json_file=False`` skips the
    JSONL sink (console only -- e.g. throwaway CLI invocations). ``max_bytes`` is
    the per-file size cap that triggers an intra-day size rollover on top of the
    UTC-midnight time rollover (see the module docstring); ``max_bytes <= 0``
    disables the size trigger, leaving pure daily rotation.

    Idempotent: previously installed AlphaForge handlers (identified by
    handler name) are closed and replaced; foreign handlers are untouched.
    """
    level_no = logging.getLevelNamesMapping().get(level.upper())
    if level_no is None or level_no == logging.NOTSET:
        raise ConfigError(f"unknown log level: {level!r}")

    structlog.configure(
        processors=[*_SHARED_PROCESSORS, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,  # safe re-configuration (tests, level changes)
    )

    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() in (_CONSOLE_HANDLER_NAME, _FILE_HANDLER_NAME):
            root.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.set_name(_CONSOLE_HANDLER_NAME)
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
            ],
            foreign_pre_chain=list(_SHARED_PROCESSORS),
        )
    )
    root.addHandler(console)

    if json_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = _SizedTimedRotatingFileHandler(
            log_dir / LOG_FILE_NAME,
            when="midnight",
            utc=True,
            backupCount=_RETENTION_DAYS,
            encoding="utf-8",
            max_bytes=max_bytes,
        )
        file_handler.set_name(_FILE_HANDLER_NAME)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(sort_keys=True),
                ],
                foreign_pre_chain=list(_SHARED_PROCESSORS),
            )
        )
        root.addHandler(file_handler)

    root.setLevel(level_no)


def prune_logs(log_dir: Path, *, keep_days: int = _RETENTION_DAYS, now: float | None = None) -> int:
    """Delete ROLLED JSONL files older than ``keep_days`` days; return the count removed.

    The strict time-based complement to the handler's count-based ``backupCount``:
    because the rollover policy is size-aware, a burst day can consume several
    backup slots and shorten the wall-clock window, so this enforces a hard AGE
    ceiling regardless of file count. It is safe to call from a cron/launchd
    sidecar independently of the running loop -- it only ``unlink``s rolled files.

    Targets ``alphaforge.jsonl.*`` rolled siblings (the live ``alphaforge.jsonl``
    is NEVER removed). A file is pruned when its modification time is more than
    ``keep_days`` days before ``now`` (epoch seconds; ``None`` reads the wall
    clock). Missing ``log_dir`` -> 0. ``keep_days <= 0`` prunes every rolled file.

    Raises ``ValueError`` if ``keep_days < 0``.
    """
    if keep_days < 0:
        raise ValueError(f"keep_days must be >= 0, got {keep_days}")
    if not log_dir.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - keep_days * 86_400.0
    removed = 0
    live = log_dir / LOG_FILE_NAME
    for path in log_dir.glob(f"{LOG_FILE_NAME}.*"):
        if path == live or not path.is_file():
            continue
        if path.stat().st_mtime < cutoff:
            path.unlink()
            removed += 1
    return removed


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return the structlog stdlib BoundLogger for ``name`` (dotted-path
    convention: ``alphaforge.data.ingest``). Lazy: safe to call at import
    time, before :func:`setup_logging` has run -- configuration binds on first
    log call."""
    return structlog.stdlib.get_logger(name)
