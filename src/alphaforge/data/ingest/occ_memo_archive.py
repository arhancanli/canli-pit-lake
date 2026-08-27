"""Immutable, content-addressed acquisition for official OCC information memos.

The archive is deliberately upstream of option-adjustment extraction.  It proves which exact PDF
bytes were observable at a local timestamp; it does not infer publication time or economic terms
from PDF prose.  Only direct HTTPS responses from OCC's canonical memo endpoint are accepted.

Each PDF is stored once under its SHA-256 digest.  Every observation gets an immutable manifest,
so a source changing later creates a new history entry rather than silently rewriting evidence.
Existing blobs and manifests are re-hashed before reuse, and all writes use a same-directory
temporary file plus an atomic hard-link promotion (create-if-absent, never overwrite).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, Self, cast
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from alphaforge.core.time import Ms

__all__ = [
    "HTTPMemoResponse",
    "OCCMemoAcquisitionError",
    "OCCMemoArchive",
    "OCCMemoArchiveConflictError",
    "OCCMemoArchiveIntegrityError",
    "OCCMemoSnapshot",
    "OCCMemoTransport",
    "UrllibOCCMemoTransport",
    "occ_memo_url",
]

_SCHEMA: Final[str] = "alphaforge.occ-memo-snapshot.v1"
_MEMO_RE: Final[re.Pattern[str]] = re.compile(r"[1-9][0-9]{0,7}")
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_OCC_HOST: Final[str] = "infomemo.theocc.com"
_OCC_PATH: Final[str] = "/infomemos"
_PDF_CONTENT_TYPE: Final[str] = "application/pdf"
_DEFAULT_MAX_BYTES: Final[int] = 20 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0


class OCCMemoAcquisitionError(ValueError):
    """The remote response cannot be admitted as an official complete OCC PDF."""


class OCCMemoArchiveIntegrityError(RuntimeError):
    """Persisted archive bytes or metadata no longer match their content address."""


class OCCMemoArchiveConflictError(RuntimeError):
    """An immutable path or one observation timestamp conflicts with different content."""


def _validate_memo_number(memo_number: str) -> None:
    if _MEMO_RE.fullmatch(memo_number) is None:
        raise ValueError("memo_number must be 1-8 digits, positive, without leading zeroes")


def occ_memo_url(memo_number: str) -> str:
    """Return the only remote URL accepted for one OCC information memo."""
    _validate_memo_number(memo_number)
    return f"https://{_OCC_HOST}{_OCC_PATH}?number={memo_number}"


def _validate_occ_url(url: str, memo_number: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _OCC_HOST
        or parsed.path != _OCC_PATH
        or parsed.fragment
        or parse_qs(parsed.query) != {"number": [memo_number]}
    ):
        raise OCCMemoAcquisitionError(
            f"response URL {url!r} is not the canonical OCC endpoint for memo {memo_number}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class HTTPMemoResponse:
    """Bounded HTTP result returned by an acquisition transport."""

    final_url: str
    status: int
    content_type: str
    body: bytes
    etag: str | None = None
    last_modified: str | None = None


class OCCMemoTransport(Protocol):
    """Injectable remote boundary used by :class:`OCCMemoArchive`."""

    def fetch(self, memo_number: str, *, max_bytes: int) -> HTTPMemoResponse:
        """Fetch at most ``max_bytes + 1`` bytes so oversize responses are detectable."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Reject redirects before a request can leave the allowlisted OCC endpoint."""

    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> Request | None:
        del req, fp, code, msg, headers, newurl
        return None


class _Headers(Protocol):
    def get(self, key: str, default: str | None = None) -> str | None: ...


class _ReadableHTTPResponse(Protocol):
    status: int
    headers: _Headers

    def geturl(self) -> str: ...

    def read(self, amt: int = -1) -> bytes: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


type _OpenURL = Callable[[Request, float], _ReadableHTTPResponse]


class UrllibOCCMemoTransport:
    """Direct, no-redirect HTTPS transport with bounded response reads."""

    __slots__ = ("_open_url", "_timeout_seconds")
    _open_url: _OpenURL
    _timeout_seconds: float

    def __init__(
        self,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        open_url: _OpenURL | None = None,
    ) -> None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self._timeout_seconds = float(timeout_seconds)
        if open_url is None:
            opener = build_opener(_NoRedirectHandler())

            def _open(request: Request, timeout: float) -> _ReadableHTTPResponse:
                return cast(_ReadableHTTPResponse, opener.open(request, timeout=timeout))

            self._open_url = _open
        else:
            self._open_url = open_url

    def fetch(self, memo_number: str, *, max_bytes: int) -> HTTPMemoResponse:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        url = occ_memo_url(memo_number)
        request = Request(
            url,
            headers={
                "Accept": _PDF_CONTENT_TYPE,
                "User-Agent": "AlphaForge-OCC-Archive/1.0",
            },
            method="GET",
        )
        try:
            with self._open_url(request, self._timeout_seconds) as response:
                body = response.read(max_bytes + 1)
                return HTTPMemoResponse(
                    final_url=response.geturl(),
                    status=response.status,
                    content_type=response.headers.get("Content-Type", "") or "",
                    body=body,
                    etag=response.headers.get("ETag"),
                    last_modified=response.headers.get("Last-Modified"),
                )
        except HTTPError as exc:
            raise OCCMemoAcquisitionError(
                f"OCC memo {memo_number} request failed with HTTP {exc.code}"
            ) from exc
        except (URLError, OSError) as exc:
            raise OCCMemoAcquisitionError(
                f"OCC memo {memo_number} request failed before a complete response"
            ) from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class OCCMemoSnapshot:
    """One immutable local observation of official memo bytes."""

    memo_number: str
    source_url: str
    observed_at: Ms
    source_sha256: str
    byte_length: int
    content_type: str
    etag: str | None
    last_modified: str | None
    blob_relpath: str
    manifest_relpath: str
    manifest_sha256: str


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest_base(snapshot: OCCMemoSnapshot) -> dict[str, object]:
    return {
        "schema": _SCHEMA,
        "memo_number": snapshot.memo_number,
        "source_url": snapshot.source_url,
        "observed_at": snapshot.observed_at,
        "source_sha256": snapshot.source_sha256,
        "byte_length": snapshot.byte_length,
        "content_type": snapshot.content_type,
        "etag": snapshot.etag,
        "last_modified": snapshot.last_modified,
        "blob_relpath": snapshot.blob_relpath,
    }


def _validate_pdf(response: HTTPMemoResponse, memo_number: str, max_bytes: int) -> bytes:
    _validate_occ_url(response.final_url, memo_number)
    if response.status != 200:
        raise OCCMemoAcquisitionError(
            f"OCC memo {memo_number} returned HTTP status {response.status}, expected 200"
        )
    media_type = response.content_type.partition(";")[0].strip().lower()
    if media_type != _PDF_CONTENT_TYPE:
        raise OCCMemoAcquisitionError(
            f"OCC memo {memo_number} returned content type {response.content_type!r}, expected PDF"
        )
    body = response.body
    if len(body) > max_bytes:
        raise OCCMemoAcquisitionError(
            f"OCC memo {memo_number} exceeded maximum size {max_bytes} bytes"
        )
    if not body.startswith(b"%PDF-"):
        raise OCCMemoAcquisitionError(f"OCC memo {memo_number} lacks a PDF header")
    if b"%%EOF" not in body[-1024:]:
        raise OCCMemoAcquisitionError(f"OCC memo {memo_number} lacks a terminal PDF EOF marker")
    return body


class OCCMemoArchive:
    """Immutable content-addressed memo archive rooted at an operator-selected directory."""

    __slots__ = ("_max_bytes", "_root", "_transport")

    def __init__(
        self,
        root: Path,
        *,
        transport: OCCMemoTransport | None = None,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self._root = root
        self._transport = UrllibOCCMemoTransport() if transport is None else transport
        self._max_bytes = max_bytes

    def acquire(self, memo_number: str, *, observed_at: Ms) -> OCCMemoSnapshot:
        """Fetch, validate, and atomically persist one official memo observation."""
        _validate_memo_number(memo_number)
        if observed_at < 0:
            raise ValueError("observed_at must be nonnegative epoch milliseconds")
        response = self._transport.fetch(memo_number, max_bytes=self._max_bytes)
        body = _validate_pdf(response, memo_number, self._max_bytes)
        digest = _sha256(body)
        blob_relpath = f"blobs/sha256/{digest[:2]}/{digest}.pdf"
        manifest_relpath = f"manifests/{memo_number}/{observed_at}-{digest}.json"
        snapshot = OCCMemoSnapshot(
            memo_number=memo_number,
            source_url=occ_memo_url(memo_number),
            observed_at=observed_at,
            source_sha256=digest,
            byte_length=len(body),
            content_type=_PDF_CONTENT_TYPE,
            etag=response.etag,
            last_modified=response.last_modified,
            blob_relpath=blob_relpath,
            manifest_relpath=manifest_relpath,
            manifest_sha256="",
        )
        base = _manifest_base(snapshot)
        manifest_digest = _sha256(_canonical_bytes(base))
        snapshot = OCCMemoSnapshot(
            memo_number=snapshot.memo_number,
            source_url=snapshot.source_url,
            observed_at=snapshot.observed_at,
            source_sha256=snapshot.source_sha256,
            byte_length=snapshot.byte_length,
            content_type=snapshot.content_type,
            etag=snapshot.etag,
            last_modified=snapshot.last_modified,
            blob_relpath=snapshot.blob_relpath,
            manifest_relpath=snapshot.manifest_relpath,
            manifest_sha256=manifest_digest,
        )
        manifest = {**base, "manifest_sha256": manifest_digest}
        self._reject_same_timestamp_mutation(snapshot)
        self._write_immutable(self._root / blob_relpath, body)
        self._write_immutable(
            self._root / manifest_relpath,
            json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n",
        )
        self.verify(snapshot)
        return snapshot

    def snapshots(self, memo_number: str) -> tuple[OCCMemoSnapshot, ...]:
        """Load and verify every archived observation for one memo in PIT order."""
        _validate_memo_number(memo_number)
        directory = self._root / "manifests" / memo_number
        if not directory.exists():
            return ()
        loaded = tuple(self._load_manifest(path) for path in directory.glob("*.json"))
        snapshots = tuple(sorted(loaded, key=lambda snapshot: snapshot.observed_at))
        observed = [snapshot.observed_at for snapshot in snapshots]
        if len(observed) != len(set(observed)):
            raise OCCMemoArchiveIntegrityError(
                f"memo {memo_number} manifests do not have unique observation times"
            )
        for snapshot in snapshots:
            self.verify(snapshot)
        return snapshots

    def verify(self, snapshot: OCCMemoSnapshot) -> None:
        """Re-hash one snapshot's manifest and PDF and validate all derived paths."""
        _validate_memo_number(snapshot.memo_number)
        _validate_occ_url(snapshot.source_url, snapshot.memo_number)
        if _SHA256_RE.fullmatch(snapshot.source_sha256) is None:
            raise OCCMemoArchiveIntegrityError("snapshot source_sha256 is malformed")
        expected_blob = (
            f"blobs/sha256/{snapshot.source_sha256[:2]}/{snapshot.source_sha256}.pdf"
        )
        expected_manifest = (
            f"manifests/{snapshot.memo_number}/{snapshot.observed_at}-"
            f"{snapshot.source_sha256}.json"
        )
        if snapshot.blob_relpath != expected_blob or snapshot.manifest_relpath != expected_manifest:
            raise OCCMemoArchiveIntegrityError("snapshot path does not match its identity")
        blob = self._read_required(self._root / expected_blob)
        if len(blob) != snapshot.byte_length or _sha256(blob) != snapshot.source_sha256:
            raise OCCMemoArchiveIntegrityError("archived OCC PDF no longer matches its digest")
        if not blob.startswith(b"%PDF-") or b"%%EOF" not in blob[-1024:]:
            raise OCCMemoArchiveIntegrityError("archived OCC blob is not a complete PDF")
        manifest_payload = self._read_manifest_payload(self._root / expected_manifest)
        expected_payload = {
            **_manifest_base(snapshot),
            "manifest_sha256": snapshot.manifest_sha256,
        }
        if manifest_payload != expected_payload:
            raise OCCMemoArchiveIntegrityError("manifest content differs from snapshot metadata")
        if _sha256(_canonical_bytes(_manifest_base(snapshot))) != snapshot.manifest_sha256:
            raise OCCMemoArchiveIntegrityError("manifest digest no longer matches metadata")

    def _reject_same_timestamp_mutation(self, snapshot: OCCMemoSnapshot) -> None:
        directory = self._root / "manifests" / snapshot.memo_number
        for path in directory.glob(f"{snapshot.observed_at}-*.json"):
            if path.name != Path(snapshot.manifest_relpath).name:
                raise OCCMemoArchiveConflictError(
                    f"memo {snapshot.memo_number} already has different content observed at "
                    f"{snapshot.observed_at}"
                )

    def _load_manifest(self, path: Path) -> OCCMemoSnapshot:
        payload = self._read_manifest_payload(path)
        expected = {
            "schema",
            "memo_number",
            "source_url",
            "observed_at",
            "source_sha256",
            "byte_length",
            "content_type",
            "etag",
            "last_modified",
            "blob_relpath",
            "manifest_sha256",
        }
        if set(payload) != expected or payload.get("schema") != _SCHEMA:
            raise OCCMemoArchiveIntegrityError(f"invalid OCC manifest schema at {path}")
        try:
            snapshot = OCCMemoSnapshot(
                memo_number=self._required_text(payload, "memo_number"),
                source_url=self._required_text(payload, "source_url"),
                observed_at=self._required_int(payload, "observed_at"),
                source_sha256=self._required_text(payload, "source_sha256"),
                byte_length=self._required_int(payload, "byte_length"),
                content_type=self._required_text(payload, "content_type"),
                etag=self._optional_text(payload, "etag"),
                last_modified=self._optional_text(payload, "last_modified"),
                blob_relpath=self._required_text(payload, "blob_relpath"),
                manifest_relpath=str(path.relative_to(self._root)),
                manifest_sha256=self._required_text(payload, "manifest_sha256"),
            )
        except (TypeError, ValueError) as exc:
            raise OCCMemoArchiveIntegrityError(f"malformed OCC manifest at {path}") from exc
        return snapshot

    @staticmethod
    def _read_manifest_payload(path: Path) -> dict[str, object]:
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise OCCMemoArchiveIntegrityError(f"cannot read OCC manifest {path}") from exc
        if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
            raise OCCMemoArchiveIntegrityError(f"OCC manifest {path} must be a JSON object")
        return payload

    @staticmethod
    def _required_text(payload: Mapping[str, object], key: str) -> str:
        value = payload[key]
        if not isinstance(value, str) or not value:
            raise TypeError(f"{key} must be nonempty text")
        return value

    @staticmethod
    def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
        value = payload[key]
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{key} must be text or null")
        return value

    @staticmethod
    def _required_int(payload: Mapping[str, object], key: str) -> int:
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be an integer")
        return value

    @staticmethod
    def _read_required(path: Path) -> bytes:
        try:
            return path.read_bytes()
        except OSError as exc:
            raise OCCMemoArchiveIntegrityError(
                f"required archive file is unreadable: {path}"
            ) from exc

    @staticmethod
    def _write_immutable(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with tmp.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(tmp, path)
            except FileExistsError:
                try:
                    existing = path.read_bytes()
                except OSError as exc:
                    raise OCCMemoArchiveIntegrityError(
                        f"immutable archive target is unreadable: {path}"
                    ) from exc
                if existing != data:
                    raise OCCMemoArchiveConflictError(
                        f"immutable archive target already contains different bytes: {path}"
                    ) from None
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            tmp.unlink(missing_ok=True)
