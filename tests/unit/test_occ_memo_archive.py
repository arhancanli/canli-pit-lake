"""Immutable OCC memo acquisition and content-addressed archive tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from alphaforge.data.ingest.occ_memo_archive import (
    HTTPMemoResponse,
    OCCMemoAcquisitionError,
    OCCMemoArchive,
    OCCMemoArchiveConflictError,
    OCCMemoArchiveIntegrityError,
    UrllibOCCMemoTransport,
    occ_memo_url,
)


def _pdf(label: str = "memo") -> bytes:
    return f"%PDF-1.7\n1 0 obj\n<< /Label ({label}) >>\nendobj\n%%EOF\n".encode()


class _Transport:
    def __init__(self, response: HTTPMemoResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, int]] = []

    def fetch(self, memo_number: str, *, max_bytes: int) -> HTTPMemoResponse:
        self.calls.append((memo_number, max_bytes))
        return self.response


def _response(
    memo_number: str = "59573",
    *,
    body: bytes | None = None,
    final_url: str | None = None,
    status: int = 200,
    content_type: str = "application/pdf",
    etag: str | None = '"abc"',
    last_modified: str | None = "Fri, 14 Aug 2026 12:00:00 GMT",
) -> HTTPMemoResponse:
    return HTTPMemoResponse(
        final_url=occ_memo_url(memo_number) if final_url is None else final_url,
        status=status,
        content_type=content_type,
        body=_pdf() if body is None else body,
        etag=etag,
        last_modified=last_modified,
    )


def _archive(tmp_path: Path, response: HTTPMemoResponse | None = None, *, max_bytes: int = 1024):
    transport = _Transport(_response() if response is None else response)
    return OCCMemoArchive(tmp_path, transport=transport, max_bytes=max_bytes), transport


def test_canonical_url_rejects_ambiguous_memo_numbers() -> None:
    assert occ_memo_url("59573") == "https://infomemo.theocc.com/infomemos?number=59573"
    for invalid in ("", "0", "00001", "12A", "123456789", "../59573"):
        with pytest.raises(ValueError, match="memo_number"):
            occ_memo_url(invalid)


def test_acquire_persists_content_addressed_pdf_and_hashed_manifest(tmp_path: Path) -> None:
    archive, transport = _archive(tmp_path)
    snapshot = archive.acquire("59573", observed_at=100)
    assert transport.calls == [("59573", 1024)]
    assert snapshot.memo_number == "59573"
    assert snapshot.source_url == occ_memo_url("59573")
    assert snapshot.byte_length == len(_pdf())
    assert len(snapshot.source_sha256) == 64
    assert len(snapshot.manifest_sha256) == 64
    assert (tmp_path / snapshot.blob_relpath).read_bytes() == _pdf()
    manifest = json.loads((tmp_path / snapshot.manifest_relpath).read_text())
    assert manifest["source_sha256"] == snapshot.source_sha256
    assert manifest["manifest_sha256"] == snapshot.manifest_sha256
    archive.verify(snapshot)
    assert archive.snapshots("59573") == (snapshot,)


def test_identical_observation_is_idempotent_and_leaves_no_temp_files(tmp_path: Path) -> None:
    archive, _ = _archive(tmp_path)
    first = archive.acquire("59573", observed_at=100)
    second = archive.acquire("59573", observed_at=100)
    assert first == second
    assert list(tmp_path.rglob("*.pdf")) == [tmp_path / first.blob_relpath]
    assert list(tmp_path.rglob("*.json")) == [tmp_path / first.manifest_relpath]
    assert list(tmp_path.rglob("*.tmp")) == []


def test_later_source_mutation_creates_history_without_overwriting(tmp_path: Path) -> None:
    archive, transport = _archive(tmp_path)
    first = archive.acquire("59573", observed_at=100)
    transport.response = _response(body=_pdf("revised"), etag='"revised"')
    second = archive.acquire("59573", observed_at=200)
    assert first.source_sha256 != second.source_sha256
    assert archive.snapshots("59573") == (first, second)
    assert (tmp_path / first.blob_relpath).read_bytes() == _pdf()
    assert (tmp_path / second.blob_relpath).read_bytes() == _pdf("revised")


def test_different_content_at_same_observation_timestamp_fails_closed(tmp_path: Path) -> None:
    archive, transport = _archive(tmp_path)
    archive.acquire("59573", observed_at=100)
    transport.response = _response(body=_pdf("mutated"))
    with pytest.raises(OCCMemoArchiveConflictError, match="different content observed"):
        archive.acquire("59573", observed_at=100)


def test_same_body_with_changed_headers_cannot_rewrite_manifest(tmp_path: Path) -> None:
    archive, transport = _archive(tmp_path)
    archive.acquire("59573", observed_at=100)
    transport.response = _response(etag='"different"')
    with pytest.raises(OCCMemoArchiveConflictError, match="different bytes"):
        archive.acquire("59573", observed_at=100)


def test_identical_pdf_for_two_memos_deduplicates_blob_only(tmp_path: Path) -> None:
    archive, transport = _archive(tmp_path)
    first = archive.acquire("59573", observed_at=100)
    transport.response = _response("59446")
    second = archive.acquire("59446", observed_at=110)
    assert first.blob_relpath == second.blob_relpath
    assert first.manifest_relpath != second.manifest_relpath
    assert len(list((tmp_path / "blobs").rglob("*.pdf"))) == 1
    assert len(list((tmp_path / "manifests").rglob("*.json"))) == 2


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_response(status=404), "HTTP status"),
        (_response(content_type="text/html", body=b"<html>not found</html>"), "content type"),
        (_response(body=b"not a pdf\n%%EOF\n"), "PDF header"),
        (_response(body=b"%PDF-1.7\ntruncated"), "EOF marker"),
        (
            _response(final_url="https://example.com/infomemos?number=59573"),
            "canonical OCC endpoint",
        ),
        (
            _response(final_url=occ_memo_url("59446")),
            "canonical OCC endpoint",
        ),
    ],
)
def test_remote_response_shape_failures_write_nothing(
    tmp_path: Path, response: HTTPMemoResponse, message: str
) -> None:
    archive, _ = _archive(tmp_path, response)
    with pytest.raises(OCCMemoAcquisitionError, match=message):
        archive.acquire("59573", observed_at=100)
    assert not tmp_path.exists() or list(tmp_path.rglob("*")) == []


def test_content_type_parameters_and_oversize_body(tmp_path: Path) -> None:
    accepted, _ = _archive(tmp_path / "accepted", _response(content_type="Application/PDF; x=1"))
    accepted.acquire("59573", observed_at=100)

    rejected, _ = _archive(tmp_path / "rejected", _response(body=_pdf() + b"x" * 100), max_bytes=20)
    with pytest.raises(OCCMemoAcquisitionError, match="maximum size"):
        rejected.acquire("59573", observed_at=100)


def test_archive_configuration_and_observation_timestamp_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        OCCMemoArchive(tmp_path, transport=_Transport(_response()), max_bytes=0)
    archive, _ = _archive(tmp_path)
    with pytest.raises(ValueError, match="observed_at"):
        archive.acquire("59573", observed_at=-1)


def test_corrupt_or_missing_blob_is_detected_on_verify_and_listing(tmp_path: Path) -> None:
    archive, _ = _archive(tmp_path)
    snapshot = archive.acquire("59573", observed_at=100)
    blob = tmp_path / snapshot.blob_relpath
    blob.write_bytes(_pdf("corrupt"))
    with pytest.raises(OCCMemoArchiveIntegrityError, match="digest"):
        archive.verify(snapshot)
    with pytest.raises(OCCMemoArchiveIntegrityError, match="digest"):
        archive.snapshots("59573")
    blob.unlink()
    with pytest.raises(OCCMemoArchiveIntegrityError, match="unreadable"):
        archive.verify(snapshot)


def test_manifest_json_schema_hash_and_derived_path_are_verified(tmp_path: Path) -> None:
    archive, _ = _archive(tmp_path)
    snapshot = archive.acquire("59573", observed_at=100)
    manifest_path = tmp_path / snapshot.manifest_relpath

    payload = json.loads(manifest_path.read_text())
    payload["manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(OCCMemoArchiveIntegrityError, match="manifest digest"):
        archive.snapshots("59573")

    manifest_path.write_text("not json")
    with pytest.raises(OCCMemoArchiveIntegrityError, match="cannot read"):
        archive.snapshots("59573")


def test_manifest_filename_cannot_redirect_verification_to_another_path(tmp_path: Path) -> None:
    archive, _ = _archive(tmp_path)
    snapshot = archive.acquire("59573", observed_at=100)
    original = tmp_path / snapshot.manifest_relpath
    copied = original.with_name(f"101-{snapshot.source_sha256}.json")
    copied.write_bytes(original.read_bytes())
    original.unlink()
    with pytest.raises(OCCMemoArchiveIntegrityError, match="path does not match"):
        archive.snapshots("59573")


def test_snapshot_listing_is_empty_and_sorted(tmp_path: Path) -> None:
    archive, transport = _archive(tmp_path)
    assert archive.snapshots("59573") == ()
    later = archive.acquire("59573", observed_at=200)
    transport.response = _response(body=_pdf("earlier-source-version"), etag=None)
    earlier = archive.acquire("59573", observed_at=100)
    assert archive.snapshots("59573") == (earlier, later)

    transport.response = _response(body=_pdf("single-digit-time"), etag=None)
    earliest = archive.acquire("59573", observed_at=9)
    assert archive.snapshots("59573") == (earliest, earlier, later)


class _HeadersMap:
    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = values

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._values.get(key, default)


class _HTTPContext:
    def __init__(self, url: str, body: bytes) -> None:
        self.status = 200
        self.headers = _HeadersMap(
            {
                "Content-Type": "application/pdf",
                "ETag": '"transport"',
                "Last-Modified": "Tue, 18 Aug 2026 00:00:00 GMT",
            }
        )
        self._url = url
        self._body = body
        self.read_amounts: list[int] = []

    def geturl(self) -> str:
        return self._url

    def read(self, amt: int = -1) -> bytes:
        self.read_amounts.append(amt)
        return self._body[:amt]

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback


def test_urllib_transport_builds_allowlisted_bounded_request() -> None:
    context = _HTTPContext(occ_memo_url("59573"), _pdf())
    captured: list[tuple[Request, float]] = []

    def open_url(request: Request, timeout: float):
        captured.append((request, timeout))
        return context

    transport = UrllibOCCMemoTransport(timeout_seconds=7.5, open_url=open_url)
    response = transport.fetch("59573", max_bytes=1024)
    assert captured[0][0].full_url == occ_memo_url("59573")
    assert captured[0][0].method == "GET"
    assert captured[0][0].get_header("Accept") == "application/pdf"
    assert captured[0][0].get_header("User-agent") == "AlphaForge-OCC-Archive/1.0"
    assert captured[0][1] == 7.5
    assert context.read_amounts == [1025]
    assert response.body == _pdf()
    assert response.etag == '"transport"'


def test_urllib_transport_converts_http_errors_and_validates_limits() -> None:
    def failing_open(request: Request, timeout: float):
        del request, timeout
        raise HTTPError(occ_memo_url("59573"), 503, "down", hdrs=None, fp=None)

    transport = UrllibOCCMemoTransport(open_url=failing_open)
    with pytest.raises(OCCMemoAcquisitionError, match="HTTP 503"):
        transport.fetch("59573", max_bytes=1024)
    with pytest.raises(ValueError, match="max_bytes"):
        transport.fetch("59573", max_bytes=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        UrllibOCCMemoTransport(timeout_seconds=0)


def test_urllib_transport_converts_network_failures() -> None:
    def failing_open(request: Request, timeout: float):
        del request, timeout
        raise URLError("TLS failure")

    transport = UrllibOCCMemoTransport(open_url=failing_open)
    with pytest.raises(OCCMemoAcquisitionError, match="before a complete response"):
        transport.fetch("59573", max_bytes=1024)
