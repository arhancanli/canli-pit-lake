#!/usr/bin/env python3
"""Prove this repository is a faithful extraction, not a fork that drifted.

Every source and test file here was copied byte-for-byte out of the ALPHAC engine
(github.com/arhancanli/alphac) at the pinned commit below. `extraction_manifest.json`
records the SHA-256 of each one at extraction time. This script re-reads those files
from the engine and fails if a single byte differs.

That matters because the whole point of publishing an extraction is that a reader
can trust it is the same code that produced the public record. A copy nobody checks
is a screenshot. This check makes drift a build failure instead of a surprise.

Usage:
    python tools/check_parity.py                 # fetch from GitHub at the pinned commit
    ALPHAC_PATH=~/alphaforge python tools/check_parity.py   # compare against a local checkout
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "extraction_manifest.json"

#: The exact ALPHAC commit this extraction was taken from. Bump it only together
#: with a re-extraction, never to make a red check go green.
ENGINE_COMMIT = "345b38d20436c4fcf2474eac5ff863e990074019"
RAW = "https://raw.githubusercontent.com/arhancanli/alphac/{commit}/{path}"


def read_local(base: Path, path: str) -> bytes | None:
    candidate = base / path
    return candidate.read_bytes() if candidate.is_file() else None


def read_remote(path: str) -> bytes | None:
    url = RAW.format(commit=ENGINE_COMMIT, path=path)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return bytes(response.read())
    except urllib.error.HTTPError:
        return None


def main() -> int:
    manifest: dict[str, str] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    local_engine = os.environ.get("ALPHAC_PATH")
    base = Path(local_engine).expanduser().resolve() if local_engine else None
    source = f"local checkout {base}" if base else f"arhancanli/alphac@{ENGINE_COMMIT[:12]}"
    print(f"comparing {len(manifest)} files against {source}")

    def check(item: tuple[str, str]) -> tuple[str, str]:
        path, expected = item
        mine = ROOT / path
        if not mine.is_file():
            return path, "MISSING HERE"
        if hashlib.sha256(mine.read_bytes()).hexdigest() != expected:
            return path, "LOCAL FILE DOES NOT MATCH ITS OWN MANIFEST"
        theirs = read_local(base, path) if base else read_remote(path)
        if theirs is None:
            return path, "MISSING IN ENGINE"
        if hashlib.sha256(theirs).hexdigest() != expected:
            return path, "DRIFTED FROM ENGINE"
        return path, ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(check, sorted(manifest.items())))

    failures = [(p, why) for p, why in results if why]
    if failures:
        print(f"\nFAIL: {len(failures)} of {len(manifest)} files are not faithful copies")
        for path, why in failures:
            print(f"  {why:<38} {path}")
        return 1
    print(f"OK: all {len(manifest)} files are byte-identical to the engine")
    return 0


if __name__ == "__main__":
    sys.exit(main())
