# canli-pit-lake

**A market-data lake that cannot accidentally tell you the future.**

[![ci](https://github.com/arhancanli/canli-pit-lake/actions/workflows/ci.yml/badge.svg)](https://github.com/arhancanli/canli-pit-lake/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![parity: byte-identical to engine](https://img.shields.io/badge/parity-byte--identical-brightgreen.svg)](tools/check_parity.py)

Every read takes an explicit `as_of` timestamp and returns only what a person standing at that
instant could actually have known. There is no `as_of=now` default, on purpose — you have to
state your information time, which is what makes backtest code and live code the same code.

This is the data layer of [ALPHAC](https://github.com/arhancanli/alphac), the research engine
behind [canlicapital.com](https://canlicapital.com), published on its own because the data layer
is where quantitative research quietly goes wrong. Built and maintained by
[Arhan Canli](https://github.com/arhancanli).

---

## The problem this exists to solve

A backtest is a claim about a decision you *could* have made. Almost every way that claim turns
out to be false is a data problem, and none of them raise an exception:

**Your index membership is from today.** You backtest "buy the S&P 500 dip" over 20 years using
today's constituent list. Every company that went bankrupt is missing. Your strategy looks
excellent because you only ever traded survivors.

**Your prices are adjusted with today's split factors.** A 4:1 split that happened in 2020 gets
retroactively applied to 2015 prices. The 2015 backtest now trades at prices no one could have
seen on a screen in 2015.

**Your fundamentals are restated.** You use the earnings figure the company reported *after* two
revisions, three quarters later, and treat it as available on the original filing date.

**Your quality flag was computed from tomorrow's bar.** You mark a print as suspicious using the
fact that it reverted on the next bar, then let a decision at *this* bar see the mark.

Each one produces a beautiful curve and a strategy that loses money. Each one is prevented
structurally here, not by remembering to be careful.

---

## How it is prevented

### One rule, in one file

The point-in-time rule lives in `PITDataReader` and nowhere else. Every analytical read takes
`as_of` (epoch milliseconds, UTC) and filters to records available at that instant.

A bar labelled `ts_open` covers `[ts_open, ts_open + Δ)` and becomes available at its **close**:

```
visible(bar)  ⟺  ts_open + Δ  ≤  as_of
```

`available_at` is *derived* for OHLCV, never stored — storing it would invite the stored value
and the rule to drift apart. Funding rows do store `available_at = ts_funding + δ` (default 5
minutes of publication lag), and every consumer joins on `available_at`, never on `ts_funding`.

### Quality flags carry their own availability lag

This is the subtle one. A bad-print detector is only convincing if it uses the *next* bar's
reversion — so its output is knowable two bars later, not now. Each flag bit `b` therefore
declares its own lag `L_b`, and the reader exposes the bit only when:

```
ts_open + (1 + L_b)·Δ  ≤  as_of
```

Bits that are not yet available are **cleared in the returned rows**; the stored file is never
touched. A bit with no declared lag is never exposed at all — fail-closed, so adding a new flag
without declaring its lag makes it invisible rather than dangerous.

### Gaps are reported, never filled

A missing bar is a fact about the world. Synthesising one to make a series contiguous invents
data. `GapCheck` reports the gap and flags the surviving bars on either side of it; nothing in
this library ever writes a price that did not print.

### Dead symbols never leave

`UniverseStore` records membership as intervals `[effective_from, effective_to)` computed only
from data available at the rebalance instant. Delisted instruments are ordinary rows. The only
operation that removes a dead symbol is a full rebuild, which regenerates it from the raw bars —
which are never deleted. Survivorship bias has nowhere to enter.

### Writes are crash-safe by protocol, not by hope

A leaf partition is always rewritten whole: to a temp file in the *same* directory (same
filesystem, so `os.replace` is atomic), then promoted. Readers only ever open `data.parquet`, so
an in-flight write is structurally invisible to them. A crashed writer leaves inert garbage, not
a torn file.

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
   vendor feeds  ─────► │  sources/     typed adapters, one per     │
   (ccxt, polygon,      │               vendor; no vendor type      │
    S3 flat files)      │               escapes this layer          │
                        └────────────────────┬─────────────────────┘
                                             │  raw rows
                        ┌────────────────────▼─────────────────────┐
                        │  ingest/      backfill · checkpoints ·   │
                        │               retry · equities ·          │
                        │               fundamentals · option       │
                        │               adjustments · OCC memos     │
                        │               resumable, idempotent       │
                        └────────────────────┬─────────────────────┘
                                             │  validated against schemas.py
                        ┌────────────────────▼─────────────────────┐
                        │  quality/     gaps · outliers · bad       │
                        │               prints · stale closes ·     │
                        │               exchange downtime           │
                        │               ANNOTATES, never alters     │
                        └────────────────────┬─────────────────────┘
                                             │  bars + flag bitmask
                        ┌────────────────────▼─────────────────────┐
                        │  store/       Parquet lake, zstd-3        │
                        │    writer     tmp-write → os.replace      │
                        │    lake       <root>/<dataset>/           │
                        │                 instrument_id=<id>/       │
                        │                   year=<YYYY>/data.parquet│
                        └────────────────────┬─────────────────────┘
                                             │
   ┌─────────────────────────────────────────▼─────────────────────┐
   │  store/reader.py  —  PITDataReader                            │
   │                                                               │
   │  read(dataset, instruments, start, end, as_of)                │
   │    ├─ OHLCV      visible ⟺ ts_open + Δ ≤ as_of                │
   │    ├─ flags      bit b exposed ⟺ ts_open + (1+L_b)·Δ ≤ as_of  │
   │    └─ funding    visible ⟺ stored available_at ≤ as_of        │
   │                                                               │
   │  THE PIT RULE LIVES HERE AND ONLY HERE                        │
   └─────────────────────────────────────────┬─────────────────────┘
                                             │
                        ┌────────────────────▼─────────────────────┐
                        │  universe/    membership as intervals;   │
                        │               delisted names persist     │
                        └──────────────────────────────────────────┘
```

DuckDB is used as a **query engine only** — it never owns storage. The files on disk are plain
Parquet you can open with anything.

---

## Quick start

```bash
git clone https://github.com/arhancanli/canli-pit-lake.git
cd canli-pit-lake
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest                      # the extracted suite
uv run python tools/check_parity.py  # prove this is the engine's code, byte for byte
```

Reading, with the rule doing its job:

```python
from alphaforge.core.time import Timeframe
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.reader import PITDataReader

reader = PITDataReader(LakePaths("data/lake"))

# Information time is 2024-06-01T00:00:00Z. Ask for a year of daily bars.
bars = reader.ohlcv(
    ["XNAS:AAPL"],
    start=1_704_067_200_000,   # 2024-01-01
    end=1_735_689_600_000,     # 2025-01-01 — deliberately past as_of
    as_of=1_717_200_000_000,   # 2024-06-01
    tf=Timeframe.D1,
)
```

`end` is in the future relative to `as_of`. That is allowed, and the result still stops at
2024-06-01: the PIT filter governs, not the requested range. A bar closing exactly at `as_of` is
visible; one millisecond earlier it is not.

Corporate actions answer to the same rule, keyed on when the action was *knowable*:

```python
actions = reader.corporate_actions(
    ["XNAS:AAPL"],
    start=1_704_067_200_000,
    end=1_735_689_600_000,
    as_of=1_717_200_000_000,
)
```

The predicate is `available_at <= as_of` — the declaration date plus publication lag — and never
`ex_date`. A split with an ex-date inside your window but declared one millisecond after your
`as_of` is invisible, because on that day nobody knew it was coming.

There is no way to spell "give me everything you have now" without writing an `as_of` of now and
meaning it.

---

## What is inside

| package | what it owns |
|---|---|
| `alphaforge.data.schemas` | the Parquet column contracts; the writer validates every table against them |
| `alphaforge.data.store` | lake layout, crash-safe writer, resampler, and `PITDataReader` |
| `alphaforge.data.ingest` | resumable backfill, checkpoints, retry, equities, fundamentals, option adjustments, OCC memo archive |
| `alphaforge.data.sources` | vendor adapters (ccxt, Polygon API, Polygon flat files, Binance Vision) |
| `alphaforge.data.quality` | the six checks and their reports |
| `alphaforge.data.universe` | point-in-time membership intervals and the rebalance builder |
| `alphaforge.core` | calendars, instrument identity, symbols, time, typed errors |

Fully typed (`mypy --strict`, `py.typed`), zero `Any` escapes across the public surface.

---

## Benchmarks

Measured, not estimated. Every number below comes from
[`benchmarks/lake_throughput.py`](benchmarks/lake_throughput.py) and you can re-run it:

```bash
uv run python benchmarks/lake_throughput.py
```

A synthetic 1,000,000-bar lake — 50 instruments × 20,000 hourly bars — built in a temp directory.
Reads are the best of five runs with the median reported beside them, so one lucky run cannot
become the headline.

| operation | throughput |
|---|---|
| write (validate → tmp file → atomic promote) | **≈1.05M bars/s** |
| read, everything visible | **≈15M rows/s** |
| read, `as_of` at the midpoint (PIT filter excluding half) | **≈14M rows/s** |
| on disk, zstd-3 | **60.6 bytes/bar** |

Measured on Apple silicon (arm64), Python 3.12.13, single process. Absolute numbers will differ on
your machine; the ratio that matters is that the point-in-time filter is close to free — it is a
predicate pushed into the DuckDB scan, not a post-filter in Python.

The PIT-cut row is also a correctness demonstration: with `as_of` at the midpoint it returns
**exactly 500,000** of the 1,000,000 bars. If the rule ever silently stopped applying, that number
would change before the timing did.

### What is verified

| check | result |
|---|---|
| tests | **941 passed**, 12 network tests deselected by default |
| types | `mypy --strict`, **0 issues** across 42 source files |
| lint | `ruff`, **clean** |
| parity with the engine | **146 files byte-identical** |

---

## Provenance

This repository is a **mechanical extraction**, not a fork.

`extraction_manifest.json` records the SHA-256 of every file at extraction time.
`tools/check_parity.py` re-reads each one from the engine repository at the pinned commit and
fails if a single byte differs. It runs in CI on every push.

The module set was not hand-picked either: it is the transitive import closure of
`alphaforge.data`, and the test set is every test in the engine whose imports are fully satisfied
by that closure. Only files the engine actually **publishes** are shipped — its `git ls-files`
set, not its working tree — so nothing the engine deliberately gitignores can be republished here.
A copy nobody checks is a screenshot; this one is checked.

```bash
python tools/check_parity.py                        # against GitHub at the pinned commit
ALPHAC_PATH=~/alphaforge python tools/check_parity.py  # against a local checkout
```

---

## Related

- **[alphac](https://github.com/arhancanli/alphac)** — the full research engine this comes from
- **[canli-backtest](https://github.com/arhancanli/canli-backtest)** — the backtester that reads
  this lake, with fill-time causality and multiple-testing accounting
- **[canlicapital.com](https://canlicapital.com)** — the live paper record these produce

## License and citation

MIT. Copyright © 2026 Arhan Canli. Machine-readable citation metadata is in
[`CITATION.cff`](CITATION.cff).

Created and maintained by **[Arhan Canli](https://github.com/arhancanli)**. Development uses
reviewed AI-assisted tooling; ownership, design decisions, and published claims are mine.
