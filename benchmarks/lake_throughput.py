"""Measured throughput for the point-in-time lake.

Run it yourself -- the numbers in the README came from this file and nothing else:

    uv run python benchmarks/lake_throughput.py

It builds a synthetic 1,000,000-bar lake (50 instruments x 20,000 hourly bars) in a
temporary directory, then times the write path and two read paths: one where every
bar is visible, and one where as_of sits at the midpoint so the point-in-time
filter has to exclude half the data. The second case returning exactly 500,000 rows
is the filter demonstrating itself, not just a speed measurement.

Reads are the best of five runs; the median is printed beside it so a single lucky
run cannot be quoted as the headline.
"""
import platform
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from alphaforge.core.time import Timeframe
from alphaforge.data.schemas import OHLCV_SCHEMA, Dataset
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.reader import PITDataReader
from alphaforge.data.store.writer import LakeWriter

HOUR = 3_600_000
N_INST, N_BARS = 50, 20_000          # 1,000,000 bars total
START = 1_600_000_000_000 - (1_600_000_000_000 % HOUR)

def make(inst):
    rng = np.random.default_rng(abs(hash(inst)) % 2**32)
    ts = START + np.arange(N_BARS, dtype=np.int64) * HOUR
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, N_BARS)))
    return pa.table({
        "instrument_id": pa.array([inst]*N_BARS, pa.string()),
        "ts_open": pa.array(ts, pa.timestamp("ms", tz="UTC")),
        "open": pa.array(close, pa.float64()), "high": pa.array(close*1.001, pa.float64()),
        "low": pa.array(close*0.999, pa.float64()), "close": pa.array(close, pa.float64()),
        "volume": pa.array(rng.random(N_BARS)*1000, pa.float64()),
        "quote_volume": pa.array(rng.random(N_BARS)*100000, pa.float64()),
        "n_trades": pa.array(rng.integers(1, 500, N_BARS), pa.int64()),
        "quality_flags": pa.array(np.zeros(N_BARS, dtype=np.int32), pa.int32()),
        "ingested_at": pa.array([START]*N_BARS, pa.timestamp("ms", tz="UTC")),
    }, schema=OHLCV_SCHEMA)

with tempfile.TemporaryDirectory() as tmp:
    paths = LakePaths(Path(tmp))
    writer = LakeWriter(paths)
    insts = [f"XBIN:PERP{i:03d}" for i in range(N_INST)]
    tables = [make(i) for i in insts]
    t0 = time.perf_counter()
    for t in tables:
        writer.write(Dataset.OHLCV, t)
    w = time.perf_counter() - t0
    total = N_INST * N_BARS
    print(f"write        {total:,} bars in {w:6.2f}s   {total/w:>12,.0f} bars/s")

    reader = PITDataReader(paths)
    as_of = START + N_BARS * HOUR          # everything visible
    mid   = START + (N_BARS // 2) * HOUR   # half visible: the PIT filter doing work
    for label, ao in (("read (full)", as_of), ("read (PIT-cut)", mid)):
        runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            tbl = reader.ohlcv(
                insts, start=START, end=START + N_BARS * HOUR, as_of=ao, tf=Timeframe.H1
            )
            runs.append(time.perf_counter() - t0)
        best, rows = min(runs), tbl.num_rows
        print(
            f"{label:<13}{rows:>10,} rows in {best:6.2f}s   {rows / best:>12,.0f} rows/s"
            f"   (median {statistics.median(runs):.2f}s of 5)"
        )
    size = sum(f.stat().st_size for f in Path(tmp).rglob("*.parquet"))
    print(
        f"on disk      {size / 1e6:.1f} MB for {total:,} bars   "
        f"{size / total:.1f} bytes/bar (zstd-3)"
    )
print(f"machine      {platform.machine()} · python {platform.python_version()}")
