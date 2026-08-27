"""Phase 3 integration: quality flags → PIT masking → resample → universe → backtest read.

One synthetic 1h lake (tmp_path only — the real ./data is NEVER touched) with six fake
perps spanning Jan-Feb 2024, exercising the full data-integrity pipeline end to end:

1. a planted bad print (price spike on thin volume that immediately reverts) is flagged
   ``OUTLIER_RETURN | BAD_PRINT_SUSPECT`` by ``apply_flags`` — prices bit-for-bit
   untouched (flags ANNOTATE, never alter);
2. the PIT reader exposes ``OUTLIER_RETURN`` at the bar's close but masks the lag-2
   ``BAD_PRINT_SUSPECT`` bit until ``ts_open + 3Δ`` (leakage finding 5);
3. a planted 2-hour gap is reported (``GAP_ADJACENT`` on both survivors), never filled,
   and propagates through the resampler as MISSING 4h/1d buckets — no synthetic bars;
4. a mid-history delisting (survivorship, finding 1): the delisted instrument enters the
   universe while alive, its membership interval closes at ``delisted_ts`` (finding 21:
   rebalance facts only — no peeking), and its bars stay readable forever;
5. ``membership_asof`` + ``PITDataReader`` compose: a backtest-style read at decision
   time ``T`` sees exactly the members and bars knowable at ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pytest

from alphaforge.config.settings import UniverseCfg
from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.time import Timeframe
from alphaforge.core.types import AssetClass, MarketType
from alphaforge.data.quality.checks import QualityRunStats, apply_flags
from alphaforge.data.schemas import Dataset, QualityFlag
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.reader import PITDataReader
from alphaforge.data.store.resample import Resampler, ResampleStats
from alphaforge.data.store.writer import LakeWriter
from alphaforge.data.universe.builder import RebuildStats, UniverseBuilder
from alphaforge.data.universe.store import UniverseStore

HOUR = 3_600_000
DAY = 86_400_000
T_JAN = 1_704_067_200_000  # 2024-01-01T00:00:00Z — first bar open
T_FEB = 1_706_745_600_000  # 2024-02-01T00:00:00Z — rebalance 1
T_MAR = 1_709_251_200_000  # 2024-03-01T00:00:00Z — rebalance 2 / end of data
FAR_FUTURE = 1_900_000_000_000  # "now" for the historical fixture: everything closed
LISTED = T_JAN - 60 * DAY  # comfortably past the 30d min listing age at T_FEB

T_BP = T_JAN + 9 * DAY + 5 * HOUR  # Jan 10 05:00 — planted bad print on AAA
GAP_OPENS = [T_FEB + 9 * DAY + 5 * HOUR, T_FEB + 9 * DAY + 6 * HOUR]  # Feb 10 05-06 (BBB)
GAP_BEFORE = T_FEB + 9 * DAY + 4 * HOUR  # surviving bar before the gap
GAP_AFTER = T_FEB + 9 * DAY + 7 * HOUR  # surviving bar after the gap
GAP_BUCKET_4H = T_FEB + 9 * DAY + 4 * HOUR  # the 4h bucket broken by the gap
GAP_BUCKET_1D = T_FEB + 9 * DAY  # the 1d bucket broken by the gap
DELIST = T_FEB + 14 * DAY  # Feb 15 00:00 — FFF delists mid-history
T_DEC = T_FEB + 9 * DAY  # Feb 10 00:00 — backtest decision instant

_TS = pa.timestamp("ms", tz="UTC")


def iid(base: str) -> str:
    return f"BINANCE:PERP:{base}USDT"


A, B, C, D, E, F = (iid(b) for b in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"))
ALL_IDS = [A, B, C, D, E, F]
# Per-hour quote volume → liquidity ranks. Feb 1 (Jan window): A1 F2 B3 C4 D5 E6.
# Mar 1 (Feb window, FFF delisted): A1 B2 C3 D4 E5.
QUOTE_VOLUME = {A: 600.0, B: 500.0, C: 400.0, D: 300.0, E: 200.0, F: 550.0}
CFG = UniverseCfg(size=4, rank_window_days=30, rebalance="monthly", entry_rank=4, exit_rank=5)


def inst(instrument_id: str, *, delisted: int | None = None) -> Instrument:
    base = instrument_id.split(":")[2].removesuffix("USDT")
    return Instrument(
        instrument_id=instrument_id,
        asset_class=AssetClass.CRYPTO_PERP,
        market_type=MarketType.PERP,
        base=base,
        quote="USDT",
        tick_size=0.01,
        lot_size=0.001,
        min_qty=0.001,
        min_notional=5.0,
        can_short=True,
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
        funding_interval_hours=8,
        listed_ts=LISTED,
        delisted_ts=delisted,
    )


def _bars(instrument_id: str, *, end: int, drop: list[int] | None = None) -> pa.Table:
    """Hourly bars ``[T_JAN, end)``: closes alternate 100.0/100.1 (tiny but nonzero
    trailing sigma), constant volume 10; AAA carries the planted bad print at T_BP
    (close 130 on volume 0.5, immediately reverting). ``drop`` removes opens (a gap)."""
    dropped = set(drop or [])
    opens: list[int] = []
    closes: list[float] = []
    volumes: list[float] = []
    k = 0
    ts = T_JAN
    while ts < end:
        if ts not in dropped:
            opens.append(ts)
            if instrument_id == A and ts == T_BP:
                closes.append(130.0)
                volumes.append(0.5)
            else:
                closes.append(100.0 if k % 2 == 0 else 100.1)
                volumes.append(10.0)
        k += 1
        ts += HOUR
    open_ = [100.0, *closes[:-1]]
    n = len(opens)
    qv = QUOTE_VOLUME[instrument_id]
    return pa.table(
        {
            "instrument_id": pa.array([instrument_id] * n, type=pa.string()),
            "ts_open": pa.array(opens, type=_TS),
            "open": pa.array(open_, type=pa.float64()),
            "high": pa.array([max(o, c) for o, c in zip(open_, closes, strict=True)]),
            "low": pa.array([min(o, c) for o, c in zip(open_, closes, strict=True)]),
            "close": pa.array(closes, type=pa.float64()),
            "volume": pa.array(volumes, type=pa.float64()),
            "quote_volume": pa.array([qv] * n, type=pa.float64()),
            "n_trades": pa.array([1] * n, type=pa.int64()),
            "quality_flags": pa.array([0] * n, type=pa.int32()),
            "ingested_at": pa.array([o + HOUR + 30_000 for o in opens], type=_TS),
        }
    )


@dataclass(frozen=True)
class Env:
    reader: PITDataReader
    ustore: UniverseStore
    quality_stats: QualityRunStats
    h4_stats: ResampleStats
    d1_stats: ResampleStats
    rebuild_stats: RebuildStats
    pre_quality: dict[str, pa.Table]  # full-history bars per instrument, pre-flagging


def _ints(tbl: pa.Table, col: str) -> list[int]:
    return tbl.column(col).cast(pa.int64()).to_pylist()  # type: ignore[no-any-return]


def _flags_at(env: Env, instrument_id: str, ts_open: int, *, as_of: int) -> int:
    tbl = env.reader.ohlcv([instrument_id], start=ts_open, end=ts_open + HOUR, as_of=as_of)
    assert tbl.num_rows == 1
    return int(tbl.column("quality_flags")[0].as_py())


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Env:
    """Build the lake, run the full pipeline once, and freeze the results."""
    root: Path = tmp_path_factory.mktemp("integrity")
    paths = LakePaths(root / "lake")
    writer = LakeWriter(paths)
    reader = PITDataReader(paths)

    instruments = InstrumentStore(root / "ops.sqlite")
    for i in [inst(x) for x in (A, B, C, D, E)] + [inst(F, delisted=DELIST)]:
        instruments.upsert(i, as_of=i.listed_ts)

    for x in (A, C, D, E):
        writer.write(Dataset.OHLCV, _bars(x, end=T_MAR), now=FAR_FUTURE)
    writer.write(Dataset.OHLCV, _bars(B, end=T_MAR, drop=GAP_OPENS), now=FAR_FUTURE)
    writer.write(Dataset.OHLCV, _bars(F, end=DELIST), now=FAR_FUTURE)

    pre = {x: reader.ohlcv([x], start=0, end=FAR_FUTURE, as_of=FAR_FUTURE) for x in ALL_IDS}

    quality_stats = apply_flags(reader, writer, ALL_IDS, tf=Timeframe.H1, now=FAR_FUTURE)
    resampler = Resampler(paths)
    h4 = resampler.resample(ALL_IDS, target=Timeframe.H4, now=FAR_FUTURE)
    d1 = resampler.resample(ALL_IDS, target=Timeframe.D1, now=FAR_FUTURE)

    ustore = UniverseStore(paths)
    builder = UniverseBuilder(reader, instruments, ustore, CFG)
    rebuild = builder.rebuild(start=T_FEB, end=T_MAR, now=FAR_FUTURE)

    return Env(
        reader=reader,
        ustore=ustore,
        quality_stats=quality_stats,
        h4_stats=h4,
        d1_stats=d1,
        rebuild_stats=rebuild,
        pre_quality=pre,
    )


# ------------------------------------------------------------------ 1. quality flags


def test_planted_bad_print_is_flagged(env: Env) -> None:
    flags = _flags_at(env, A, T_BP, as_of=FAR_FUTURE)
    assert flags & int(QualityFlag.OUTLIER_RETURN)
    assert flags & int(QualityFlag.BAD_PRINT_SUSPECT)
    kinds = {f["kind"] for f in env.quality_stats.findings}
    assert {"gap", "outlier_return", "bad_print"} <= kinds
    assert env.quality_stats.rows_reflagged >= 3  # bad print + two gap survivors


def test_flags_annotate_prices_never_altered(env: Env) -> None:
    """Every price/volume column round-trips bit-for-bit through apply_flags."""
    for x in ALL_IDS:
        post = env.reader.ohlcv([x], start=0, end=FAR_FUTURE, as_of=FAR_FUTURE)
        pre = env.pre_quality[x]
        assert post.num_rows == pre.num_rows
        for col in ("ts_open", "open", "high", "low", "close", "volume", "quote_volume"):
            assert post.column(col).to_pylist() == pre.column(col).to_pylist(), (x, col)


# ----------------------------------------------------- 2. PIT masking (finding 5)


def test_pit_reader_masks_lag2_bad_print_bit(env: Env) -> None:
    """OUTLIER_RETURN (lag 0) is visible at the bar's close; BAD_PRINT_SUSPECT (lag 2,
    encodes t+1 information) stays masked until ts_open + 3Δ — never earlier."""
    at_close = _flags_at(env, A, T_BP, as_of=T_BP + HOUR)
    assert at_close & int(QualityFlag.OUTLIER_RETURN)
    assert not at_close & int(QualityFlag.BAD_PRINT_SUSPECT)

    one_early = _flags_at(env, A, T_BP, as_of=T_BP + 3 * HOUR - 1)
    assert not one_early & int(QualityFlag.BAD_PRINT_SUSPECT)

    visible = _flags_at(env, A, T_BP, as_of=T_BP + 3 * HOUR)
    assert visible & int(QualityFlag.BAD_PRINT_SUSPECT)
    assert visible & int(QualityFlag.OUTLIER_RETURN)


# ------------------------------------------------- 3. gaps: reported, never filled


def test_gap_reported_and_survivors_flagged(env: Env) -> None:
    day_end = GAP_BUCKET_1D + DAY
    assert env.reader.gaps(B, start=GAP_BUCKET_1D, end=day_end) == GAP_OPENS
    before = _flags_at(env, B, GAP_BEFORE, as_of=FAR_FUTURE)
    after = _flags_at(env, B, GAP_AFTER, as_of=FAR_FUTURE)
    assert before & int(QualityFlag.GAP_ADJACENT)
    assert after & int(QualityFlag.GAP_ADJACENT)
    # 2/6 instruments missing the hour is a per-instrument gap, not exchange downtime.
    assert not (before | after) & int(QualityFlag.EXCHANGE_DOWNTIME)


def test_gap_propagates_to_missing_derived_buckets(env: Env) -> None:
    """The broken 4h and 1d buckets are MISSING — never synthesized from partial data."""
    day_end = GAP_BUCKET_1D + DAY
    assert env.reader.gaps(B, start=GAP_BUCKET_1D, end=day_end, tf=Timeframe.H4) == [GAP_BUCKET_4H]
    assert env.reader.gaps(B, start=GAP_BUCKET_1D - DAY, end=day_end + DAY, tf=Timeframe.D1) == [
        GAP_BUCKET_1D
    ]
    assert env.h4_stats.buckets_incomplete == 1  # exactly BBB's broken bucket
    assert env.d1_stats.buckets_incomplete == 1
    # Complete instruments resampled the full span: 60 days for AAA, 45 for FFF.
    derived_a = env.reader.ohlcv([A], start=0, end=FAR_FUTURE, as_of=FAR_FUTURE, tf=Timeframe.D1)
    assert derived_a.num_rows == 60
    derived_f = env.reader.ohlcv([F], start=0, end=FAR_FUTURE, as_of=FAR_FUTURE, tf=Timeframe.D1)
    assert _ints(derived_f, "ts_open") == [T_JAN + k * DAY for k in range(45)]


def test_resampled_bar_aggregates_and_inherits_flags(env: Env) -> None:
    bucket = T_JAN + 4 * DAY  # Jan 5 00:00, an arbitrary clean 4h bucket of AAA
    src = env.reader.ohlcv([A], start=bucket, end=bucket + 4 * HOUR, as_of=FAR_FUTURE)
    h4 = env.reader.ohlcv(
        [A], start=bucket, end=bucket + 4 * HOUR, as_of=FAR_FUTURE, tf=Timeframe.H4
    )
    assert h4.num_rows == 1
    assert h4.column("open")[0].as_py() == src.column("open")[0].as_py()
    assert h4.column("close")[0].as_py() == src.column("close")[-1].as_py()
    assert h4.column("high")[0].as_py() == max(src.column("high").to_pylist())
    assert h4.column("low")[0].as_py() == min(src.column("low").to_pylist())
    assert h4.column("volume")[0].as_py() == pytest.approx(sum(src.column("volume").to_pylist()))
    # Flags OR up: the 4h bucket containing the bad print carries its bits.
    bp_bucket = T_BP - T_BP % (4 * HOUR)
    h4_bp = env.reader.ohlcv(
        [A], start=bp_bucket, end=bp_bucket + 4 * HOUR, as_of=FAR_FUTURE, tf=Timeframe.H4
    )
    bp_flags = int(h4_bp.column("quality_flags")[0].as_py())
    assert bp_flags & int(QualityFlag.OUTLIER_RETURN)
    assert bp_flags & int(QualityFlag.BAD_PRINT_SUSPECT)


# --------------------------------------- 4. universe: survivorship + delisting exit


def test_universe_intervals_close_at_delisted_ts(env: Env) -> None:
    assert env.rebuild_stats.rebalances == 2  # Feb 1 and Mar 1
    tbl = env.ustore.read_intervals()
    rows = {
        r_id: (r_from, r_to)
        for r_id, r_from, r_to in zip(
            tbl.column("instrument_id").to_pylist(),
            _ints(tbl, "effective_from"),
            tbl.column("effective_to").cast(pa.int64()).to_pylist(),
            strict=True,
        )
    }
    # FFF entered while alive and its interval closes at delisted_ts, not month-end.
    assert rows[F] == (T_FEB, DELIST)
    for member in (A, B, C):
        assert rows[member] == (T_FEB, None)
    assert rows[D] == (T_MAR, None)  # took FFF's slot at the next rebalance
    assert E not in rows  # rank 5-6, never inside entry_rank=4


def test_membership_asof_half_open_semantics(env: Env) -> None:
    assert env.ustore.membership_asof(T_DEC) == frozenset({A, B, C, F})
    assert env.ustore.membership_asof(DELIST - 1) >= frozenset({F})
    assert F not in env.ustore.membership_asof(DELIST)  # non-member at the close instant
    assert env.ustore.membership_asof(T_MAR) == frozenset({A, B, C, D})


# --------------------------------------------------- 5. backtest-style composition


def test_backtest_read_sees_only_what_was_knowable(env: Env) -> None:
    """membership_asof(T) x PITDataReader(as_of=T): exactly the members and bars at T."""
    members = sorted(env.ustore.membership_asof(T_DEC))
    assert members == sorted([A, B, C, F])
    bars = env.reader.ohlcv(members, start=T_DEC - 2 * DAY, end=T_DEC + DAY, as_of=T_DEC)
    opens = _ints(bars, "ts_open")
    assert max(opens) == T_DEC - HOUR  # last knowable bar closed exactly at T
    assert set(bars.column("instrument_id").to_pylist()) == set(members)
    assert len(opens) == 4 * 48  # 48 fully-closed hourly bars per member, no gaps here
    # The delisted instrument's history remains readable forever (survivorship-free).
    f_history = env.reader.ohlcv([F], start=0, end=FAR_FUTURE, as_of=FAR_FUTURE)
    assert f_history.num_rows == 45 * 24
    assert max(_ints(f_history, "ts_open")) == DELIST - HOUR
