"""End-to-end lake integration: BackfillJob → Parquet lake → PITDataReader.

The full Phase 2 chain on one merged lake, with zero network: a FakeSource (REST
double) and a FakeVision (archive double) feed BackfillJob.run over two instruments
across three calendar months — BTC stays LIVE (REST path) while LUNA is DELISTED
mid-range (Vision monthly path, survivorship fix per leakage finding 1). The merged
lake is then interrogated through the real PITDataReader:

(a) the PIT boundary holds exactly (a bar is invisible at ``close - 1ms``, visible
    at ``close``) on the merged multi-instrument lake;
(b) ``reader.gaps`` pinpoints a bar the venue deliberately never served — and the
    lake never synthesized;
(c) funding visibility is gated on the stored ``available_at`` (settlement +
    publication lag), never on ``ts_funding`` (leakage finding 18);
(d) re-running via ``BackfillJob.update`` adds zero new logical rows (idempotent
    resume: overlap re-fetch dedupes in the writer);
(e) checkpoint watermarks equal the max natural-key timestamp durably in the lake
    (for the delisted instrument: the documented convergence to ``delisted_ts - 1``,
    which covers its max key).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.time import Ms, Timeframe
from alphaforge.core.types import AssetClass, MarketType
from alphaforge.data.ingest.backfill import BackfillJob
from alphaforge.data.ingest.checkpoints import CheckpointStore
from alphaforge.data.schemas import Dataset
from alphaforge.data.sources.base import DataSource
from alphaforge.data.sources.ccxt_source import FUNDING_PUBLICATION_LAG_MS
from alphaforge.data.sources.vision import BinanceVisionClient, YearMonth
from alphaforge.data.store.lake import LakePaths
from alphaforge.data.store.reader import PITDataReader
from alphaforge.data.store.writer import LakeWriter

if TYPE_CHECKING:
    from pathlib import Path

HOUR = 3_600_000
DAY = 86_400_000
FUNDING_STEP = 8 * HOUR

JAN_01 = 1_704_067_200_000  # 2024-01-01T00:00:00Z (hour- and funding-grid aligned)
FEB_01 = 1_706_745_600_000  # 2024-02-01T00:00:00Z
MAR_01 = 1_709_251_200_000  # 2024-03-01T00:00:00Z
MAR_05 = MAR_01 + 4 * DAY  # 2024-03-05T00:00:00Z — LUNA delisting instant
APR_01 = 1_711_929_600_000  # 2024-04-01T00:00:00Z — "now" for the backfill

BTC = "BINANCE:PERP:BTCUSDT"
LUNA = "BINANCE:PERP:LUNAUSDT"

OMITTED_BAR = FEB_01 + 9 * DAY + 12 * HOUR  # 2024-02-10T12:00Z — venue never serves it


def make_inst(symbol: str, *, listed: Ms, delisted: Ms | None = None) -> Instrument:
    """A minimal valid USDT perp Instrument for tests."""
    return Instrument(
        instrument_id=f"BINANCE:PERP:{symbol}",
        asset_class=AssetClass.CRYPTO_PERP,
        market_type=MarketType.PERP,
        base=symbol.removesuffix("USDT"),
        quote="USDT",
        tick_size=0.1,
        lot_size=0.001,
        min_qty=0.001,
        min_notional=5.0,
        can_short=True,
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
        funding_interval_hours=8,
        listed_ts=listed,
        delisted_ts=delisted,
    )


def _grid(lo: Ms, hi: Ms, step: int) -> list[int]:
    """Grid timestamps t = k*step with t in [lo, hi) (lo/hi need not be aligned)."""
    first = -(-lo // step) * step
    return list(range(first, hi, step))


def ohlcv_table(instrument_id: str, ts_opens: list[int]) -> pa.Table:
    """OHLCV-schema table with one synthetic closed bar per ts_open."""
    n = len(ts_opens)
    return pa.table(
        {
            "instrument_id": pa.array([instrument_id] * n, type=pa.string()),
            "ts_open": pa.array(ts_opens, type=pa.timestamp("ms", tz="UTC")),
            "open": pa.array([99.0] * n, type=pa.float64()),
            "high": pa.array([101.0] * n, type=pa.float64()),
            "low": pa.array([98.0] * n, type=pa.float64()),
            "close": pa.array([float(100 + i % 7) for i in range(n)], type=pa.float64()),
            "volume": pa.array([10.0] * n, type=pa.float64()),
            "quote_volume": pa.array([1_000.0] * n, type=pa.float64()),
            "n_trades": pa.array([42] * n, type=pa.int64()),
            "quality_flags": pa.array([0] * n, type=pa.int32()),
            "ingested_at": pa.array(
                [t + HOUR + 30_000 for t in ts_opens], type=pa.timestamp("ms", tz="UTC")
            ),
        }
    )


def funding_table(instrument_id: str, ts_fundings: list[int]) -> pa.Table:
    """FUNDING-schema table with available_at = ts_funding + publication lag."""
    n = len(ts_fundings)
    return pa.table(
        {
            "instrument_id": pa.array([instrument_id] * n, type=pa.string()),
            "ts_funding": pa.array(ts_fundings, type=pa.timestamp("ms", tz="UTC")),
            "rate": pa.array([1e-4] * n, type=pa.float64()),
            "available_at": pa.array(
                [t + FUNDING_PUBLICATION_LAG_MS for t in ts_fundings],
                type=pa.timestamp("ms", tz="UTC"),
            ),
            "ingested_at": pa.array(
                [t + FUNDING_PUBLICATION_LAG_MS + 1_000 for t in ts_fundings],
                type=pa.timestamp("ms", tz="UTC"),
            ),
        }
    )


class FakeSource(DataSource):
    """In-memory REST double serving grid bars over per-id spans, minus omitted opens."""

    name = "fake"

    def __init__(
        self, data: dict[str, tuple[Ms, Ms]], *, omit: frozenset[Ms] = frozenset()
    ) -> None:
        self._data = data
        self._omit = omit

    def list_instruments(self, *, as_of: Ms) -> list[Instrument]:
        return []

    def fetch_ohlcv(
        self, instrument_id: str, *, tf: Timeframe, since: Ms, until: Ms, now: Ms | None = None
    ) -> pa.Table:
        lo, hi = self._data.get(instrument_id, (0, 0))
        cut = min(until, hi)
        if now is not None:
            cut = min(cut, now - tf.ms + 1)  # closed bars only: ts_open + Δ <= now
        opens = [t for t in _grid(max(since, lo), cut, tf.ms) if t not in self._omit]
        return ohlcv_table(instrument_id, opens)

    def fetch_funding(self, instrument_id: str, *, since: Ms, until: Ms) -> pa.Table:
        lo, hi = self._data.get(instrument_id, (0, 0))
        return funding_table(instrument_id, _grid(max(since, lo), min(until, hi), FUNDING_STEP))


class FakeVision(BinanceVisionClient):
    """In-memory archive double keyed by exchange symbol (whole-month files)."""

    def __init__(self, data: dict[str, tuple[Ms, Ms]]) -> None:
        super().__init__()
        self._fake_data = data

    def _month_span(self, symbol: str, year: int, month: int) -> tuple[Ms, Ms]:
        lo, hi = self._fake_data.get(symbol, (0, 0))
        ym = YearMonth(year, month)
        return max(lo, ym.first_ms()), min(hi, ym.next_month_first_ms())

    def fetch_ohlcv_month(
        self, symbol: str, year: int, month: int, *, now: Ms | None = None
    ) -> pa.Table:
        lo, hi = self._month_span(symbol, year, month)
        if now is not None:
            hi = min(hi, now - HOUR + 1)
        return ohlcv_table(f"BINANCE:PERP:{symbol}", _grid(lo, hi, HOUR))

    def fetch_funding_month(self, symbol: str, year: int, month: int) -> pa.Table:
        lo, hi = self._month_span(symbol, year, month)
        return funding_table(f"BINANCE:PERP:{symbol}", _grid(lo, hi, FUNDING_STEP))


class LakeEnv:
    """One backfilled lake (BTC live + LUNA delisted, Jan-Mar 2024) plus its reader."""

    def __init__(self, root: Path) -> None:
        self.paths = LakePaths(root / "lake")
        self.checkpoints = CheckpointStore(root / "ops.sqlite")
        self.store = InstrumentStore(root / "ops.sqlite")
        self.store.upsert(make_inst("BTCUSDT", listed=JAN_01), as_of=JAN_01)
        self.store.upsert(make_inst("LUNAUSDT", listed=JAN_01, delisted=MAR_05), as_of=JAN_01)
        self.job = BackfillJob(
            FakeSource({BTC: (JAN_01, APR_01 + 30 * DAY)}, omit=frozenset({OMITTED_BAR})),
            FakeVision({"LUNAUSDT": (JAN_01, MAR_05)}),
            LakeWriter(self.paths),
            self.checkpoints,
            self.store,
        )
        self.reader = PITDataReader(self.paths)

    def close(self) -> None:
        self.checkpoints.close()
        self.store.close()


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LakeEnv]:
    """Build the merged lake once; every test reads the same artifact."""
    lake_env = LakeEnv(tmp_path_factory.mktemp("e2e"))
    report = lake_env.job.run(
        datasets=[Dataset.OHLCV, Dataset.FUNDING], start_default=JAN_01, now=APR_01
    )
    assert report.failed_count == 0, report.summary()
    assert report.ok_count == 4  # 2 instruments x 2 datasets
    yield lake_env
    lake_env.close()


def _only(tbl: pa.Table, instrument_id: str) -> pa.Table:
    """Rows of ``tbl`` belonging to ``instrument_id``."""
    ids = tbl.column("instrument_id").to_pylist()
    return tbl.filter(pa.array([x == instrument_id for x in ids], type=pa.bool_()))


def _keys(tbl: pa.Table, column: str) -> list[int]:
    """Sorted epoch-ms values of a ``timestamp[ms, UTC]`` column of ``tbl``."""
    raw = tbl.column(column).to_numpy(zero_copy_only=False)
    return sorted(raw.astype("datetime64[ms]").astype("int64").tolist())


def _opens(tbl: pa.Table, instrument_id: str | None = None) -> list[int]:
    """Sorted epoch-ms ts_open values of ``tbl`` (optionally one instrument's)."""
    if instrument_id is not None:
        tbl = _only(tbl, instrument_id)
    return _keys(tbl, "ts_open")


class TestPITBoundary:
    """(a) The PIT rule holds exactly on the merged multi-instrument lake."""

    def test_bar_invisible_at_close_minus_1ms_visible_at_close(self, env: LakeEnv) -> None:
        last_open = APR_01 - HOUR  # the freshest closed BTC bar
        at_close = env.reader.ohlcv([BTC], start=JAN_01, end=APR_01, as_of=last_open + HOUR)
        before_close = env.reader.ohlcv([BTC], start=JAN_01, end=APR_01, as_of=last_open + HOUR - 1)
        assert _opens(at_close)[-1] == last_open
        assert _opens(before_close)[-1] == last_open - HOUR  # one bar less, exactly

    def test_merged_lake_respects_each_instruments_life(self, env: LakeEnv) -> None:
        tbl = env.reader.ohlcv([BTC, LUNA], start=JAN_01, end=APR_01, as_of=APR_01)
        btc_opens = _opens(tbl, BTC)
        luna_opens = _opens(tbl, LUNA)
        # BTC: full grid minus the omitted bar, last close exactly at as_of.
        assert btc_opens == [t for t in _grid(JAN_01, APR_01, HOUR) if t != OMITTED_BAR]
        # LUNA: nothing at/after its delisting instant, nothing synthesized.
        assert luna_opens == _grid(JAN_01, MAR_05, HOUR)
        assert luna_opens[-1] == MAR_05 - HOUR

    def test_as_of_mid_history_truncates_both_instruments(self, env: LakeEnv) -> None:
        as_of = FEB_01 + 12 * HOUR + 30 * 60_000  # unaligned mid-Feb instant
        tbl = env.reader.ohlcv([BTC, LUNA], start=JAN_01, end=APR_01, as_of=as_of)
        last_visible = FEB_01 + 11 * HOUR  # max ts_open with ts_open + 1h <= as_of
        assert _opens(tbl, BTC)[-1] == last_visible
        assert _opens(tbl, LUNA)[-1] == last_visible


class TestGapDetection:
    """(b) The intentionally omitted bar is reported as a gap, never filled."""

    def test_gaps_finds_exactly_the_omitted_bar(self, env: LakeEnv) -> None:
        assert env.reader.gaps(BTC, start=JAN_01, end=APR_01) == [OMITTED_BAR]

    def test_delisted_instrument_has_gapless_life(self, env: LakeEnv) -> None:
        assert env.reader.gaps(LUNA, start=JAN_01, end=MAR_05) == []

    def test_omitted_bar_never_served_by_reader(self, env: LakeEnv) -> None:
        tbl = env.reader.ohlcv([BTC], start=OMITTED_BAR, end=OMITTED_BAR + HOUR, as_of=APR_01)
        assert tbl.num_rows == 0


class TestFundingAvailability:
    """(c) Funding joins on the stored available_at, never on ts_funding."""

    def test_settlement_invisible_until_available_at(self, env: LakeEnv) -> None:
        settlement = FEB_01  # one mid-history funding event on the 8h grid
        available = settlement + FUNDING_PUBLICATION_LAG_MS
        visible = env.reader.funding([BTC], start=settlement, end=APR_01, as_of=available)
        hidden = env.reader.funding([BTC], start=settlement, end=APR_01, as_of=available - 1)
        visible_ts = visible.column("ts_funding").to_numpy(zero_copy_only=False)
        assert int(visible_ts.astype("datetime64[ms]").astype("int64").min()) == settlement
        assert hidden.num_rows == 0  # at as_of = available_at - 1ms it does not exist yet

    def test_full_funding_history_at_late_as_of(self, env: LakeEnv) -> None:
        tbl = env.reader.funding([BTC, LUNA], start=JAN_01, end=APR_01, as_of=APR_01)
        per_id = {
            BTC: _grid(JAN_01, APR_01, FUNDING_STEP),
            LUNA: _grid(JAN_01, MAR_05, FUNDING_STEP),
        }
        for iid, expected in per_id.items():
            assert _keys(_only(tbl, iid), "ts_funding") == expected


class TestIdempotentUpdate:
    """(d) Re-running via update() adds zero new logical rows."""

    def test_update_is_a_logical_noop(self, env: LakeEnv) -> None:
        before_ohlcv = env.reader.ohlcv([BTC, LUNA], start=JAN_01, end=APR_01, as_of=APR_01)
        before_funding = env.reader.funding([BTC, LUNA], start=JAN_01, end=APR_01, as_of=APR_01)
        marks_before = {
            ds: env.checkpoints.all_watermarks(ds) for ds in (Dataset.OHLCV, Dataset.FUNDING)
        }

        report = env.job.update(as_of=APR_01)

        assert report.failed_count == 0, report.summary()
        assert report.total_rows == 0  # nothing new at the same as_of
        after_ohlcv = env.reader.ohlcv([BTC, LUNA], start=JAN_01, end=APR_01, as_of=APR_01)
        after_funding = env.reader.funding([BTC, LUNA], start=JAN_01, end=APR_01, as_of=APR_01)
        # Logical content identical: same keys per instrument, same row counts.
        assert after_ohlcv.num_rows == before_ohlcv.num_rows
        assert after_funding.num_rows == before_funding.num_rows
        for iid in (BTC, LUNA):
            assert _opens(after_ohlcv, iid) == _opens(before_ohlcv, iid)
        assert env.reader.gaps(BTC, start=JAN_01, end=APR_01) == [OMITTED_BAR]
        marks_after = {
            ds: env.checkpoints.all_watermarks(ds) for ds in (Dataset.OHLCV, Dataset.FUNDING)
        }
        assert marks_after == marks_before


class TestWatermarks:
    """(e) Checkpoint watermarks equal the max natural-key timestamp in the lake."""

    def test_live_watermarks_equal_max_keys_in_lake(self, env: LakeEnv) -> None:
        ohlcv = env.reader.ohlcv([BTC], start=JAN_01, end=APR_01, as_of=APR_01)
        funding = env.reader.funding([BTC], start=JAN_01, end=APR_01, as_of=APR_01)
        assert env.checkpoints.get(Dataset.OHLCV, BTC) == max(_opens(ohlcv))
        fund_ts = funding.column("ts_funding").to_numpy(zero_copy_only=False)
        max_funding = int(fund_ts.astype("datetime64[ms]").astype("int64").max())
        assert env.checkpoints.get(Dataset.FUNDING, BTC) == max_funding
        assert env.checkpoints.get(Dataset.OHLCV, BTC) == APR_01 - HOUR
        assert env.checkpoints.get(Dataset.FUNDING, BTC) == APR_01 - FUNDING_STEP

    def test_delisted_watermarks_converged_and_cover_max_keys(self, env: LakeEnv) -> None:
        # A gap-free fully-walked delisted instrument converges to delisted_ts - 1
        # (documented in BackfillJob._ingest_vision) — which covers its max lake key,
        # so later runs skip it instead of re-polling the archive forever.
        ohlcv = env.reader.ohlcv([LUNA], start=JAN_01, end=APR_01, as_of=APR_01)
        wm_ohlcv = env.checkpoints.get(Dataset.OHLCV, LUNA)
        wm_funding = env.checkpoints.get(Dataset.FUNDING, LUNA)
        assert wm_ohlcv == MAR_05 - 1
        assert wm_funding == MAR_05 - 1
        assert wm_ohlcv >= max(_opens(ohlcv)) == MAR_05 - HOUR
