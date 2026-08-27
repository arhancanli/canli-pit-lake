"""Network integration tests for the Binance Vision archive client.

These hit the real ``data.binance.vision`` public S3 bucket and are therefore
deselected by default (``addopts = -m 'not network'``). Run explicitly with::

    uv run pytest tests/integration/test_vision_network.py -m network -q

Each test must finish in well under 30 seconds; the fixtures are scoped to the
module so the expensive full-catalog listing happens exactly once per run.

What they prove (and unit fixtures cannot):

* The S3 listing pagination really works against the live bucket (>500 symbols
  means multiple 1000-item pages of keys under the klines prefix).
* The historical catalog contains at least one famously delisted perp
  (FTTUSDT / SRMUSDT / LUNAUSDT) alongside BTCUSDT — the empirical heart of
  leakage finding 1: the archive sees what the live REST API has forgotten.
* A real, old (immutable, header-era-stable) BTCUSDT month parses and validates
  against ``OHLCV_SCHEMA`` with exact bar arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import httpx
import pytest

from alphaforge.core.time import Timeframe
from alphaforge.data.schemas import Dataset, validate_table
from alphaforge.data.sources.vision import BinanceVisionClient

pytestmark = pytest.mark.network

#: Perps Binance delisted years ago; absent from the live REST API, present forever
#: in the Vision archive. Any one sufficing keeps the test robust to bucket changes.
KNOWN_DELISTINGS = ("FTTUSDT", "SRMUSDT", "LUNAUSDT")

JAN_2021_FIRST_OPEN_MS = 1_609_459_200_000  # 2021-01-01T00:00:00Z
H1_MS = Timeframe.H1.ms


@pytest.fixture(scope="module")
def client() -> Iterator[BinanceVisionClient]:
    """A real-network client with a short timeout so failures stay under 30s."""
    with httpx.Client(timeout=20.0) as http:
        yield BinanceVisionClient(http=http)


@pytest.fixture(scope="module")
def symbols(client: BinanceVisionClient) -> list[str]:
    """The full historical USDT-perp catalog, listed once for the module."""
    return client.list_symbols()


class TestLiveCatalog:
    def test_catalog_is_large_and_clean(self, symbols: list[str]) -> None:
        """Pagination was followed (catalog >> one page of CommonPrefixes is not
        guaranteed, but >500 symbols proves multi-page key traversal upstream)
        and the v1 filters held: USDT-quoted, no dated delivery contracts."""
        assert len(symbols) > 300, f"suspiciously small catalog: {len(symbols)}"
        assert len(symbols) == len(set(symbols))
        assert symbols == sorted(symbols)
        assert all("_" not in s for s in symbols)
        assert all(s.endswith("USDT") for s in symbols)

    def test_contains_btcusdt(self, symbols: list[str]) -> None:
        assert "BTCUSDT" in symbols

    def test_contains_a_known_delisting(self, symbols: list[str]) -> None:
        """The survivorship-bias killer, empirically: at least one perp that the
        live REST API no longer serves must be present in the archive catalog."""
        present = [s for s in KNOWN_DELISTINGS if s in symbols]
        assert present, (
            f"none of {KNOWN_DELISTINGS} found in the Vision catalog — either the "
            "bucket layout changed or the listing/filtering logic regressed"
        )


class TestLiveFetch:
    def test_old_btcusdt_month_validates(self, client: BinanceVisionClient) -> None:
        """Fetch BTCUSDT 2021-01 (immutable history) and validate it end to end."""
        tbl = client.fetch_ohlcv_month("BTCUSDT", 2021, 1)
        validate_table(tbl, Dataset.OHLCV)

        # January 2021 is complete and long closed: exactly 31 * 24 hourly bars.
        assert tbl.num_rows == 31 * 24

        ts_open = tbl.column("ts_open").to_pylist()
        first, last = ts_open[0], ts_open[-1]
        assert first == datetime(2021, 1, 1, 0, tzinfo=UTC)
        assert last == datetime(2021, 1, 31, 23, tzinfo=UTC)
        # sorted, unique, gapless 1h grid
        opens_ms = [JAN_2021_FIRST_OPEN_MS + i * H1_MS for i in range(31 * 24)]
        assert [int(v.timestamp() * 1000) for v in ts_open] == opens_ms

        assert set(tbl.column("instrument_id").to_pylist()) == {"BINANCE:PERP:BTCUSDT"}
        # BTC traded ~$29k-$42k in Jan 2021 — sanity-bound the closes loosely.
        closes = tbl.column("close").to_pylist()
        assert all(20_000.0 < c < 50_000.0 for c in closes)
        assert all(v >= 0.0 for v in tbl.column("volume").to_pylist())
        assert all(n > 0 for n in tbl.column("n_trades").to_pylist())
        assert set(tbl.column("quality_flags").to_pylist()) == {0}
