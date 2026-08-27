"""``DataSource`` — the asset-agnostic market-data vendor interface.

Implementations are dumb fetchers (dataDesign.md §4.1): they normalize a vendor's
payloads to the canonical lake schemas in :mod:`alphaforge.data.schemas` and NEVER
write to the lake themselves — persistence (dedupe, atomic replace, partial-bar
write guard) is the :class:`LakeWriter`'s job, and point-in-time filtering is the
reader's. Crypto (ccxt/Binance) ships first; an equities source implements the same
three methods against a session calendar without any engine changes.

Temporal contract (dataDesign.md §0, non-negotiable):

* All timestamps crossing this interface are integer epoch milliseconds, UTC
  (:data:`~alphaforge.core.time.Ms`).
* Bars are labeled by OPEN time (``ts_open``); a bar of timeframe Δ is available at
  ``ts_open + Δ`` (its close). Sources must never return an unclosed (partial) bar.
* All ranges are half-open: ``[since, until)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pyarrow as pa

from alphaforge.core.instruments import Instrument
from alphaforge.core.time import Ms, Timeframe

__all__ = ["DataSource"]


class DataSource(ABC):
    """A market-data vendor adapter normalizing to the canonical lake schemas.

    Subclasses set :attr:`name` (e.g. ``"ccxt.binanceusdm"``) — it is recorded in
    ingest watermarks and run logs so multiple vendors can coexist per dataset.

    Implementations must be deterministic given the vendor's responses: same remote
    data in, same table out (modulo ``ingested_at``, which is wall-clock audit
    metadata). They handle pagination, rate limiting, and transient-error retry
    internally; callers see one synchronous call per logical range.
    """

    #: Stable source identifier, e.g. ``"ccxt.binanceusdm"``. Used as the ``source``
    #: key in ingest watermarks and logs; must never change for a deployed source.
    #: Subclasses set it at class level or in ``__init__`` (e.g. derived from a
    #: configurable exchange id).
    name: str

    @abstractmethod
    def list_instruments(self, *, as_of: Ms) -> list[Instrument]:
        """Return instruments from the venue's CURRENT listing endpoint.

        .. warning:: **Survivorship-biased on its own** (leakageCritique.md finding
           1). This reflects what the exchange lists *now* (plus whatever residue of
           delisted markets the venue still reports): any instrument fully purged
           from the live endpoint before this system first ran never appears here,
           so a candidate universe seeded *only* from this method has already
           "survived" to the present. A historical listing catalog (e.g. Binance
           Vision dumps / archived ``exchangeInfo`` snapshots) must complement this
           method before any backtest universe predating go-live is trusted; until
           then such backtests are survivorship-biased by construction.

        Args:
            as_of: Observation timestamp (epoch ms, UTC). The snapshot is taken
                *at* this instant: it stamps SCD2 versioning upstream
                (``InstrumentStore.upsert(inst, as_of=...)``) and serves as the
                fallback ``delisted_ts`` for markets observed in a dead state with
                no better venue-provided timestamp. It does NOT time-travel the
                listing — the venue only ever reports its current state.

        Returns:
            Fully-populated :class:`~alphaforge.core.instruments.Instrument`
            records (exchange filters, fees, lifecycle timestamps), sorted by
            ``instrument_id``.
        """

    @abstractmethod
    def fetch_ohlcv(
        self,
        instrument_id: str,
        *,
        tf: Timeframe,
        since: Ms,
        until: Ms,
        now: Ms | None = None,
    ) -> pa.Table:
        """Fetch closed OHLCV bars with ``ts_open`` in ``[since, until)``.

        The returned table conforms exactly to
        :data:`~alphaforge.data.schemas.OHLCV_SCHEMA`, is sorted ascending and
        unique on ``ts_open``, and contains **only closed bars**: every row
        satisfies ``ts_open + tf.ms <= (now or wall clock)``. An in-progress bar
        the vendor happens to return is dropped here, before it can ever reach the
        writer (which guards independently).

        Args:
            instrument_id: Canonical id, e.g. ``"BINANCE:PERP:BTCUSDT"``.
            tf: Bar timeframe; bar duration Δ = ``tf.ms``.
            since: Inclusive lower bound on ``ts_open`` (epoch ms, UTC).
            until: Exclusive upper bound on ``ts_open`` (epoch ms, UTC).
            now: Injectable "current time" for the closed-bar cutoff (epoch ms,
                UTC). ``None`` means read the wall clock once via
                :func:`~alphaforge.core.time.now_ms`. Tests inject this; production
                callers leave it ``None``.

        Returns:
            A :class:`pyarrow.Table` in OHLCV schema (possibly empty).
        """

    @abstractmethod
    def fetch_funding(self, instrument_id: str, *, since: Ms, until: Ms) -> pa.Table:
        """Fetch settled funding records with ``ts_funding`` in ``[since, until)``.

        The returned table conforms exactly to
        :data:`~alphaforge.data.schemas.FUNDING_SCHEMA`, sorted ascending and
        unique on ``ts_funding``. Each row carries an explicit point-in-time
        ``available_at`` (settlement time plus the source's publication lag); ALL
        downstream consumers join on ``available_at``, never on ``ts_funding``
        (dataDesign.md §3.2, leakageCritique.md finding 18).

        Args:
            instrument_id: Canonical id of a perpetual, e.g.
                ``"BINANCE:PERP:BTCUSDT"``.
            since: Inclusive lower bound on ``ts_funding`` (epoch ms, UTC).
            until: Exclusive upper bound on ``ts_funding`` (epoch ms, UTC).

        Returns:
            A :class:`pyarrow.Table` in funding schema (possibly empty).
        """
