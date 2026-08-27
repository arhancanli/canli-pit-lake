"""UniverseBuilder — PIT liquidity-ranked universe with entry/exit hysteresis.

Implements dataDesign.md §6 with the leakage-critique fixes baked in:

- **PIT by construction** (finding 1, §6.2): every rebalance decision at instant ``T``
  is computed from a :class:`PITDataReader` read with ``as_of=T`` — only bars whose
  close is at or before ``T`` can influence membership at ``T``. Volume printed after
  ``T`` is structurally invisible; peeking is impossible, not merely discouraged.
- **Survivorship-bias-free**: candidates come from the SCD2 instrument record (seeded
  from the historical archive, finding 1), so coins delisted years ago are ranked
  while they lived and their membership intervals exist in history. A delisting
  *between* rebalances closes the interval at ``delisted_ts`` — not at month-end.
- **Live/backtest rebalance ordering** (finding 21): membership ``effective_from`` is
  the rebalance instant ``T`` *exactly*. CONTRACT FOR THE LIVE LOOP: each cycle must
  update/refresh the universe BEFORE computing signals — on the 1st of the month at
  00:00 UTC the new membership is already in force at the 00:00 decision, exactly as
  the backtest assumes. A loop that ranks signals first would trade the stale
  universe for one cycle and silently diverge from research.
- **Deterministic**: a pure function of (lake, instrument record, config). Rank ties
  are broken by ``instrument_id`` ascending; iteration orders are sorted; rebuilding
  twice yields byte-identical intervals. Rebuilds are full-replace through
  :class:`UniverseStore` because the table is derived data — the lake is the source
  of truth and regeneration must not leave stale intervals behind.

Ranking signal at rebalance ``T`` (dataDesign.md §6.1): the trailing
``rank_window_days``-day **median of daily quote volume**, from ``rank_tf`` bars with
``ts_open in [T - window, T)`` available at ``T``, summed per UTC day (per-bar
fallback ``volume * close`` where ``quote_volume`` is null). Median over ~30 days is
robust to single-day listing/news spikes. Raw bars are consumed as stored — quality
flags annotate, nothing is altered or filled here.

``rank_tf`` selects the source timeframe of the ranking read (EQUITIES_SLEEVE.md §5):

- **Crypto** (default ``Timeframe.H1``): many intraday bars per UTC day are summed into
  a daily quote volume, then the trailing median is taken — exactly dataDesign.md §6.1.
- **Equities** (``Timeframe.D1``): the native bar IS daily and ``quote_volume`` is
  already that session's dollar volume, so the per-UTC-day summing collapses to a
  pass-through (one bar per session) and the trailing median over the window IS the
  classic **ADV (average dollar volume)** ranking signal. Pass ``D1`` and feed the
  builder an equity SCD2 record (including delisted names — §6.2 survivorship); no
  other change. The PIT rule is identical: a session bar for date ``D`` (labeled
  ``ts_open`` = ``D`` 00:00 UTC) is available at ``D + tf.ms`` and so a bar straddling
  the 00:00-on-the-1st rebalance instant ``T`` is correctly invisible at ``T`` — the
  equity window sees the sessions through the prior day, as a trader at ``T`` would.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import pyarrow as pa

from alphaforge.core.errors import ConfigError
from alphaforge.core.time import Ms, Timeframe, from_ms, to_ms

if TYPE_CHECKING:
    from alphaforge.config.settings import UniverseCfg
    from alphaforge.core.instruments import Instrument, InstrumentStore
    from alphaforge.core.types import AssetClass
    from alphaforge.data.store.reader import PITDataReader
    from alphaforge.data.universe.store import UniverseStore

__all__ = ["DEFAULT_MIN_DAYS", "DEFAULT_MIN_LISTING_AGE_MS", "RebuildStats", "UniverseBuilder"]

_DAY_MS: Final[int] = 86_400_000

DEFAULT_MIN_DAYS: Final[int] = 20
"""Minimum distinct UTC days with volume inside the ranking window (excludes zombie
listings whose sparse prints would make a 30d median meaningless). Calibrated for the
24/7 crypto sleeve, where a ``rank_window_days``-day window spans that many distinct UTC
days. See :data:`DEFAULT_MIN_DAYS_D1` for the equity (session-calendar) default."""

DEFAULT_MIN_DAYS_D1: Final[int] = 15
"""Minimum distinct session days for a D1 (equity) sleeve. A 30-CALENDAR-day window holds
only ~19-23 XNYS SESSIONS (not 30), so the crypto default of 20 would drop EVERY name in a
19-session month and collapse the universe to empty. 15 (~70% of a ~21-session window)
keeps fully-liquid names while still excluding sparse/halted zombies. The window itself
(``rank_window_days`` calendar days) is unchanged — only the distinct-day floor adapts to
the trading calendar."""

DEFAULT_MIN_LISTING_AGE_MS: Final[Ms] = 30 * _DAY_MS
"""Minimum age since ``listed_ts`` at the rebalance instant — no week-old listings
whose hype volume has no stable history behind it."""

_REASON_DELISTED: Final[str] = "delisted"
_REASON_INELIGIBLE: Final[str] = "ineligible"


@dataclass(frozen=True)
class RebuildStats:
    """Accounting for one :meth:`UniverseBuilder.rebuild` call.

    - ``rebalances``: monthly rebalance instants actually processed.
    - ``intervals``: total membership interval rows written (closed + open).
    - ``entries``: intervals opened (an instrument re-entering counts again).
    - ``exits``: intervals closed (rank exit, eligibility loss, or delisting).
    - ``open_members``: members still open (``effective_to`` null) at the end.
    """

    rebalances: int
    intervals: int
    entries: int
    exits: int
    open_members: int


@dataclass
class _Span:
    """Mutable in-progress membership interval (internal to the rebuild walk)."""

    instrument_id: str
    effective_from: Ms
    effective_to: Ms | None
    rank: int  # liquidity rank at ENTRY (UNIVERSE_SCHEMA contract)
    reason: str


class UniverseBuilder:
    """Monthly PIT universe builder (see module docstring for the full contract).

    Hysteresis (``UniverseCfg``): a non-member ENTERS when its rank is
    ``<= entry_rank``; an existing member STAYS while its rank is ``<= exit_rank``
    (``entry_rank < exit_rank``, validated by the config) — the band dampens
    membership churn from rank noise around the cutoff. Leaving the eligible set
    (delisting, listing-age failure, insufficient volume days) forces an exit
    regardless of the band.

    Eligibility at rebalance ``T``: ``listed_ts <= T - min_listing_age_ms``,
    ``delisted_ts`` null or ``> T``, and at least ``min_days`` distinct UTC days of
    volume inside the ranking window. Lifecycle facts (``listed_ts``/``delisted_ts``)
    are read from the SCD2 record as of ``now`` — they are historical events that a
    trader at ``T`` had already observed when they predate ``T``, and events after
    ``T`` cannot change eligibility *at* ``T`` (the predicate only asks whether
    ``delisted_ts > T``), so this is PIT-safe.
    """

    def __init__(
        self,
        reader: PITDataReader,
        instruments: InstrumentStore,
        store: UniverseStore,
        cfg: UniverseCfg,
        *,
        min_days: int | None = None,
        min_listing_age_ms: Ms = DEFAULT_MIN_LISTING_AGE_MS,
        rank_tf: Timeframe = Timeframe.H1,
        asset_class: AssetClass | None = None,
    ) -> None:
        """Raises :class:`ConfigError` unless ``cfg.rebalance == "monthly"`` — the v1
        schedule is fixed at 00:00 UTC on the 1st (dataDesign.md §6.1).

        ``rank_tf`` is the source timeframe for the liquidity read (see module
        docstring): ``Timeframe.H1`` for crypto (intraday bars summed per UTC day),
        ``Timeframe.D1`` for equities (native daily bars → ADV pass-through).

        ``asset_class`` restricts the candidate set to ONE sleeve. The SCD2 store holds
        every sleeve (crypto perps AND equities), and crypto perps carry resampled D1 bars
        too, so an equity (D1) rank that did not filter would rank crypto perps INTO the
        equity universe. ``None`` (the default) ranks every known instrument — byte-identical
        to the pre-sleeve behavior on a single-sleeve store; the CLI passes the profile's
        ``asset_class`` so each sleeve's universe contains only its own names.

        ``min_days`` (distinct-day floor inside the window) defaults per sleeve when left
        ``None``: :data:`DEFAULT_MIN_DAYS` (20) for H1 crypto, :data:`DEFAULT_MIN_DAYS_D1`
        (15) for D1 equities — a 30-calendar-day window holds ~30 distinct UTC days on the
        24/7 grid but only ~19-23 XNYS sessions, so the crypto floor would empty the equity
        universe in a short month. An explicit value overrides the per-sleeve default.
        """
        if cfg.rebalance != "monthly":
            raise ConfigError(
                f"UniverseBuilder v1 supports monthly rebalance only, got {cfg.rebalance!r}"
            )
        resolved_min_days = (
            min_days
            if min_days is not None
            else (DEFAULT_MIN_DAYS_D1 if rank_tf is Timeframe.D1 else DEFAULT_MIN_DAYS)
        )
        if resolved_min_days <= 0:
            raise ConfigError(f"min_days must be > 0, got {resolved_min_days}")
        if min_listing_age_ms < 0:
            raise ConfigError(f"min_listing_age_ms must be >= 0, got {min_listing_age_ms}")
        self._reader = reader
        self._instruments = instruments
        self._store = store
        self._cfg = cfg
        self._min_days = resolved_min_days
        self._min_listing_age_ms = min_listing_age_ms
        self._rank_tf = rank_tf
        self._asset_class = asset_class

    def rebuild(self, *, start: Ms, end: Ms, now: Ms) -> RebuildStats:
        """Regenerate the full membership history over ``[start, end]`` (epoch ms UTC).

        Walks every monthly rebalance instant in ``[start, end]`` (skipping instants
        after ``now`` — a rebalance cannot be taken before its time), reading the
        lake PIT at each instant, and writes the resulting interval rows through
        :meth:`UniverseStore.write_intervals` with full-replace semantics (derived
        data; safe and necessary — see module docstring). Deterministic: rebuilding
        twice yields identical intervals.

        Raises :class:`ValueError` if ``end < start``.
        """
        if end < start:
            raise ValueError(f"end ({end}) must be >= start ({start})")
        instants = [t for t in _monthly_rebalances(start, end) if t <= now]
        lifecycle = {
            i.instrument_id: i
            for i in self._instruments.all_known(as_of=now)
            if self._asset_class is None or i.asset_class is self._asset_class
        }

        open_members: dict[str, _Span] = {}
        closed: list[_Span] = []
        entries = 0
        exits = 0

        for t in instants:
            # Delistings since the previous rebalance close membership at delisted_ts,
            # not at month-end — the live loop would have flattened the book that day.
            exits += self._close_delisted(open_members, closed, lifecycle, until=t)
            ranked = self._rank(lifecycle, t)
            rank_of = dict(ranked)
            for iid in sorted(open_members):
                rank = rank_of.get(iid)
                if rank is None:
                    self._close(open_members, closed, iid, at=t, reason=_REASON_INELIGIBLE)
                    exits += 1
                elif rank > self._cfg.exit_rank:
                    self._close(open_members, closed, iid, at=t, reason=f"exit_rank_{rank}")
                    exits += 1
            for iid, rank in ranked:
                if rank <= self._cfg.entry_rank and iid not in open_members:
                    open_members[iid] = _Span(
                        instrument_id=iid,
                        effective_from=t,
                        effective_to=None,
                        rank=rank,
                        reason=f"enter_top{self._cfg.entry_rank}",
                    )
                    entries += 1

        # Delistings after the final rebalance but inside the rebuild horizon.
        exits += self._close_delisted(open_members, closed, lifecycle, until=end)

        spans = sorted(
            closed + list(open_members.values()),
            key=lambda s: (s.instrument_id, s.effective_from),
        )
        # Per-sleeve replace when this builder is scoped to one asset class: clear only
        # THIS sleeve's partitions (every known instrument_id of the class, incl. delisted)
        # so a rebuild of one sleeve never wipes the other — the two universes coexist in
        # one lake for the unified cross-asset book. ``asset_class=None`` keeps the legacy
        # full-replace (single-sleeve store; byte-identical to pre-sleeve behavior).
        scope_ids = frozenset(lifecycle) if self._asset_class is not None else None
        self._store.write_intervals(_spans_table(spans), replace=True, scope_ids=scope_ids)
        return RebuildStats(
            rebalances=len(instants),
            intervals=len(spans),
            entries=entries,
            exits=exits,
            open_members=len(open_members),
        )

    def members_asof(self, as_of: Ms) -> list[str]:
        """Research helper: stored membership at ``as_of``, sorted instrument ids.

        Reads the persisted intervals (half-open ``[effective_from, effective_to)``);
        inherently PIT because every span was computable at its ``effective_from``.
        """
        return sorted(self._store.membership_asof(as_of))

    def current(self, as_of: Ms) -> list[str]:
        """Live-loop helper: the universe in force at decision instant ``as_of``.

        Identical read path to :meth:`members_asof` — backtest and live consult the
        same intervals, which is the point. ORDERING CONTRACT (leakage finding 21):
        the live cycle must refresh membership (rebuild/update for a due rebalance or
        an observed delisting) BEFORE calling this and computing signals; membership
        flips at the rebalance instant exactly, as the backtest assumes.
        """
        return self.members_asof(as_of)

    # ------------------------------------------------------------------ internals

    def _close(
        self,
        open_members: dict[str, _Span],
        closed: list[_Span],
        iid: str,
        *,
        at: Ms,
        reason: str,
    ) -> None:
        span = open_members.pop(iid)
        span.effective_to = at
        span.reason = reason
        closed.append(span)

    def _close_delisted(
        self,
        open_members: dict[str, _Span],
        closed: list[_Span],
        lifecycle: dict[str, Instrument],
        *,
        until: Ms,
    ) -> int:
        """Close every open member whose ``delisted_ts <= until`` at its ``delisted_ts``."""
        n = 0
        for iid in sorted(open_members):
            inst = lifecycle.get(iid)
            if inst is not None and inst.delisted_ts is not None and inst.delisted_ts <= until:
                self._close(open_members, closed, iid, at=inst.delisted_ts, reason=_REASON_DELISTED)
                n += 1
        return n

    def _rank(self, lifecycle: dict[str, Instrument], t: Ms) -> list[tuple[str, int]]:
        """Liquidity ranking at rebalance instant ``t``: ``[(instrument_id, rank)]``.

        Rank 1 = most liquid. Only eligible instruments appear (lifecycle filters
        plus ``min_days`` of volume). Ties on the median-volume signal are broken by
        ``instrument_id`` ascending — stable and deterministic.
        """
        eligible = [
            iid
            for iid, inst in sorted(lifecycle.items())
            if inst.listed_ts <= t - self._min_listing_age_ms
            and (inst.delisted_ts is None or inst.delisted_ts > t)
        ]
        signal = self._median_daily_quote_volume(eligible, t)
        ordered = sorted(signal.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(iid, rank) for rank, (iid, _) in enumerate(ordered, start=1)]

    def _median_daily_quote_volume(self, instrument_ids: list[str], t: Ms) -> dict[str, float]:
        """Trailing median of daily quote volume per instrument, PIT at ``t``.

        Reads ``rank_tf`` bars with ``ts_open`` in ``[t - window, t)`` and ``as_of=t``
        — the reader's PIT rule guarantees every consumed bar closed at or before
        ``t``. Per bar, quote volume is the stored ``quote_volume`` or the
        ``volume * close`` fallback when null (dataDesign.md §6.1); bars are summed per
        UTC day and instruments with fewer than ``min_days`` distinct days are dropped
        (ineligible). For an equity ``D1`` read the per-day sum is a one-bar
        pass-through, so the trailing median is the average daily dollar volume (ADV).
        """
        if not instrument_ids:
            return {}
        window_start = t - self._cfg.rank_window_days * _DAY_MS
        bars = self._reader.ohlcv(
            instrument_ids, start=window_start, end=t, as_of=t, tf=self._rank_tf
        )
        ids: list[str] = bars.column("instrument_id").to_pylist()
        opens: list[int] = bars.column("ts_open").cast(pa.int64()).to_pylist()
        quote_vols: list[float | None] = bars.column("quote_volume").to_pylist()
        vols: list[float] = bars.column("volume").to_pylist()
        closes: list[float] = bars.column("close").to_pylist()

        daily: dict[str, dict[int, float]] = {}
        for iid, ts_open, qv, vol, close in zip(ids, opens, quote_vols, vols, closes, strict=True):
            day = ts_open // _DAY_MS
            value = qv if qv is not None else vol * close
            per_day = daily.setdefault(iid, {})
            per_day[day] = per_day.get(day, 0.0) + value
        return {
            iid: float(statistics.median(per_day.values()))
            for iid, per_day in daily.items()
            if len(per_day) >= self._min_days
        }


def _monthly_rebalances(start: Ms, end: Ms) -> list[Ms]:
    """All 00:00-UTC-on-the-1st instants in ``[start, end]``, ascending (exact ints)."""
    out: list[Ms] = []
    first = from_ms(start)
    year, month = first.year, first.month
    if to_ms(datetime(year, month, 1, tzinfo=UTC)) < start:
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    while True:
        t = to_ms(datetime(year, month, 1, tzinfo=UTC))
        if t > end:
            return out
        out.append(t)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def _spans_table(spans: list[_Span]) -> pa.Table:
    """Spans → a table conforming exactly to ``UNIVERSE_SCHEMA`` column contracts."""
    ts_type = pa.timestamp("ms", tz="UTC")
    return pa.table(
        {
            "instrument_id": pa.array([s.instrument_id for s in spans], type=pa.string()),
            "effective_from": pa.array([s.effective_from for s in spans], type=ts_type),
            "effective_to": pa.array([s.effective_to for s in spans], type=ts_type),
            "rank": pa.array([s.rank for s in spans], type=pa.int32()),
            "reason": pa.array([s.reason for s in spans], type=pa.string()),
        }
    )
