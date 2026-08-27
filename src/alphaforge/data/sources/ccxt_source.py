"""``CCXTDataSource`` — Binance USDT-margined perpetuals via ccxt raw endpoints.

Scope (dataDesign.md §4.2, buildabilityCritique.md Phase 2): **USDT-margined
perpetual swaps only.** Dated futures, USDC-margined contracts, and anything whose
ccxt market is not ``swap + linear + USDT-quoted`` is skipped at listing time.

Why raw (implicit) endpoints instead of ccxt's unified methods:

* ``fapiPublicGetKlines`` returns the full 12-element kline arrays, including quote
  volume (index 7) and trade count (index 8), which unified ``fetch_ohlcv`` drops —
  and exact quote volume is what the universe builder ranks on (dataDesign.md §6.1).
* ``fapiPublicGetFundingRate`` returns settled funding with explicit settlement
  timestamps, which we stamp with a conservative publication lag to form
  ``available_at`` (dataDesign.md §3.2; leakageCritique.md finding 18).
* ``fapiPublicGetFundingInfo`` carries per-symbol funding intervals — many alt perps
  fund every 4h (some 1h), not 8h (leakageCritique.md finding 6).

Reliability: every page is metered through an optional
:class:`~alphaforge.data.ingest.retry.RateBudget` and wrapped in
:func:`~alphaforge.data.ingest.retry.transient_retry` over ccxt's transient error
family. Permanent errors (``BadSymbol``, auth) propagate immediately.

All public methods take/return integer epoch milliseconds UTC (:data:`Ms`) and map
``instrument_id`` ↔ raw exchange symbol via
:class:`~alphaforge.core.symbols.SymbolMapper`. No API keys are required — all
endpoints used here are public.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import ccxt
import pyarrow as pa

from alphaforge.core.errors import ConfigError, SchemaError
from alphaforge.core.instruments import Instrument
from alphaforge.core.symbols import SymbolMapper
from alphaforge.core.time import Ms, Timeframe, now_ms
from alphaforge.core.types import AssetClass, MarketType
from alphaforge.data.ingest.retry import RateBudget, transient_retry
from alphaforge.data.schemas import (
    FUNDING_SCHEMA,
    OHLCV_SCHEMA,
    Dataset,
    validate_table,
)
from alphaforge.data.sources.base import DataSource

__all__ = [
    "FUNDING_PUBLICATION_LAG_MS",
    "CCXTDataSource",
]

FUNDING_PUBLICATION_LAG_MS: Final[int] = 300_000
"""Funding-rate publication lag δ_funding: 5 minutes, in milliseconds.

A funding rate is *determined* at settlement (``ts_funding``) but is only
published/queryable on the REST API within minutes of settlement; 5 minutes is the
conservative point-in-time lag per dataDesign.md §3.2 — it costs nothing at 1h
decision frequency and guarantees a backtest never consumes a rate before live
trading could have. Every funding row's ``available_at`` is
``ts_funding + FUNDING_PUBLICATION_LAG_MS``.
"""

_KLINES_PAGE_LIMIT: Final[int] = 1500
"""Max rows per ``/fapi/v1/klines`` request (Binance hard cap)."""

_KLINES_PAGE_WEIGHT: Final[int] = 10
"""Binance request weight of one klines page at limit 1001-1500."""

_FUNDING_PAGE_LIMIT: Final[int] = 1000
"""Max rows per ``/fapi/v1/fundingRate`` request (Binance hard cap)."""

_FUNDING_PAGE_WEIGHT: Final[int] = 1
"""Budget weight charged per funding-history page (cheap, unweighted endpoint)."""

_MARKETS_WEIGHT: Final[int] = 1
"""Budget weight charged for ``exchangeInfo`` (via ``load_markets``)."""

_FUNDING_INFO_WEIGHT: Final[int] = 1
"""Budget weight charged for the one-shot ``/fapi/v1/fundingInfo`` call."""

_DEFAULT_FUNDING_INTERVAL_HOURS: Final[int] = 8
"""Funding interval for symbols absent from ``fundingInfo`` (Binance default)."""

_DEFAULT_MAKER_FEE_BPS: Final[float] = 2.0
_DEFAULT_TAKER_FEE_BPS: Final[float] = 5.0
"""Binance USDT-M VIP0 fees, basis points of notional, no BNB discount
(conservative; dataDesign.md §2.3 instruments table)."""

_TRANSIENT_EXCS: Final[tuple[type[BaseException], ...]] = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RateLimitExceeded,
    ccxt.RequestTimeout,
)
"""ccxt's transient failure family — retried with backoff. ``BadSymbol``,
``AuthenticationError`` etc. are NOT listed and propagate immediately."""

_LIVE_STATUSES: Final[frozenset[str]] = frozenset({"TRADING", "PENDING_TRADING"})
"""exchangeInfo ``status`` values that mean the contract has not ended life.
Anything else (``SETTLED``, ``CLOSE``, ``DELIVERING``, ...) is treated as delisted."""

_EXCHANGE_LABELS: Final[dict[str, str]] = {
    "binanceusdm": "BINANCE",
    "binance": "BINANCE",
    "binancecoinm": "BINANCE",
}
"""ccxt exchange id → canonical exchange segment of ``instrument_id``. Unlisted ids
fall back to ``exchange_id.upper()``."""


def _opt_float(value: Any) -> float | None:
    """Parse an exchange-payload number (str/int/float) to float; None/'' → None."""
    if value is None or value == "":
        return None
    return float(value)


class CCXTDataSource(DataSource):
    """USDT-margined perpetual market data from Binance futures via ccxt.

    Construction performs no I/O; the ccxt client is built lazily-cheap here and
    every network call happens inside the fetch methods, each metered by the
    optional rate budget and retried on transient errors.

    Args:
        exchange_id: ccxt exchange id (default ``"binanceusdm"`` — Binance USDT-M
            futures). Must name a class on the ``ccxt`` module when ``client`` is
            not injected; otherwise raises
            :class:`~alphaforge.core.errors.ConfigError`.
        rate_budget: Optional client-side request-weight budget; when present,
            ``acquire(weight)`` is called before *every* REST request (klines pages
            cost 10, the rest 1). ``None`` relies on ccxt's own throttler only.
        client: Injectable ccxt-shaped client (tests pass a fake). ``None``
            constructs ``ccxt.<exchange_id>({"enableRateLimit": True})``.
    """

    def __init__(
        self,
        exchange_id: str = "binanceusdm",
        *,
        rate_budget: RateBudget | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            exchange_cls = getattr(ccxt, exchange_id, None)
            if exchange_cls is None:
                raise ConfigError(
                    f"unknown ccxt exchange id {exchange_id!r}: no such class on the "
                    "ccxt module; cannot construct a client"
                )
            client = exchange_cls({"enableRateLimit": True})
        self._client = client
        self._budget = rate_budget
        self._exchange_id = exchange_id
        self._exchange_label = _EXCHANGE_LABELS.get(exchange_id, exchange_id.upper())
        self.name = f"ccxt.{exchange_id}"
        self._retry = transient_retry(_TRANSIENT_EXCS)
        #: Lazy per-instance cache of /fapi/v1/fundingInfo: raw symbol → interval
        #: hours for symbols with NON-default funding; fetched at most once.
        self._funding_intervals: dict[str, int] | None = None

    # ------------------------------------------------------------------ helpers

    def _acquire(self, weight: int) -> None:
        """Charge ``weight`` against the rate budget (no-op when budget is None)."""
        if self._budget is not None:
            self._budget.acquire(weight)

    def _raw_symbol(self, instrument_id: str) -> str:
        """Canonical id → raw exchange symbol (``"BINANCE:PERP:BTCUSDT"`` → ``"BTCUSDT"``).

        Raises:
            SchemaError: malformed ``instrument_id`` (propagated from the mapper).
            ValueError: well-formed id for a market this source does not serve
                (non-PERP, or an exchange other than this client's).
        """
        exchange, market_type, symbol = SymbolMapper.parse_instrument_id(instrument_id)
        if market_type is not MarketType.PERP:
            raise ValueError(
                f"CCXTDataSource({self._exchange_id!r}) serves USDT-margined PERPs "
                f"only; got {instrument_id!r} with market type {market_type.value!r}"
            )
        if exchange != self._exchange_label:
            raise ValueError(
                f"instrument {instrument_id!r} belongs to exchange {exchange!r}, "
                f"but this source is {self._exchange_label!r} ({self._exchange_id!r})"
            )
        return symbol

    def _funding_intervals_map(self) -> dict[str, int]:
        """Fetch (once per instance) the non-default funding intervals overlay.

        ``/fapi/v1/fundingInfo`` lists ONLY symbols whose funding deviates from the
        8h default (4h or 1h, plus adjusted caps); symbols absent from the response
        fund every :data:`_DEFAULT_FUNDING_INTERVAL_HOURS` hours. The result is
        cached on the instance — listing N instruments costs one call, not N.
        """
        if self._funding_intervals is None:
            self._acquire(_FUNDING_INFO_WEIGHT)
            rows: Any = self._retry(self._client.fapiPublicGetFundingInfo)
            intervals: dict[str, int] = {}
            for row in rows:
                symbol = row.get("symbol")
                hours = row.get("fundingIntervalHours")
                if symbol and hours is not None:
                    intervals[str(symbol)] = int(hours)
            self._funding_intervals = intervals
        return self._funding_intervals

    @staticmethod
    def _is_usdt_perp(market: Mapping[str, Any]) -> bool:
        """True iff the ccxt market is a linear (USDT-quoted) perpetual swap.

        Excludes dated futures (``swap`` falsy), inverse/COIN-M contracts
        (``linear`` falsy), and non-USDT quotes (USDC-margined etc.).
        """
        return (
            bool(market.get("swap"))
            and bool(market.get("linear"))
            and market.get("quote") == "USDT"
        )

    def _instrument_from_market(
        self,
        market: Mapping[str, Any],
        funding_intervals: Mapping[str, int],
        as_of: Ms,
    ) -> Instrument:
        """Map one ccxt market dict to a fully-populated :class:`Instrument`.

        Exchange filters are read from the raw Binance ``exchangeInfo`` payload
        (``market["info"]["filters"]``: PRICE_FILTER ``tickSize``, LOT_SIZE
        ``stepSize``/``minQty``, MIN_NOTIONAL ``notional``/``minNotional``),
        falling back to ccxt's unified ``precision``/``limits`` fields (Binance
        runs ccxt's tick-size precision mode, so ``precision.price`` IS the tick).

        Lifecycle: ``listed_ts`` from ``info.onboardDate`` (epoch ms; 0 when the
        venue omits it). ``delisted_ts`` is ``None`` while ``status`` is live
        (TRADING / PENDING_TRADING); otherwise best-effort — the venue's
        ``deliveryDate`` when it is a plausible past instant, else the observation
        time ``as_of`` (we can only know a delisting when we observe it).

        Raises:
            SchemaError: no usable tick/lot/min-qty filter for the market.
        """
        info: Mapping[str, Any] = market.get("info") or {}
        ccxt_symbol = str(market["symbol"])
        instrument_id = SymbolMapper.from_ccxt(self._exchange_label, ccxt_symbol)
        _, _, exchange_symbol = SymbolMapper.parse_instrument_id(instrument_id)
        base, quote = SymbolMapper.split_exchange_symbol(exchange_symbol)

        filters: dict[str, Mapping[str, Any]] = {}
        for flt in info.get("filters") or []:
            filter_type = flt.get("filterType")
            if isinstance(filter_type, str):
                filters[filter_type] = flt

        precision: Mapping[str, Any] = market.get("precision") or {}
        limits: Mapping[str, Any] = market.get("limits") or {}

        tick_size = _opt_float(filters.get("PRICE_FILTER", {}).get("tickSize"))
        if tick_size is None:
            tick_size = _opt_float(precision.get("price"))

        lot_filter = filters.get("LOT_SIZE", {})
        lot_size = _opt_float(lot_filter.get("stepSize"))
        if lot_size is None:
            lot_size = _opt_float(precision.get("amount"))

        min_qty = _opt_float(lot_filter.get("minQty"))
        if min_qty is None:
            min_qty = _opt_float((limits.get("amount") or {}).get("min"))
        if min_qty is None:
            min_qty = lot_size

        if tick_size is None or lot_size is None or min_qty is None:
            raise SchemaError(
                f"market {ccxt_symbol!r}: no usable PRICE_FILTER/LOT_SIZE filter and "
                "no unified precision/limits fallback — cannot build Instrument"
            )

        notional_filter = filters.get("MIN_NOTIONAL", {})
        min_notional = _opt_float(notional_filter.get("notional"))
        if min_notional is None:
            min_notional = _opt_float(notional_filter.get("minNotional"))
        if min_notional is None:
            min_notional = _opt_float((limits.get("cost") or {}).get("min"))
        if min_notional is None:
            min_notional = 0.0

        onboard = info.get("onboardDate")
        listed_ts: Ms = int(onboard) if onboard is not None else 0

        status = str(info.get("status") or ("TRADING" if market.get("active", True) else ""))
        if status in _LIVE_STATUSES:
            delisted_ts: Ms | None = None
        else:
            delivery = _opt_float(info.get("deliveryDate"))
            delivery_ms = None if delivery is None else int(delivery)
            if delivery_ms is not None and listed_ts < delivery_ms <= as_of:
                delisted_ts = delivery_ms
            else:
                delisted_ts = max(as_of, listed_ts + 1)

        return Instrument(
            instrument_id=instrument_id,
            asset_class=AssetClass.CRYPTO_PERP,
            market_type=MarketType.PERP,
            base=base,
            quote=quote,
            tick_size=tick_size,
            lot_size=lot_size,
            min_qty=min_qty,
            min_notional=min_notional,
            contract_multiplier=1.0,
            can_short=True,
            maker_fee_bps=_DEFAULT_MAKER_FEE_BPS,
            taker_fee_bps=_DEFAULT_TAKER_FEE_BPS,
            funding_interval_hours=funding_intervals.get(
                exchange_symbol, _DEFAULT_FUNDING_INTERVAL_HOURS
            ),
            listed_ts=listed_ts,
            delisted_ts=delisted_ts,
        )

    # ----------------------------------------------------------- DataSource API

    def list_instruments(self, *, as_of: Ms) -> list[Instrument]:
        """Return all USDT-margined perpetuals from the venue's CURRENT listing.

        .. warning:: **Survivorship-biased on its own** (leakageCritique.md finding
           1): this is the live ``exchangeInfo`` endpoint — markets Binance fully
           purged before this system first observed them never appear here, so a
           historical candidate universe seeded only from this method has already
           "survived" to today. The historical listing catalog (Binance Vision
           dumps / archived ``exchangeInfo``) complements this method; until it
           exists, backtest universes predating go-live are biased by construction.
           Markets the venue still reports in a dead state (e.g. ``SETTLED``) ARE
           returned, with ``delisted_ts`` set best-effort.

        Filtering: only ``swap + linear + USDT-quoted`` ccxt markets survive —
        dated futures, USDC-margined and inverse contracts are skipped. Funding
        intervals come from one cached ``/fapi/v1/fundingInfo`` call (4h/1h
        overlays; absent symbols default to 8h — leakage finding 6).

        Args:
            as_of: Observation timestamp (epoch ms UTC); stamps SCD2 versioning
                upstream and is the fallback ``delisted_ts`` for dead markets
                without a usable venue timestamp.

        Returns:
            Instruments sorted by ``instrument_id``.
        """
        self._acquire(_MARKETS_WEIGHT)
        markets: Mapping[str, Any] = self._retry(self._client.load_markets)
        funding_intervals = self._funding_intervals_map()
        out = [
            self._instrument_from_market(market, funding_intervals, as_of)
            for market in markets.values()
            if self._is_usdt_perp(market)
        ]
        out.sort(key=lambda inst: inst.instrument_id)
        return out

    def fetch_ohlcv(
        self,
        instrument_id: str,
        *,
        tf: Timeframe,
        since: Ms,
        until: Ms,
        now: Ms | None = None,
    ) -> pa.Table:
        """Fetch closed bars with ``ts_open`` in ``[since, until)`` from raw klines.

        Uses ``/fapi/v1/klines`` (NOT unified ``fetch_ohlcv``, which drops quote
        volume and trade count). Kline array mapping: index 0 ``open_time`` →
        ``ts_open``; 1-4 OHLC; 5 base ``volume``; 7 ``quote_volume``; 8
        ``n_trades`` (floats parsed from the venue's decimal strings).
        ``quality_flags`` is 0 (clean — flagging is the quality layer's job) and
        ``ingested_at`` is the wall clock at fetch time (audit only).

        Pagination walks ``startTime`` in pages of 1500 (weight 10 each against the
        rate budget) until the page is empty/short or ``until`` is reached. Page
        timestamps must be strictly increasing — a non-monotonic payload raises
        :class:`SchemaError` rather than being silently reordered.

        **No partial bars, ever:** any row with ``ts_open + tf.ms > (now or wall
        clock)`` is dropped here — an in-progress bar can never leave this method,
        independent of the writer's own guard (dataDesign.md §3.3).

        Args:
            instrument_id: Canonical perp id, e.g. ``"BINANCE:PERP:BTCUSDT"``.
            tf: Bar timeframe (Binance interval = ``tf.value``).
            since: Inclusive lower bound on ``ts_open`` (epoch ms UTC).
            until: Exclusive upper bound on ``ts_open`` (epoch ms UTC).
            now: Injectable closed-bar cutoff (epoch ms UTC); ``None`` reads the
                wall clock once via :func:`now_ms`.

        Returns:
            Table conforming to :data:`OHLCV_SCHEMA`, sorted and unique on
            ``ts_open`` (possibly empty; ``until == since`` is an empty no-op).

        Raises:
            ValueError: ``until < since``, or an id this source does not serve.
            SchemaError: malformed id or non-monotonic exchange payload.
        """
        symbol = self._raw_symbol(instrument_id)
        if until < since:
            raise ValueError(f"until ({until}) must be >= since ({since})")
        now_eff: Ms = now_ms() if now is None else now
        ingested_at: Ms = now_ms()

        klines: list[list[Any]] = []
        prev_ts: int | None = None
        cursor = since
        while cursor < until:
            self._acquire(_KLINES_PAGE_WEIGHT)
            params = {
                "symbol": symbol,
                "interval": tf.value,
                "startTime": cursor,
                "limit": _KLINES_PAGE_LIMIT,
            }
            page: Any = self._retry(self._client.fapiPublicGetKlines, params)
            if not page:
                break
            for row in page:
                ts = int(row[0])
                if prev_ts is not None and ts <= prev_ts:
                    raise SchemaError(
                        f"klines for {symbol!r} not strictly increasing: "
                        f"open_time {ts} after {prev_ts}"
                    )
                prev_ts = ts
                if ts < since or ts >= until:
                    continue
                if ts + tf.ms > now_eff:
                    continue  # in-progress bar: close is in the future — never serve it
                klines.append(list(row))
            cursor = int(page[-1][0]) + tf.ms
            if len(page) < _KLINES_PAGE_LIMIT:
                break

        table = pa.Table.from_pydict(
            {
                "instrument_id": [instrument_id] * len(klines),
                "ts_open": [int(k[0]) for k in klines],
                "open": [float(k[1]) for k in klines],
                "high": [float(k[2]) for k in klines],
                "low": [float(k[3]) for k in klines],
                "close": [float(k[4]) for k in klines],
                "volume": [float(k[5]) for k in klines],
                "quote_volume": [_opt_float(k[7]) for k in klines],
                "n_trades": [None if k[8] is None else int(k[8]) for k in klines],
                "quality_flags": [0] * len(klines),
                "ingested_at": [ingested_at] * len(klines),
            },
            schema=OHLCV_SCHEMA,
        )
        validate_table(table, Dataset.OHLCV)
        return table

    def fetch_funding(self, instrument_id: str, *, since: Ms, until: Ms) -> pa.Table:
        """Fetch settled funding with ``ts_funding`` in ``[since, until)``.

        Uses ``/fapi/v1/fundingRate`` paginated by ``startTime`` in pages of 1000
        (weight 1 each against the rate budget) until the page is empty/short or
        ``until`` is reached. Settlement timestamps must be strictly increasing —
        a non-monotonic payload raises :class:`SchemaError`.

        Point-in-time: each row's ``available_at`` is ``ts_funding +``
        :data:`FUNDING_PUBLICATION_LAG_MS` (5 min — rates are published/queryable
        within minutes of settlement; dataDesign.md §3.2). Downstream consumers
        join on ``available_at`` only (leakage finding 18). Rates are stored
        exactly as returned (per-interval decimals) — never rescaled.

        Args:
            instrument_id: Canonical perp id, e.g. ``"BINANCE:PERP:BTCUSDT"``.
            since: Inclusive lower bound on ``ts_funding`` (epoch ms UTC).
            until: Exclusive upper bound on ``ts_funding`` (epoch ms UTC).

        Returns:
            Table conforming to :data:`FUNDING_SCHEMA`, sorted and unique on
            ``ts_funding`` (possibly empty; ``until == since`` is an empty no-op).

        Raises:
            ValueError: ``until < since``, or an id this source does not serve.
            SchemaError: malformed id or non-monotonic exchange payload.
        """
        symbol = self._raw_symbol(instrument_id)
        if until < since:
            raise ValueError(f"until ({until}) must be >= since ({since})")
        ingested_at: Ms = now_ms()

        events: list[tuple[int, float]] = []
        prev_ts: int | None = None
        cursor = since
        while cursor < until:
            self._acquire(_FUNDING_PAGE_WEIGHT)
            params = {"symbol": symbol, "startTime": cursor, "limit": _FUNDING_PAGE_LIMIT}
            page: Any = self._retry(self._client.fapiPublicGetFundingRate, params)
            if not page:
                break
            for row in page:
                ts = int(row["fundingTime"])
                if prev_ts is not None and ts <= prev_ts:
                    raise SchemaError(
                        f"funding history for {symbol!r} not strictly increasing: "
                        f"fundingTime {ts} after {prev_ts}"
                    )
                prev_ts = ts
                if since <= ts < until:
                    events.append((ts, float(row["fundingRate"])))
            cursor = int(page[-1]["fundingTime"]) + 1
            if len(page) < _FUNDING_PAGE_LIMIT:
                break

        table = pa.Table.from_pydict(
            {
                "instrument_id": [instrument_id] * len(events),
                "ts_funding": [ts for ts, _ in events],
                "rate": [rate for _, rate in events],
                "available_at": [ts + FUNDING_PUBLICATION_LAG_MS for ts, _ in events],
                "ingested_at": [ingested_at] * len(events),
            },
            schema=FUNDING_SCHEMA,
        )
        validate_table(table, Dataset.FUNDING)
        return table
