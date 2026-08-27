"""Symbol mapping: canonical instrument_id ↔ exchange symbol ↔ ccxt unified symbol.

Canonical instrument identity (dataDesign.md §2.3, buildabilityCritique.md ruling 3.2):

    "<EXCHANGE>:<MARKET>:<EXCHANGE_SYMBOL>"   e.g. "BINANCE:PERP:BTCUSDT"

- ``EXCHANGE`` and ``EXCHANGE_SYMBOL`` are uppercase; ``MARKET`` is a
  :class:`~alphaforge.core.types.MarketType` member name.
  (the :class:`~alphaforge.core.types.MarketType` member name).
- ccxt unified symbols: spot ``"BTC/USDT"``; linear perp ``"BTC/USDT:USDT"`` (settle
  currency equals the quote — inverse contracts are out of scope and rejected).
- Exchange symbols concatenate base+quote with no separator; the quote is recovered by
  longest-suffix match against :data:`KNOWN_QUOTES`, which handles multi-character
  quotes (``FDUSD`` vs ``USD``) and oddball bases (``1000SHIB``, ``BTCDOM``).

All functions are pure; malformed input raises :class:`~alphaforge.core.errors.SchemaError`.
"""

from __future__ import annotations

from typing import Final

from alphaforge.core.errors import SchemaError
from alphaforge.core.types import MarketType

__all__ = [
    "EQUITY_MIC",
    "KNOWN_QUOTES",
    "SymbolMapper",
]

#: Quote currencies recognized when splitting concatenated exchange symbols, ordered
#: longest-first so that e.g. ``BTCFDUSD`` splits as ``BTC``/``FDUSD``, never ``BTCFD``/``USD``.
KNOWN_QUOTES: Final[tuple[str, ...]] = (
    "FDUSD",
    "BUSD",
    "TUSD",
    "USDC",
    "USDT",
    "DAI",
    "BNB",
    "BTC",
    "ETH",
    "EUR",
    "TRY",
    "USD",
)

#: Synthetic quote currency pinned to every CASH-equity exchange symbol so the canonical
#: id grammar and the Instrument ``base+quote`` invariant hold without a real quote
#: currency. Stripped by fixed length in :meth:`SymbolMapper.parse_equity_id` (never by
#: KNOWN_QUOTES suffix matching), so quote-colliding tickers round-trip (critique B4).
_EQUITY_QUOTE: Final[str] = "USD"

#: Canonical synthetic venue segment for EVERY US cash-equity instrument_id ("XUSE" =
#: "US equities, exchange-unspecified"). It is THE single source of truth for the equity
#: id's MIC segment: the Polygon flat-files day-aggs carry no per-row listing exchange, so
#: bars must use a synthetic venue — and the reference seeder MUST use the SAME one or the
#: bars and the SCD2 instrument record key under different ids and never join. The real
#: listing venue is intentionally NOT in the id (and is not needed: ``calendar_for`` is
#: keyed on AssetClass, not MIC). Both ``polygon_flatfiles`` and ``polygon_source`` build
#: ids with this constant.
EQUITY_MIC: Final[str] = "XUSE"

_MARKET_LABELS: Final[dict[str, MarketType]] = {m.name: m for m in MarketType}


class SymbolMapper:
    """Pure, stateless conversions between the three symbol vocabularies.

    All methods are static; the class is a namespace, never instantiated with state.
    """

    @staticmethod
    def to_instrument_id(exchange: str, market_type: MarketType, exchange_symbol: str) -> str:
        """Build a canonical instrument_id; normalizes ``exchange``/``exchange_symbol`` to upper.

        Raises :class:`SchemaError` if either string is empty or contains ``":"``.
        """
        for name, value in (("exchange", exchange), ("exchange_symbol", exchange_symbol)):
            if not value or ":" in value:
                raise SchemaError(f"{name} must be a non-empty string without ':', got {value!r}")
        return f"{exchange.upper()}:{market_type.name}:{exchange_symbol.upper()}"

    @staticmethod
    def parse_instrument_id(instrument_id: str) -> tuple[str, MarketType, str]:
        """Split a canonical instrument_id into ``(exchange, market_type, exchange_symbol)``.

        Strict: exactly three non-empty colon-separated segments, exchange and symbol
        uppercase, market segment a :class:`MarketType` member name. Raises
        :class:`SchemaError` otherwise (never silently normalizes).
        """
        parts = instrument_id.split(":")
        if len(parts) != 3 or not all(parts):
            raise SchemaError(
                f"instrument_id must be 'EXCHANGE:MARKET:SYMBOL', got {instrument_id!r}"
            )
        exchange, market_label, symbol = parts
        if exchange != exchange.upper() or symbol != symbol.upper():
            raise SchemaError(f"instrument_id segments must be uppercase, got {instrument_id!r}")
        market_type = _MARKET_LABELS.get(market_label)
        if market_type is None:
            raise SchemaError(
                f"unknown market type {market_label!r} in {instrument_id!r}; "
                f"expected one of {sorted(_MARKET_LABELS)}"
            )
        return exchange, market_type, symbol

    @staticmethod
    def split_exchange_symbol(exchange_symbol: str) -> tuple[str, str]:
        """Split a concatenated exchange symbol into ``(base, quote)``.

        Longest-suffix match against :data:`KNOWN_QUOTES`; the base must be non-empty.
        Raises :class:`SchemaError` if no known quote matches or the base is empty.
        """
        for quote in KNOWN_QUOTES:
            if exchange_symbol.endswith(quote):
                base = exchange_symbol[: -len(quote)]
                if not base:
                    raise SchemaError(
                        f"exchange symbol {exchange_symbol!r} has empty base after "
                        f"stripping quote {quote!r}"
                    )
                return base, quote
        raise SchemaError(
            f"exchange symbol {exchange_symbol!r} ends with no known quote (known: {KNOWN_QUOTES})"
        )

    @staticmethod
    def equity_instrument_id(mic: str, ticker: str) -> str:
        """Build a canonical CASH-equity instrument_id from a listing MIC and ticker.

        Equities have no quote currency, but the canonical id grammar
        ``"<EXCHANGE>:<MARKET>:<EXCHANGE_SYMBOL>"`` and the
        :class:`~alphaforge.core.instruments.Instrument` ``base+quote`` invariant both
        want a non-empty quote. We pin a synthetic quote ``"USD"`` and store the
        exchange symbol as ``"<TICKER>USD"``, so ``"XNAS:CASH:AAPLUSD"`` reconstructs
        from ``base="AAPL"`` + ``quote="USD"``. **Crucially this is NOT routed through
        :meth:`split_exchange_symbol`** — equity tickers collide with the crypto quote
        vocabulary (``BTC``, ``ETH``, ``EUR`` are real tickers), and longest-suffix
        matching would mis-split them (EQUITIES_CRITIQUE.md B4). Recover the ticker only
        via :meth:`parse_equity_id`, never via ``split_exchange_symbol``.

        Dotted/class tickers (``BRK.B``) must be canonicalized to a separator-free form
        (``BRKB``) by the caller before this call; the raw vendor ticker is carried
        separately by the source for the fetch round-trip. ``mic`` is the listing MIC
        (``XNAS``, ``XNYS``, ...). Raises :class:`SchemaError` if either string is
        empty, contains ``":"``, or — for ``ticker`` — contains a ``"."`` (un-normalized
        dotted ticker) or already ends in the synthetic ``"USD"`` quote (which would make
        the id non-round-trippable).
        """
        for name, value in (("mic", mic), ("ticker", ticker)):
            if not value or ":" in value:
                raise SchemaError(f"{name} must be a non-empty string without ':', got {value!r}")
        if "." in ticker:
            raise SchemaError(
                f"ticker {ticker!r} contains '.'; canonicalize dotted/class tickers "
                "(e.g. 'BRK.B' -> 'BRKB') before building an equity id"
            )
        if ticker.upper().endswith(_EQUITY_QUOTE):
            raise SchemaError(
                f"ticker {ticker!r} ends with the synthetic equity quote {_EQUITY_QUOTE!r}; "
                "such tickers cannot round-trip through parse_equity_id"
            )
        return f"{mic.upper()}:{MarketType.CASH.name}:{ticker.upper()}{_EQUITY_QUOTE}"

    @staticmethod
    def parse_equity_id(instrument_id: str) -> tuple[str, str]:
        """Split a CASH-equity instrument_id into ``(mic, ticker)`` — collision-free.

        Inverse of :meth:`equity_instrument_id`: validates the market segment is
        ``CASH`` and strips exactly the synthetic ``"USD"`` quote suffix (a fixed-length
        strip, NOT a :data:`KNOWN_QUOTES` longest-suffix match), so a ticker that itself
        ends in a crypto quote token is recovered intact. Raises :class:`SchemaError` on
        a malformed id, a non-``CASH`` market segment, or a symbol not ending in the
        synthetic quote.
        """
        exchange, market_type, symbol = SymbolMapper.parse_instrument_id(instrument_id)
        if market_type is not MarketType.CASH:
            raise SchemaError(
                f"parse_equity_id expects a CASH-equity id, got market type "
                f"{market_type.value!r} in {instrument_id!r}"
            )
        if not symbol.endswith(_EQUITY_QUOTE) or symbol == _EQUITY_QUOTE:
            raise SchemaError(
                f"equity exchange symbol {symbol!r} of {instrument_id!r} does not end in "
                f"the synthetic quote {_EQUITY_QUOTE!r}"
            )
        return exchange, symbol[: -len(_EQUITY_QUOTE)]

    @staticmethod
    def to_ccxt(instrument_id: str) -> str:
        """Canonical instrument_id → ccxt unified symbol.

        ``"BINANCE:PERP:BTCUSDT"`` → ``"BTC/USDT:USDT"`` (linear perp, settle = quote);
        ``"BINANCE:SPOT:BTCUSDT"`` → ``"BTC/USDT"``. Raises :class:`SchemaError` on a
        malformed id, an unrecognized quote suffix, or any non-crypto market id (ccxt is a
        crypto-venue vocabulary; routing cash equities, dated futures, or options here would
        emit a spurious crypto market symbol, so they are rejected loudly, not mis-translated).
        """
        _, market_type, symbol = SymbolMapper.parse_instrument_id(instrument_id)
        if market_type not in (MarketType.PERP, MarketType.SPOT):
            raise SchemaError(
                f"cannot map {market_type.name} id {instrument_id!r} to a ccxt symbol; "
                "only crypto PERP/SPOT markets use ccxt routing"
            )
        base, quote = SymbolMapper.split_exchange_symbol(symbol)
        if market_type is MarketType.PERP:
            return f"{base}/{quote}:{quote}"
        return f"{base}/{quote}"

    @staticmethod
    def from_ccxt(exchange: str, ccxt_symbol: str) -> str:
        """ccxt unified symbol → canonical instrument_id.

        ``"BTC/USDT:USDT"`` → perp, ``"BTC/USDT"`` → spot. The settle suffix (if any)
        must equal the quote (linear contracts only) and the quote must be in
        :data:`KNOWN_QUOTES` so the resulting id round-trips through :meth:`to_ccxt`.
        Raises :class:`SchemaError` on malformed/inverse/unknown-quote symbols.
        """
        head, sep, settle = ccxt_symbol.partition(":")
        base, slash, quote = head.partition("/")
        if not slash or not base or not quote or "/" in quote:
            raise SchemaError(f"ccxt symbol must be 'BASE/QUOTE[:SETTLE]', got {ccxt_symbol!r}")
        if sep:
            if ":" in settle or not settle:
                raise SchemaError(f"malformed settle suffix in ccxt symbol {ccxt_symbol!r}")
            if settle != quote:
                raise SchemaError(
                    f"inverse/non-linear contract {ccxt_symbol!r} unsupported: "
                    f"settle {settle!r} != quote {quote!r}"
                )
            market_type = MarketType.PERP
        else:
            market_type = MarketType.SPOT
        if quote.upper() not in KNOWN_QUOTES:
            raise SchemaError(
                f"quote {quote!r} of {ccxt_symbol!r} not in KNOWN_QUOTES; "
                "the id would not round-trip through to_ccxt"
            )
        return SymbolMapper.to_instrument_id(exchange, market_type, f"{base}{quote}")
