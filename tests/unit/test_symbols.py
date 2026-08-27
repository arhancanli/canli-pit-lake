"""Unit tests for alphaforge.core.symbols — canonical id ↔ exchange ↔ ccxt mapping."""

from __future__ import annotations

import pytest

from alphaforge.core.errors import SchemaError
from alphaforge.core.symbols import KNOWN_QUOTES, SymbolMapper
from alphaforge.core.types import MarketType

# (market_type, exchange_symbol, instrument_id, ccxt_symbol) — incl. multi-char quote
# and oddball-base cases (1000SHIBUSDT, BTCDOMUSDT) that break naive 3/4-char splits.
CORPUS: list[tuple[MarketType, str, str, str]] = [
    (MarketType.PERP, "BTCUSDT", "BINANCE:PERP:BTCUSDT", "BTC/USDT:USDT"),
    (MarketType.SPOT, "BTCUSDT", "BINANCE:SPOT:BTCUSDT", "BTC/USDT"),
    (MarketType.PERP, "ETHUSDT", "BINANCE:PERP:ETHUSDT", "ETH/USDT:USDT"),
    (MarketType.PERP, "1000SHIBUSDT", "BINANCE:PERP:1000SHIBUSDT", "1000SHIB/USDT:USDT"),
    (MarketType.PERP, "1000PEPEUSDT", "BINANCE:PERP:1000PEPEUSDT", "1000PEPE/USDT:USDT"),
    (MarketType.PERP, "BTCDOMUSDT", "BINANCE:PERP:BTCDOMUSDT", "BTCDOM/USDT:USDT"),
    (MarketType.PERP, "ETHUSDC", "BINANCE:PERP:ETHUSDC", "ETH/USDC:USDC"),
    (MarketType.SPOT, "BTCFDUSD", "BINANCE:SPOT:BTCFDUSD", "BTC/FDUSD"),
    (MarketType.SPOT, "BNBBUSD", "BINANCE:SPOT:BNBBUSD", "BNB/BUSD"),
    (MarketType.SPOT, "ETHBTC", "BINANCE:SPOT:ETHBTC", "ETH/BTC"),
    (MarketType.SPOT, "USDCUSDT", "BINANCE:SPOT:USDCUSDT", "USDC/USDT"),
    (MarketType.PERP, "DOGEUSDT", "BINANCE:PERP:DOGEUSDT", "DOGE/USDT:USDT"),
]


class TestToInstrumentId:
    def test_basic_perp(self) -> None:
        iid = SymbolMapper.to_instrument_id("BINANCE", MarketType.PERP, "BTCUSDT")
        assert iid == "BINANCE:PERP:BTCUSDT"

    def test_normalizes_case(self) -> None:
        iid = SymbolMapper.to_instrument_id("binance", MarketType.SPOT, "btcusdt")
        assert iid == "BINANCE:SPOT:BTCUSDT"

    @pytest.mark.parametrize(
        ("exchange", "symbol"),
        [("", "BTCUSDT"), ("BINANCE", ""), ("BIN:ANCE", "BTCUSDT"), ("BINANCE", "BTC:USDT")],
    )
    def test_rejects_empty_or_colon(self, exchange: str, symbol: str) -> None:
        with pytest.raises(SchemaError):
            SymbolMapper.to_instrument_id(exchange, MarketType.PERP, symbol)


class TestParseInstrumentId:
    def test_round_trips_to_instrument_id(self) -> None:
        for market_type, symbol, iid, _ in CORPUS:
            assert SymbolMapper.parse_instrument_id(iid) == ("BINANCE", market_type, symbol)
            assert SymbolMapper.to_instrument_id("BINANCE", market_type, symbol) == iid

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "BTCUSDT",
            "BINANCE:BTCUSDT",
            "BINANCE:PERP:BTC:USDT",
            "BINANCE:FUTURES:BTCUSDT",
            "BINANCE:perp:BTCUSDT",
            "binance:PERP:BTCUSDT",
            "BINANCE:PERP:btcusdt",
            "BINANCE:PERP:",
            ":PERP:BTCUSDT",
            "BINANCE::BTCUSDT",
        ],
    )
    def test_malformed_raises_schema_error(self, bad: str) -> None:
        with pytest.raises(SchemaError):
            SymbolMapper.parse_instrument_id(bad)


class TestSplitExchangeSymbol:
    def test_longest_quote_wins(self) -> None:
        # FDUSD must beat the shorter USD suffix.
        assert SymbolMapper.split_exchange_symbol("BTCFDUSD") == ("BTC", "FDUSD")
        assert SymbolMapper.split_exchange_symbol("BNBBUSD") == ("BNB", "BUSD")

    def test_oddball_bases(self) -> None:
        assert SymbolMapper.split_exchange_symbol("1000SHIBUSDT") == ("1000SHIB", "USDT")
        assert SymbolMapper.split_exchange_symbol("BTCDOMUSDT") == ("BTCDOM", "USDT")

    def test_unknown_quote_raises(self) -> None:
        with pytest.raises(SchemaError):
            SymbolMapper.split_exchange_symbol("BTCXYZ")

    def test_empty_base_raises(self) -> None:
        with pytest.raises(SchemaError):
            SymbolMapper.split_exchange_symbol("USDT")

    def test_known_quotes_ordered_longest_first(self) -> None:
        lengths = [len(q) for q in KNOWN_QUOTES]
        assert lengths == sorted(lengths, reverse=True)


class TestToCcxt:
    @pytest.mark.parametrize(("_mt", "_sym", "iid", "ccxt"), CORPUS)
    def test_corpus(self, _mt: MarketType, _sym: str, iid: str, ccxt: str) -> None:
        assert SymbolMapper.to_ccxt(iid) == ccxt

    def test_unknown_quote_raises(self) -> None:
        with pytest.raises(SchemaError):
            SymbolMapper.to_ccxt("BINANCE:PERP:BTCXYZ")

    def test_malformed_id_raises(self) -> None:
        with pytest.raises(SchemaError):
            SymbolMapper.to_ccxt("BTC/USDT:USDT")


class TestFromCcxt:
    @pytest.mark.parametrize(("_mt", "_sym", "iid", "ccxt"), CORPUS)
    def test_corpus(self, _mt: MarketType, _sym: str, iid: str, ccxt: str) -> None:
        assert SymbolMapper.from_ccxt("BINANCE", ccxt) == iid

    def test_lowercase_exchange_normalized(self) -> None:
        assert SymbolMapper.from_ccxt("binance", "BTC/USDT:USDT") == "BINANCE:PERP:BTCUSDT"

    def test_inverse_contract_rejected(self) -> None:
        with pytest.raises(SchemaError):
            SymbolMapper.from_ccxt("BINANCE", "BTC/USD:BTC")

    @pytest.mark.parametrize(
        "bad",
        ["BTCUSDT", "BTC/", "/USDT", "BTC/USDT:", "BTC/USDT:USDT:USDT", "BTC//USDT", ""],
    )
    def test_malformed_raises(self, bad: str) -> None:
        with pytest.raises(SchemaError):
            SymbolMapper.from_ccxt("BINANCE", bad)

    def test_unknown_quote_rejected_early(self) -> None:
        # Would not round-trip through to_ccxt, so it must fail at ingestion.
        with pytest.raises(SchemaError):
            SymbolMapper.from_ccxt("BINANCE", "BTC/XYZ")


class TestRoundTrip:
    """to_ccxt ∘ from_ccxt = id and from_ccxt ∘ to_ccxt = id over the corpus."""

    @pytest.mark.parametrize(("_mt", "_sym", "iid", "ccxt"), CORPUS)
    def test_ccxt_fixed_point(self, _mt: MarketType, _sym: str, iid: str, ccxt: str) -> None:
        assert SymbolMapper.to_ccxt(SymbolMapper.from_ccxt("BINANCE", ccxt)) == ccxt

    @pytest.mark.parametrize(("_mt", "_sym", "iid", "ccxt"), CORPUS)
    def test_instrument_id_fixed_point(
        self, _mt: MarketType, _sym: str, iid: str, ccxt: str
    ) -> None:
        assert SymbolMapper.from_ccxt("BINANCE", SymbolMapper.to_ccxt(iid)) == iid


class TestEquityInstrumentId:
    """The collision-free CASH-equity id helpers (EQUITIES_CRITIQUE.md B4).

    The equity id deliberately does NOT route through ``split_exchange_symbol`` — the
    crypto longest-suffix matcher would mis-split equity tickers that collide with the
    crypto quote vocabulary. ``equity_instrument_id`` / ``parse_equity_id`` strip exactly
    the synthetic ``USD`` quote by fixed length, so a quote-colliding ticker round-trips
    intact.
    """

    def test_build_basic(self) -> None:
        assert SymbolMapper.equity_instrument_id("XNAS", "AAPL") == "XNAS:CASH:AAPLUSD"

    def test_build_normalizes_case(self) -> None:
        assert SymbolMapper.equity_instrument_id("xnas", "aapl") == "XNAS:CASH:AAPLUSD"

    def test_round_trip_common_ticker(self) -> None:
        iid = SymbolMapper.equity_instrument_id("XNYS", "BRKB")
        assert SymbolMapper.parse_equity_id(iid) == ("XNYS", "BRKB")

    # THE B4 regression: tickers that ARE crypto quote tokens (BTC/ETH/EUR are real
    # tickers — Grayscale trusts, ADRs). The crypto split_exchange_symbol would mis-split
    # these; the equity helpers must round-trip them byte-for-byte.
    @pytest.mark.parametrize("ticker", ["BTC", "ETH", "EUR", "TRY", "BNB", "DAI"])
    def test_quote_colliding_ticker_round_trips(self, ticker: str) -> None:
        iid = SymbolMapper.equity_instrument_id("XNAS", ticker)
        assert iid == f"XNAS:CASH:{ticker}USD"
        # Recovered intact — NOT mis-split into ("", ticker) or similar by KNOWN_QUOTES.
        assert SymbolMapper.parse_equity_id(iid) == ("XNAS", ticker)

    def test_quote_colliding_ticker_not_routed_through_split_exchange_symbol(self) -> None:
        # Proof of the hazard the equity path avoids: the SHARED crypto helper, fed the
        # equity exchange symbol "ETHUSD", would strip a crypto quote and mis-split it.
        # parse_equity_id must NOT agree with that wrong answer.
        mis_split_base, _ = SymbolMapper.split_exchange_symbol("ETHUSD")
        assert mis_split_base == "ETH"  # crypto matcher peels USD, base "ETH" — fine HERE,
        # but the point is parse_equity_id is independent and recovers the FULL ticker:
        _, ticker = SymbolMapper.parse_equity_id("XNAS:CASH:ETHUSD")
        assert ticker == "ETH"

    @pytest.mark.parametrize(
        ("mic", "ticker"),
        [("", "AAPL"), ("XNAS", ""), ("X:NAS", "AAPL"), ("XNAS", "AA:PL")],
    )
    def test_build_rejects_empty_or_colon(self, mic: str, ticker: str) -> None:
        with pytest.raises(SchemaError):
            SymbolMapper.equity_instrument_id(mic, ticker)

    def test_build_rejects_dotted_ticker(self) -> None:
        # Dotted/class tickers must be canonicalized (BRK.B -> BRKB) before id-building.
        with pytest.raises(SchemaError, match="canonicalize"):
            SymbolMapper.equity_instrument_id("XNYS", "BRK.B")

    def test_build_rejects_ticker_ending_in_synthetic_quote(self) -> None:
        # A ticker literally ending in "USD" cannot round-trip (the strip is ambiguous).
        with pytest.raises(SchemaError, match="synthetic equity quote"):
            SymbolMapper.equity_instrument_id("XNAS", "FUSD")

    def test_parse_rejects_non_cash_market(self) -> None:
        with pytest.raises(SchemaError, match="CASH"):
            SymbolMapper.parse_equity_id("BINANCE:PERP:BTCUSDT")

    def test_parse_rejects_symbol_without_synthetic_quote(self) -> None:
        with pytest.raises(SchemaError, match="synthetic quote"):
            SymbolMapper.parse_equity_id("XNAS:CASH:AAPLEUR")

    def test_to_ccxt_rejects_equity_id(self) -> None:
        # An equity id must NEVER be mis-translated into a (crypto) ccxt symbol (B4).
        with pytest.raises(SchemaError, match="ccxt"):
            SymbolMapper.to_ccxt("XNAS:CASH:AAPLUSD")
