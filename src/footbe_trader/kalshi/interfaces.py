"""Kalshi API interfaces and data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EventData:
    """Event data from Kalshi."""

    event_ticker: str
    title: str
    subtitle: str = ""
    category: str = ""
    sub_category: str = ""
    series_ticker: str = ""
    mutually_exclusive: bool = True
    strike_date: datetime | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketData:
    """Market data from Kalshi."""

    ticker: str
    event_ticker: str
    title: str
    subtitle: str = ""
    status: str = "open"
    open_time: datetime | None = None
    close_time: datetime | None = None
    expiration_time: datetime | None = None
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    no_bid: float = 0.0
    no_ask: float = 0.0
    last_price: float = 0.0
    previous_yes_bid: float = 0.0
    previous_yes_ask: float = 0.0
    volume: int = 0
    volume_24h: int = 0
    open_interest: int = 0
    result: str = ""  # "yes", "no", or "" if not settled
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderbookLevel:
    """Single level in an orderbook."""

    price: float
    quantity: int


@dataclass
class OrderbookData:
    """Orderbook data from Kalshi."""

    ticker: str
    yes_bids: list[OrderbookLevel] = field(default_factory=list)
    yes_asks: list[OrderbookLevel] = field(default_factory=list)
    no_bids: list[OrderbookLevel] = field(default_factory=list)
    no_asks: list[OrderbookLevel] = field(default_factory=list)
    timestamp: datetime | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)

    @property
    def best_yes_bid(self) -> float:
        """Best yes bid price (0 if no bids)."""
        return self.yes_bids[0].price if self.yes_bids else 0.0

    @property
    def best_yes_ask(self) -> float:
        """Best yes ask price (1 if no asks)."""
        return self.yes_asks[0].price if self.yes_asks else 1.0

    @property
    def mid_price(self) -> float:
        """Mid price between best bid and ask."""
        if not self.yes_bids and not self.yes_asks:
            return 0.5
        bid = self.best_yes_bid
        ask = self.best_yes_ask
        return (bid + ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.best_yes_ask - self.best_yes_bid

    @property
    def total_bid_volume(self) -> int:
        """Total volume on bid side."""
        return sum(level.quantity for level in self.yes_bids)

    @property
    def total_ask_volume(self) -> int:
        """Total volume on ask side."""
        return sum(level.quantity for level in self.yes_asks)


@dataclass
class BalanceData:
    """Account balance data from Kalshi."""

    balance: float  # Available balance in dollars
    portfolio_value: float = 0.0  # Total portfolio value
    payout: float = 0.0  # Pending payouts
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderData:
    """Order data from Kalshi."""

    order_id: str
    ticker: str
    side: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    order_type: str  # "limit" or "market"
    price: float
    quantity: int
    filled_quantity: int = 0
    remaining_quantity: int = 0
    status: str = "pending"  # pending, resting, canceled, executed
    created_time: datetime | None = None
    expiration_time: datetime | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionData:
    """Position data from Kalshi."""

    ticker: str
    market_id: str = ""
    position: int = 0  # Net position (positive = yes, negative = no)
    market_exposure: float = 0.0
    realized_pnl: float = 0.0
    resting_orders_count: int = 0
    total_cost: float = 0.0
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class FillData:
    """Fill/execution data from Kalshi.
    
    Note on Kalshi's API normalization:
    - Kalshi stores all fills from the YES perspective
    - 'price' is always the YES price in decimal form
    - 'yes_price' and 'no_price' are in cents and complement each other
    - When you "Buy NO at $0.44", the API returns:
      - side="no", action="sell", price=0.56 (YES price!)
      - yes_price=56 (cents), no_price=44 (cents)
    """

    trade_id: str
    ticker: str
    order_id: str
    side: str  # "yes" or "no" - which side of the market
    action: str  # "buy" or "sell" - from YES perspective
    price: float  # Price in dollars (this is the YES price!)
    count: int  # Number of contracts
    yes_price: float = 0.0  # YES price in dollars
    no_price: float = 0.0  # NO price in dollars
    is_taker: bool = False
    created_time: datetime | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)
    
    @property
    def actual_price(self) -> float:
        """Get the actual price paid/received based on side.
        
        Returns:
            The price in dollars for the side (YES or NO) that was traded.
        """
        if self.side == "yes":
            return self.yes_price if self.yes_price > 0 else self.price
        else:
            return self.no_price if self.no_price > 0 else (1.0 - self.price)


class IKalshiClient(ABC):
    """Interface for Kalshi trading client (read-only methods)."""

    @abstractmethod
    async def list_events(
        self,
        series_ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[EventData], str | None]:
        """List events, optionally filtered.

        Args:
            series_ticker: Filter by series ticker.
            status: Filter by status (open, closed, settled).
            limit: Maximum results to return.
            cursor: Pagination cursor.

        Returns:
            Tuple of (events list, next cursor or None).
        """
        ...

    @abstractmethod
    async def list_markets(
        self,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        status: str | None = None,
        tickers: list[str] | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[MarketData], str | None]:
        """List markets, optionally filtered.

        Args:
            event_ticker: Filter by event ticker.
            series_ticker: Filter by series ticker.
            status: Filter by status (open, closed, settled).
            tickers: Filter by specific tickers.
            limit: Maximum results to return.
            cursor: Pagination cursor.

        Returns:
            Tuple of (markets list, next cursor or None).
        """
        ...

    @abstractmethod
    async def get_market(self, ticker: str) -> MarketData | None:
        """Get a specific market by ticker.

        Args:
            ticker: Market ticker.

        Returns:
            Market data or None if not found.
        """
        ...

    @abstractmethod
    async def get_orderbook(self, ticker: str, depth: int = 10) -> OrderbookData:
        """Get orderbook for a market.

        Args:
            ticker: Market ticker.
            depth: Number of price levels to return.

        Returns:
            Orderbook data.
        """
        ...

    @abstractmethod
    async def get_balance(self) -> BalanceData:
        """Get account balance.

        Returns:
            Balance data.
        """
        ...

    @abstractmethod
    async def get_positions(
        self,
        limit: int = 100,
        cursor: str | None = None,
        settlement_status: str | None = None,
    ) -> tuple[list[PositionData], str | None]:
        """Get current positions.

        Args:
            limit: Maximum results to return.
            cursor: Pagination cursor.
            settlement_status: Filter by settlement status.

        Returns:
            Tuple of (positions list, next cursor or None).
        """
        ...

    @abstractmethod
    async def get_fills(
        self,
        ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[FillData], str | None]:
        """Get fills/trades.

        Args:
            ticker: Filter by market ticker.
            min_ts: Minimum timestamp (seconds since epoch).
            max_ts: Maximum timestamp (seconds since epoch).
            limit: Maximum results to return.
            cursor: Pagination cursor.

        Returns:
            Tuple of (fills list, next cursor or None).
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if API is accessible.

        Returns:
            True if API is healthy.
        """
        ...


class IKalshiTradingClient(IKalshiClient):
    """Extended interface for Kalshi trading client with order placement."""

    @abstractmethod
    async def place_limit_order(
        self,
        ticker: str,
        side: str,
        action: str,
        price: float,
        quantity: int,
        client_order_id: str | None = None,
        expiration_ts: int | None = None,
    ) -> OrderData:
        """Place a limit order.

        Args:
            ticker: Market ticker.
            side: "yes" or "no".
            action: "buy" or "sell".
            price: Limit price (0.01 to 0.99).
            quantity: Number of contracts.
            client_order_id: Optional client-specified order ID.
            expiration_ts: Optional Unix timestamp in milliseconds when order expires.

        Returns:
            Order data with order_id and status.

        Raises:
            KalshiApiError: If order placement fails.
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancellation was successful.

        Raises:
            KalshiApiError: If cancellation fails.
        """
        ...

    @abstractmethod
    async def list_orders(
        self,
        ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[OrderData], str | None]:
        """List orders, optionally filtered.

        Args:
            ticker: Filter by market ticker.
            status: Filter by status (resting, canceled, executed, pending).
            limit: Maximum results to return.
            cursor: Pagination cursor.

        Returns:
            Tuple of (orders list, next cursor or None).
        """
        ...

    @abstractmethod
    async def get_order(self, order_id: str) -> OrderData | None:
        """Get a specific order by ID.

        Args:
            order_id: Order ID.

        Returns:
            Order data or None if not found.
        """
        ...
