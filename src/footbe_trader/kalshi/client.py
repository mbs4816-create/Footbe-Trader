"""Kalshi API client implementation with authentication, rate limiting, and retries."""

import asyncio
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

import httpx

from footbe_trader.common.config import KalshiConfig
from footbe_trader.common.logging import get_logger
from footbe_trader.kalshi.auth import KalshiAuth
from footbe_trader.kalshi.interfaces import (
    BalanceData,
    EventData,
    FillData,
    IKalshiTradingClient,
    MarketData,
    OrderbookData,
    OrderbookLevel,
    OrderData,
    PositionData,
)
from footbe_trader.kalshi.rate_limit import RateLimiter, RequestLogger, RetryConfig

logger = get_logger(__name__)


class KalshiApiError(Exception):
    """Exception for Kalshi API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class KalshiClient(IKalshiTradingClient):
    """Kalshi API client with authentication, rate limiting, and retries.

    This client implements both read and write methods for fetching market data,
    orderbooks, positions, account information, and placing/managing orders.
    """

    def __init__(self, config: KalshiConfig):
        """Initialize client.

        Args:
            config: Kalshi API configuration.
        """
        self.config = config
        self.auth = KalshiAuth(config)
        self.rate_limiter = RateLimiter(
            requests_per_second=config.rate_limit_per_second,
            burst_size=10,
        )
        self.retry_config = RetryConfig(
            max_retries=config.max_retries,
            base_delay=config.retry_backoff_base,
        )
        self.request_logger = RequestLogger("kalshi_api")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "KalshiClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.config.effective_base_url,
            timeout=self.config.timeout_seconds,
        )
        return self

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async context manager.")
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated request with rate limiting and retries.

        Args:
            method: HTTP method.
            path: API path (e.g., "/portfolio/balance").
            params: Query parameters.
            json_body: JSON body for POST/PUT requests.

        Returns:
            Response JSON as dictionary.

        Raises:
            KalshiApiError: If request fails after retries.
        """
        # Build full path for signing
        full_path = f"/trade-api/v2{path}"
        if params:
            query_string = urlencode({k: v for k, v in params.items() if v is not None})
            if query_string:
                full_path = f"{full_path}?{query_string}"

        last_error: Exception | None = None

        for attempt in range(self.retry_config.max_retries + 1):
            # Rate limiting
            wait_time = await self.rate_limiter.acquire()
            if wait_time > 0:
                logger.debug("rate_limit_wait", wait_seconds=wait_time)

            # Sign request
            signed = self.auth.sign_request(method, full_path)

            # Log request
            url = f"{self.config.effective_base_url}{path}"
            start_time = self.request_logger.log_request(method, url, params)

            try:
                response = await self.client.request(
                    method=method,
                    url=path,
                    params=params,
                    json=json_body,
                    headers=signed.headers,
                )

                # Log response
                self.request_logger.log_response(
                    method, url, response.status_code, start_time
                )

                # Handle response
                if response.status_code in (200, 201):
                    return response.json()
                elif response.status_code == 204:
                    return {}
                elif response.status_code in self.retry_config.retryable_status_codes:
                    error_body = self._safe_json(response)
                    last_error = KalshiApiError(
                        f"HTTP {response.status_code}: {error_body}",
                        status_code=response.status_code,
                        response_body=error_body,
                    )

                    if attempt < self.retry_config.max_retries:
                        delay = self.retry_config.get_delay(attempt)
                        self.request_logger.log_retry(
                            method, url, attempt + 1, delay, f"HTTP {response.status_code}"
                        )
                        await asyncio.sleep(delay)
                        continue
                else:
                    error_body = self._safe_json(response)
                    raise KalshiApiError(
                        f"HTTP {response.status_code}: {error_body}",
                        status_code=response.status_code,
                        response_body=error_body,
                    )

            except httpx.RequestError as e:
                last_error = e
                self.request_logger.log_response(
                    method, url, 0, start_time, error=str(e)
                )

                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    self.request_logger.log_retry(
                        method, url, attempt + 1, delay, str(e)
                    )
                    await asyncio.sleep(delay)
                    continue

        # All retries exhausted
        raise KalshiApiError(
            f"Request failed after {self.retry_config.max_retries + 1} attempts"
        ) from last_error

    def _safe_json(self, response: httpx.Response) -> dict[str, Any]:
        """Safely parse JSON response."""
        try:
            return response.json()
        except Exception:
            return {"raw": response.text[:500]}

    # --- Event Methods ---

    async def list_events(
        self,
        series_ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[EventData], str | None]:
        """List events, optionally filtered."""
        params: dict[str, Any] = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        data = await self._request("GET", "/events", params=params)
        events = [self._parse_event(e) for e in data.get("events", [])]
        next_cursor = data.get("cursor")
        return events, next_cursor

    def _parse_event(self, data: dict[str, Any]) -> EventData:
        """Parse event data from API response."""
        return EventData(
            event_ticker=data.get("event_ticker", ""),
            title=data.get("title", ""),
            subtitle=data.get("subtitle", ""),
            category=data.get("category", ""),
            sub_category=data.get("sub_category", ""),
            series_ticker=data.get("series_ticker", ""),
            mutually_exclusive=data.get("mutually_exclusive", True),
            strike_date=self._parse_datetime(data.get("strike_date")),
            raw_data=data,
        )

    # --- Market Methods ---

    async def list_markets(
        self,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        status: str | None = None,
        tickers: list[str] | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[MarketData], str | None]:
        """List markets, optionally filtered."""
        params: dict[str, Any] = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if tickers:
            params["tickers"] = ",".join(tickers)
        if cursor:
            params["cursor"] = cursor

        data = await self._request("GET", "/markets", params=params)
        markets = [self._parse_market(m) for m in data.get("markets", [])]
        next_cursor = data.get("cursor")
        return markets, next_cursor

    async def get_market(self, ticker: str) -> MarketData | None:
        """Get a specific market by ticker."""
        try:
            data = await self._request("GET", f"/markets/{ticker}")
            market_data = data.get("market", data)
            return self._parse_market(market_data)
        except KalshiApiError as e:
            if e.status_code == 404:
                return None
            raise

    async def get_events_for_series(
        self, series_ticker: str, status: str | None = "open"
    ) -> list[EventData]:
        """Get all events for a series.
        
        Handles pagination to get all events.
        
        Args:
            series_ticker: Series ticker (e.g., "KXNBAGAME", "KXEPLGAME").
            status: Filter by status (e.g., "open", "closed").
            
        Returns:
            List of all events in the series.
        """
        all_events: list[EventData] = []
        cursor: str | None = None
        
        while True:
            events, next_cursor = await self.list_events(
                series_ticker=series_ticker,
                status=status,
                limit=100,
                cursor=cursor,
            )
            all_events.extend(events)
            
            if not next_cursor or not events:
                break
            cursor = next_cursor
        
        return all_events

    async def get_markets_for_event(
        self, event_ticker: str, status: str | None = None
    ) -> list[MarketData]:
        """Get all markets for an event.
        
        Args:
            event_ticker: Event ticker.
            status: Optional status filter.
            
        Returns:
            List of markets for the event.
        """
        markets, _ = await self.list_markets(
            event_ticker=event_ticker,
            status=status,
            limit=100,
        )
        return markets

    def _parse_market(self, data: dict[str, Any]) -> MarketData:
        """Parse market data from API response."""
        return MarketData(
            ticker=data.get("ticker", ""),
            event_ticker=data.get("event_ticker", ""),
            title=data.get("title", ""),
            subtitle=data.get("subtitle", ""),
            status=data.get("status", ""),
            open_time=self._parse_datetime(data.get("open_time")),
            close_time=self._parse_datetime(data.get("close_time")),
            expiration_time=self._parse_datetime(data.get("expiration_time")),
            yes_bid=self._cents_to_dollars(data.get("yes_bid", 0)),
            yes_ask=self._cents_to_dollars(data.get("yes_ask", 0)),
            no_bid=self._cents_to_dollars(data.get("no_bid", 0)),
            no_ask=self._cents_to_dollars(data.get("no_ask", 0)),
            last_price=self._cents_to_dollars(data.get("last_price", 0)),
            previous_yes_bid=self._cents_to_dollars(data.get("previous_yes_bid", 0)),
            previous_yes_ask=self._cents_to_dollars(data.get("previous_yes_ask", 0)),
            volume=data.get("volume", 0),
            volume_24h=data.get("volume_24h", 0),
            open_interest=data.get("open_interest", 0),
            result=data.get("result", ""),
            raw_data=data,
        )

    # --- Orderbook Methods ---

    async def get_orderbook(self, ticker: str, depth: int = 10) -> OrderbookData:
        """Get orderbook for a market."""
        params: dict[str, Any] = {"depth": depth}
        data = await self._request("GET", f"/markets/{ticker}/orderbook", params=params)
        return self._parse_orderbook(ticker, data)

    def _parse_orderbook(self, ticker: str, data: dict[str, Any]) -> OrderbookData:
        """Parse orderbook data from API response."""
        orderbook = data.get("orderbook", data)

        def parse_levels(levels: list[list[Any]] | None) -> list[OrderbookLevel]:
            if not levels:
                return []
            return [
                OrderbookLevel(
                    price=self._cents_to_dollars(level[0]),
                    quantity=level[1],
                )
                for level in levels
                if len(level) >= 2
            ]

        return OrderbookData(
            ticker=ticker,
            yes_bids=parse_levels(orderbook.get("yes", [])),
            yes_asks=parse_levels(orderbook.get("no", [])),  # yes asks = no bids
            no_bids=parse_levels(orderbook.get("no", [])),
            no_asks=parse_levels(orderbook.get("yes", [])),  # no asks = yes bids
            timestamp=datetime.now(UTC),
            raw_data=data,
        )

    # --- Portfolio Methods ---

    async def get_balance(self) -> BalanceData:
        """Get account balance."""
        data = await self._request("GET", "/portfolio/balance")
        return BalanceData(
            balance=self._cents_to_dollars(data.get("balance", 0)),
            portfolio_value=self._cents_to_dollars(data.get("portfolio_value", 0)),
            payout=self._cents_to_dollars(data.get("payout", 0)),
            raw_data=data,
        )

    async def get_positions(
        self,
        limit: int | None = None,
        cursor: str | None = None,
        settlement_status: str | None = None,
    ) -> tuple[list[PositionData], str | None]:
        """Get current positions."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        if settlement_status:
            params["settlement_status"] = settlement_status

        data = await self._request("GET", "/portfolio/positions", params=params if params else None)
        positions = [
            self._parse_position(p)
            for p in data.get("market_positions", [])
        ]
        next_cursor = data.get("cursor")
        return positions, next_cursor

    def _parse_position(self, data: dict[str, Any]) -> PositionData:
        """Parse position data from API response."""
        return PositionData(
            ticker=data.get("ticker", ""),
            market_id=data.get("market_id", ""),
            position=data.get("position", 0),
            market_exposure=self._cents_to_dollars(data.get("market_exposure", 0)),
            realized_pnl=self._cents_to_dollars(data.get("realized_pnl", 0)),
            resting_orders_count=data.get("resting_orders_count", 0),
            total_cost=self._cents_to_dollars(data.get("total_cost", 0)),
            raw_data=data,
        )

    async def get_fills(
        self,
        ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> tuple[list[FillData], str | None]:
        """Get fills/trades."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if ticker:
            params["ticker"] = ticker
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor

        data = await self._request("GET", "/portfolio/fills", params=params if params else None)
        fills = [self._parse_fill(f) for f in data.get("fills", [])]
        next_cursor = data.get("cursor")
        return fills, next_cursor

    def _parse_fill(self, data: dict[str, Any]) -> FillData:
        """Parse fill data from API response."""
        # Note: price in fills is already in decimal format (e.g., 0.26), not cents
        # Use yes_price/no_price for cents, but 'price' is already decimal
        price = data.get("price", 0)
        if isinstance(price, (int, float)) and price < 1:
            # Already in decimal format
            price_dollars = float(price)
        else:
            # Some older API responses might have cents
            price_dollars = self._cents_to_dollars(price)
        
        return FillData(
            trade_id=data.get("trade_id", ""),
            ticker=data.get("ticker", ""),
            order_id=data.get("order_id", ""),
            side=data.get("side", ""),
            action=data.get("action", ""),
            price=price_dollars,
            count=data.get("count", 0),
            is_taker=data.get("is_taker", False),
            created_time=self._parse_datetime(data.get("created_time")),
            raw_data=data,
        )

    # --- Health Check ---

    async def health_check(self) -> bool:
        """Check if API is accessible."""
        try:
            await self.get_balance()
            return True
        except Exception as e:
            logger.warning("health_check_failed", error=str(e))
            return False

    # --- Utility Methods ---

    def _cents_to_dollars(self, cents: int | float | None) -> float:
        """Convert cents to dollars."""
        if cents is None:
            return 0.0
        return float(cents) / 100.0

    def _parse_datetime(self, value: str | None) -> datetime | None:
        """Parse ISO datetime string."""
        if not value:
            return None
        try:
            # Handle various ISO formats
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    # --- Order Methods ---

    async def place_limit_order(
        self,
        ticker: str,
        side: str,
        action: str,
        price: float,
        quantity: int,
        client_order_id: str | None = None,
    ) -> OrderData:
        """Place a limit order.

        Args:
            ticker: Market ticker.
            side: "yes" or "no".
            action: "buy" or "sell".
            price: Limit price (0.01 to 0.99).
            quantity: Number of contracts.
            client_order_id: Optional client-specified order ID.

        Returns:
            Order data with order_id and status.

        Raises:
            KalshiApiError: If order placement fails.
        """
        # Validate inputs
        if side not in ("yes", "no"):
            raise KalshiApiError(f"Invalid side: {side}. Must be 'yes' or 'no'.")
        if action not in ("buy", "sell"):
            raise KalshiApiError(f"Invalid action: {action}. Must be 'buy' or 'sell'.")
        if not 0.01 <= price <= 0.99:
            raise KalshiApiError(f"Invalid price: {price}. Must be between 0.01 and 0.99.")
        if quantity < 1:
            raise KalshiApiError(f"Invalid quantity: {quantity}. Must be at least 1.")

        # Convert price to cents
        price_cents = int(round(price * 100))

        body: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "type": "limit",
            "count": quantity,
            "yes_price": price_cents if side == "yes" else None,
            "no_price": price_cents if side == "no" else None,
        }

        if client_order_id:
            body["client_order_id"] = client_order_id

        # Remove None values
        body = {k: v for k, v in body.items() if v is not None}

        logger.info(
            "placing_limit_order",
            ticker=ticker,
            side=side,
            action=action,
            price=price,
            quantity=quantity,
        )

        data = await self._request("POST", "/portfolio/orders", json_body=body)
        order = data.get("order", data)

        logger.info(
            "order_placed",
            order_id=order.get("order_id"),
            status=order.get("status"),
        )

        return self._parse_order(order)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancellation was successful.

        Raises:
            KalshiApiError: If cancellation fails.
        """
        logger.info("canceling_order", order_id=order_id)

        try:
            await self._request("DELETE", f"/portfolio/orders/{order_id}")
            logger.info("order_canceled", order_id=order_id)
            return True
        except KalshiApiError as e:
            if e.status_code == 404:
                logger.warning("order_not_found", order_id=order_id)
                return False
            raise

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
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        data = await self._request("GET", "/portfolio/orders", params=params)
        orders = [self._parse_order(o) for o in data.get("orders", [])]
        next_cursor = data.get("cursor")
        return orders, next_cursor

    async def get_order(self, order_id: str) -> OrderData | None:
        """Get a specific order by ID.

        Args:
            order_id: Order ID.

        Returns:
            Order data or None if not found.
        """
        try:
            data = await self._request("GET", f"/portfolio/orders/{order_id}")
            order = data.get("order", data)
            return self._parse_order(order)
        except KalshiApiError as e:
            if e.status_code == 404:
                return None
            raise

    def _parse_order(self, data: dict[str, Any]) -> OrderData:
        """Parse order data from API response."""
        return OrderData(
            order_id=data.get("order_id", ""),
            ticker=data.get("ticker", ""),
            side=data.get("side", ""),
            action=data.get("action", ""),
            order_type=data.get("type", "limit"),
            price=self._cents_to_dollars(
                data.get("yes_price") or data.get("no_price") or 0
            ),
            quantity=data.get("initial_count", data.get("count", 0)),
            filled_quantity=data.get("fill_count", data.get("filled_count", 0)),
            remaining_quantity=data.get("remaining_count", 0),
            status=data.get("status", ""),
            created_time=self._parse_datetime(data.get("created_time")),
            expiration_time=self._parse_datetime(data.get("expiration_time")),
            raw_data=data,
        )
