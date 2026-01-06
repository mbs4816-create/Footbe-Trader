#!/usr/bin/env python3
"""Monitor for game kickoffs and cancel pre-game orders immediately.

This runs as a background task alongside the main agent, checking every
minute for games that are starting and cancelling any resting orders
for those games immediately (not waiting for the 15-minute agent cycle).
"""

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.common.logging import get_logger
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.storage.database import Database

logger = get_logger(__name__)


async def get_upcoming_kickoffs(db: Database, lookahead_minutes: int = 5) -> list[tuple[str, datetime]]:
    """Get games starting within the next N minutes.

    Args:
        db: Database connection.
        lookahead_minutes: How far ahead to look for kickoffs.

    Returns:
        List of (ticker, kickoff_time) tuples.
    """
    now = datetime.now(UTC)
    cutoff = now + timedelta(minutes=lookahead_minutes)

    cursor = db.connection.cursor()
    cursor.execute(
        """
        SELECT fmm.ticker_home_win, fmm.ticker_draw, fmm.ticker_away_win, f.kickoff_time
        FROM fixture_market_map fmm
        JOIN fixtures f ON fmm.fixture_id = f.id
        WHERE f.kickoff_time > ?
          AND f.kickoff_time <= ?
          AND f.status = 'NS'  -- Not started
        ORDER BY f.kickoff_time
        """,
        (now.isoformat(), cutoff.isoformat()),
    )

    # Extract all tickers for upcoming games
    results = []
    for row in cursor.fetchall():
        kickoff_time = datetime.fromisoformat(row[3])
        for ticker in [row[0], row[1], row[2]]:
            if ticker:  # Some might be None
                results.append((ticker, kickoff_time))

    return results


async def cancel_orders_for_game(client: KalshiClient, ticker_prefix: str) -> int:
    """Cancel all resting orders for a specific game.

    Args:
        client: Kalshi client.
        ticker_prefix: Ticker prefix to match (e.g., "WHUNFO" for West Ham vs Forest).

    Returns:
        Number of orders cancelled.
    """
    try:
        # Get all resting orders
        orders, _ = await client.list_orders(status="resting", limit=200)

        # Filter for this game
        game_orders = [o for o in orders if ticker_prefix in o.ticker]

        if not game_orders:
            return 0

        logger.info(
            "kickoff_detected",
            ticker_prefix=ticker_prefix,
            orders_found=len(game_orders),
        )

        # Cancel all orders for this game
        cancelled = 0
        for order in game_orders:
            try:
                success = await client.cancel_order(order.order_id)
                if success:
                    cancelled += 1
                    logger.info(
                        "kickoff_order_cancelled",
                        order_id=order.order_id,
                        ticker=order.ticker,
                        price=order.price,
                    )
            except Exception as e:
                logger.warning(
                    "kickoff_cancel_failed",
                    order_id=order.order_id,
                    ticker=order.ticker,
                    error=str(e),
                )

        return cancelled

    except Exception as e:
        logger.error("cancel_orders_for_game_failed", ticker_prefix=ticker_prefix, error=str(e))
        return 0


async def monitor_loop(config_path: str = "configs/dev.yaml", check_interval: int = 60):
    """Main monitoring loop.

    Args:
        config_path: Path to config file.
        check_interval: How often to check for kickoffs (seconds).
    """
    config = load_config(config_path)
    db = Database(config.database.path)
    db.connect()

    logger.info("kickoff_monitor_started", check_interval=check_interval)

    # Track games we've already processed to avoid double-cancellation
    processed_games = set()

    try:
        while True:
            try:
                async with KalshiClient(config.kalshi) as client:
                    # Get games starting in next 5 minutes
                    upcoming = await get_upcoming_kickoffs(db, lookahead_minutes=5)

                    for ticker, kickoff_time in upcoming:
                        # Extract game identifier from ticker (e.g., "WHUNFO" from "KXEPLGAME-26JAN06WHUNFO-NFO")
                        parts = ticker.split("-")
                        if len(parts) >= 2:
                            game_id = parts[1]  # e.g., "26JAN06WHUNFO"

                            # Skip if already processed
                            if game_id in processed_games:
                                continue

                            now = datetime.now(UTC)
                            time_to_kickoff = (kickoff_time - now).total_seconds()

                            # Cancel if within 3 minutes of kickoff
                            if time_to_kickoff <= 180:  # 3 minutes
                                logger.info(
                                    "game_starting_soon",
                                    game_id=game_id,
                                    kickoff_time=kickoff_time.isoformat(),
                                    seconds_to_kickoff=time_to_kickoff,
                                )

                                # Cancel all orders for this game
                                cancelled = await cancel_orders_for_game(client, game_id)

                                if cancelled > 0:
                                    logger.info(
                                        "kickoff_cancellation_complete",
                                        game_id=game_id,
                                        orders_cancelled=cancelled,
                                    )

                                # Mark as processed
                                processed_games.add(game_id)

                # Clean up old processed games (older than 2 hours)
                if len(processed_games) > 100:
                    processed_games.clear()

            except Exception as e:
                logger.error("monitor_loop_error", error=str(e))

            # Wait before next check
            await asyncio.sleep(check_interval)
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor for game kickoffs and cancel orders")
    parser.add_argument(
        "--config",
        default="configs/dev.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(monitor_loop(args.config, args.interval))
    except KeyboardInterrupt:
        logger.info("kickoff_monitor_stopped")
        sys.exit(0)
