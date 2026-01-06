"""Telegram notifications for trading agent.

Sends formatted messages about agent runs, trades, and strategy narratives.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any

import httpx

from footbe_trader.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = True
    
    @property
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.bot_token and self.chat_id)


class TelegramNotifier:
    """Sends notifications via Telegram bot.
    
    Usage:
        notifier = TelegramNotifier(config)
        await notifier.send_message("Hello from the agent!")
        await notifier.send_run_summary(summary, narrative)
    """
    
    BASE_URL = "https://api.telegram.org"
    
    def __init__(self, config: TelegramConfig):
        """Initialize notifier.
        
        Args:
            config: Telegram configuration with bot_token and chat_id.
        """
        self.config = config
        self._client: httpx.AsyncClient | None = None
    
    async def __aenter__(self) -> "TelegramNotifier":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if self._client is None:
            # Create a one-shot client if not in context manager
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> bool:
        """Send a message via Telegram.
        
        Args:
            text: Message text (supports HTML formatting).
            parse_mode: "HTML" or "Markdown".
            disable_notification: If True, send silently.
            
        Returns:
            True if message sent successfully.
        """
        if not self.config.is_configured:
            logger.debug("telegram_not_configured")
            return False
        
        if not self.config.enabled:
            logger.debug("telegram_disabled")
            return False
        
        url = f"{self.BASE_URL}/bot{self.config.bot_token}/sendMessage"
        
        payload = {
            "chat_id": self.config.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }
        
        try:
            response = await self.client.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info("telegram_message_sent", length=len(text))
                return True
            else:
                logger.warning(
                    "telegram_send_failed",
                    status=response.status_code,
                    response=response.text[:200],
                )
                return False
                
        except Exception as e:
            logger.error("telegram_error", error=str(e))
            return False
    
    async def send_run_summary(
        self,
        run_id: int,
        mode: str,
        fixtures_evaluated: int,
        markets_evaluated: int,
        decisions_made: int,
        orders_placed: int,
        orders_filled: int,
        realized_pnl: float,
        unrealized_pnl: float,
        total_exposure: float,
        position_count: int,
        narrative: str,
        duration_seconds: float | None = None,
    ) -> bool:
        """Send a formatted run summary.
        
        Args:
            Various run statistics and narrative.
            
        Returns:
            True if sent successfully.
        """
        # Build emoji based on P&L
        total_pnl = realized_pnl + unrealized_pnl
        if total_pnl > 10:
            pnl_emoji = "🚀"
        elif total_pnl > 0:
            pnl_emoji = "📈"
        elif total_pnl < -10:
            pnl_emoji = "📉"
        elif total_pnl < 0:
            pnl_emoji = "⚠️"
        else:
            pnl_emoji = "➡️"
        
        # Format duration
        duration_str = ""
        if duration_seconds:
            if duration_seconds < 60:
                duration_str = f" ({duration_seconds:.1f}s)"
            else:
                duration_str = f" ({duration_seconds/60:.1f}m)"
        
        message = f"""<b>🤖 Agent Run #{run_id} Complete</b>{duration_str}

<b>Mode:</b> {mode.upper()}
<b>Time:</b> {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}

<b>📊 Activity Summary</b>
• Fixtures evaluated: {fixtures_evaluated}
• Markets evaluated: {markets_evaluated}
• Decisions made: {decisions_made}
• Orders placed: {orders_placed}
• Orders filled: {orders_filled}

<b>💰 Portfolio Status</b> {pnl_emoji}
• Realized P&L: ${realized_pnl:+.2f}
• Unrealized P&L: ${unrealized_pnl:+.2f}
• <b>Total P&L: ${total_pnl:+.2f}</b>
• Exposure: ${total_exposure:.2f}
• Open positions: {position_count}

<b>📝 Strategy Narrative</b>
{narrative}
"""
        return await self.send_message(message.strip())
    
    async def send_error_alert(
        self,
        error_message: str,
        run_id: int | None = None,
    ) -> bool:
        """Send an error alert.
        
        Args:
            error_message: Description of the error.
            run_id: Optional run ID where error occurred.
            
        Returns:
            True if sent successfully.
        """
        run_info = f" (Run #{run_id})" if run_id else ""
        
        message = f"""<b>🚨 Agent Error{run_info}</b>

<b>Time:</b> {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}

<b>Error:</b>
<code>{error_message[:500]}</code>
"""
        return await self.send_message(message.strip())
    
    async def send_trade_alert(
        self,
        action: str,
        ticker: str,
        side: str,
        quantity: int,
        price: float,
        edge: float,
        fixture_info: str,
    ) -> bool:
        """Send alert about a trade execution.
        
        Args:
            action: "BUY" or "SELL".
            ticker: Market ticker.
            side: "YES" or "NO".
            quantity: Number of contracts.
            price: Execution price.
            edge: Calculated edge.
            fixture_info: Human-readable fixture description.
            
        Returns:
            True if sent successfully.
        """
        emoji = "🟢" if action == "BUY" else "🔴"
        
        message = f"""{emoji} <b>Trade Executed</b>

<b>{action} {quantity}x {side}</b> @ ${price:.2f}
<b>Ticker:</b> <code>{ticker}</code>
<b>Edge:</b> {edge:+.1%}
<b>Match:</b> {fixture_info}
"""
        return await self.send_message(message.strip())


class NarrativeGenerator:
    """Generates human-readable narratives about trading activity.
    
    Explains what the agent did and why, making it easy to understand
    the strategy's behavior without diving into logs.
    """
    
    def generate_run_narrative(
        self,
        fixtures_evaluated: int,
        markets_evaluated: int,
        decisions_made: int,
        orders_placed: int,
        orders_filled: int,
        skipped_reasons: dict[str, int],
        trades_by_outcome: dict[str, int],
        live_games: list[str],
        cancelled_stale: int,
        edge_summary: dict[str, float],
    ) -> str:
        """Generate a narrative summary of a run.
        
        Args:
            fixtures_evaluated: Number of fixtures checked.
            markets_evaluated: Number of markets analyzed.
            decisions_made: Total decisions made.
            orders_placed: Orders sent to market.
            orders_filled: Orders that executed.
            skipped_reasons: Dict of skip reasons and counts.
            trades_by_outcome: Trades by outcome type (home_win, draw, away_win).
            live_games: List of games currently live.
            cancelled_stale: Number of stale orders cancelled.
            edge_summary: Dict with avg/max edge found.
            
        Returns:
            Human-readable narrative string.
        """
        paragraphs = []
        
        # Opening summary
        if orders_placed == 0:
            if fixtures_evaluated == 0:
                paragraphs.append(
                    "No fixtures found to evaluate this run. This could mean no games "
                    "are scheduled in our tradeable window, or market mappings haven't "
                    "been refreshed recently."
                )
            else:
                # Explain why no orders
                paragraphs.append(
                    f"Evaluated {fixtures_evaluated} fixtures across {markets_evaluated} "
                    f"markets but found no opportunities meeting our criteria."
                )
                
                if skipped_reasons:
                    top_reasons = sorted(
                        skipped_reasons.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3]
                    reason_parts = [f"{r}: {c}" for r, c in top_reasons]
                    paragraphs.append(
                        f"Main skip reasons: {', '.join(reason_parts)}."
                    )
        else:
            # Successful trades
            paragraphs.append(
                f"Found edge in {orders_placed} market(s) and placed orders. "
                f"{orders_filled} filled immediately."
            )
            
            if trades_by_outcome:
                outcome_parts = [
                    f"{count} {outcome.replace('_', ' ')}"
                    for outcome, count in trades_by_outcome.items()
                    if count > 0
                ]
                if outcome_parts:
                    paragraphs.append(f"Bet types: {', '.join(outcome_parts)}.")
        
        # Edge analysis
        if edge_summary:
            avg_edge = edge_summary.get("avg", 0)
            max_edge = edge_summary.get("max", 0)
            if max_edge > 0:
                paragraphs.append(
                    f"Best edge found: {max_edge:.1%}. "
                    f"Average edge across opportunities: {avg_edge:.1%}."
                )
        
        # Live games awareness
        if live_games:
            if len(live_games) == 1:
                paragraphs.append(f"Currently live: {live_games[0]}.")
            else:
                paragraphs.append(
                    f"{len(live_games)} games currently live. "
                    f"Monitoring for in-game opportunities."
                )
        
        # Stale order cleanup
        if cancelled_stale > 0:
            paragraphs.append(
                f"Cancelled {cancelled_stale} stale order(s) where market price "
                f"diverged significantly from our limits."
            )
        
        # Strategy health
        if orders_placed == 0 and decisions_made > 10:
            paragraphs.append(
                "Strategy is actively filtering. Consider reviewing edge thresholds "
                "if this pattern continues across multiple runs."
            )
        
        return " ".join(paragraphs)
    
    def generate_position_narrative(
        self,
        positions: list[dict[str, Any]],
        total_pnl: float,
    ) -> str:
        """Generate narrative about current positions.
        
        Args:
            positions: List of position dicts with ticker, pnl, entry_price, etc.
            total_pnl: Total P&L across all positions.
            
        Returns:
            Human-readable position narrative.
        """
        if not positions:
            return "No open positions currently. Bankroll is fully available."
        
        parts = []
        
        winning = [p for p in positions if p.get("unrealized_pnl", 0) > 0]
        losing = [p for p in positions if p.get("unrealized_pnl", 0) < 0]
        
        parts.append(f"Holding {len(positions)} position(s).")
        
        if winning and losing:
            parts.append(
                f"{len(winning)} in profit, {len(losing)} underwater."
            )
        elif winning:
            parts.append("All positions currently profitable.")
        elif losing:
            parts.append(
                "All positions currently showing losses. "
                "Will exit if stop-loss triggers or edge flips."
            )
        
        if total_pnl > 0:
            parts.append(f"Net unrealized gain: ${total_pnl:.2f}.")
        elif total_pnl < 0:
            parts.append(f"Net unrealized loss: ${abs(total_pnl):.2f}.")
        
        return " ".join(parts)
