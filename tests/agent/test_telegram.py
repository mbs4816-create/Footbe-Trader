"""Tests for Telegram notification module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from footbe_trader.agent.telegram import (
    NarrativeGenerator,
    TelegramConfig,
    TelegramNotifier,
)


class TestTelegramConfig:
    """Tests for TelegramConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TelegramConfig()
        assert config.bot_token == ""
        assert config.chat_id == ""
        assert config.enabled is True

    def test_is_configured_false_when_empty(self):
        """Test is_configured returns False when credentials empty."""
        config = TelegramConfig()
        assert config.is_configured is False

    def test_is_configured_false_when_partial(self):
        """Test is_configured returns False when only token set."""
        config = TelegramConfig(bot_token="token")
        assert config.is_configured is False

        config = TelegramConfig(chat_id="12345")
        assert config.is_configured is False

    def test_is_configured_true_when_complete(self):
        """Test is_configured returns True when both set."""
        config = TelegramConfig(bot_token="token", chat_id="12345")
        assert config.is_configured is True


class TestTelegramNotifier:
    """Tests for TelegramNotifier."""

    def test_init_with_config(self):
        """Test initialization with config."""
        config = TelegramConfig(bot_token="token", chat_id="12345")
        notifier = TelegramNotifier(config)
        assert notifier.config == config

    @pytest.mark.asyncio
    async def test_send_message_skipped_when_not_configured(self):
        """Test message sending is skipped when not configured."""
        config = TelegramConfig()  # Not configured
        notifier = TelegramNotifier(config)
        
        # Should not raise, just return False
        result = await notifier.send_message("Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_skipped_when_disabled(self):
        """Test message sending is skipped when disabled."""
        config = TelegramConfig(
            bot_token="token",
            chat_id="12345",
            enabled=False,
        )
        notifier = TelegramNotifier(config)
        
        result = await notifier.send_message("Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_makes_request(self):
        """Test message sending makes HTTP request."""
        config = TelegramConfig(bot_token="token", chat_id="12345")
        notifier = TelegramNotifier(config)
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"ok": True}
            
            result = await notifier.send_message("Test message")
            
            assert result is True
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "token" in call_args[0][0]  # URL contains token
            assert call_args[1]["json"]["chat_id"] == "12345"
            assert call_args[1]["json"]["text"] == "Test message"

    @pytest.mark.asyncio
    async def test_send_run_summary(self):
        """Test sending run summary."""
        config = TelegramConfig(bot_token="token", chat_id="12345")
        notifier = TelegramNotifier(config)
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"ok": True}
            
            result = await notifier.send_run_summary(
                run_id=1,
                mode="paper",
                fixtures_evaluated=5,
                markets_evaluated=10,
                decisions_made=3,
                orders_placed=2,
                orders_filled=1,
                realized_pnl=10.50,
                unrealized_pnl=5.25,
                total_exposure=50.00,
                position_count=2,
                narrative="Test narrative",
            )
            
            assert result is True
            call_args = mock_post.call_args
            message = call_args[1]["json"]["text"]
            assert "Run #1" in message
            assert "10" in message  # markets
            assert "10.50" in message  # realized pnl

    @pytest.mark.asyncio
    async def test_send_error_alert(self):
        """Test sending error alert."""
        config = TelegramConfig(bot_token="token", chat_id="12345")
        notifier = TelegramNotifier(config)
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"ok": True}
            
            result = await notifier.send_error_alert(
                error_message="Something went wrong",
                run_id=5,
            )
            
            assert result is True
            call_args = mock_post.call_args
            message = call_args[1]["json"]["text"]
            assert "Error" in message
            assert "Something went wrong" in message

    @pytest.mark.asyncio
    async def test_send_trade_alert(self):
        """Test sending trade alert."""
        config = TelegramConfig(bot_token="token", chat_id="12345")
        notifier = TelegramNotifier(config)
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"ok": True}
            
            result = await notifier.send_trade_alert(
                action="BUY",
                ticker="SOCCER-EPL-ARS-YES",
                side="YES",
                quantity=10,
                price=0.65,
                edge=0.08,
                fixture_info="Arsenal vs Chelsea",
            )
            
            assert result is True
            call_args = mock_post.call_args
            message = call_args[1]["json"]["text"]
            assert "Trade" in message
            assert "SOCCER-EPL-ARS-YES" in message
            assert "BUY" in message


class TestNarrativeGenerator:
    """Tests for NarrativeGenerator."""

    def test_generate_run_narrative_no_activity(self):
        """Test narrative for run with no activity."""
        generator = NarrativeGenerator()
        
        narrative = generator.generate_run_narrative(
            fixtures_evaluated=0,
            markets_evaluated=0,
            decisions_made=0,
            orders_placed=0,
            orders_filled=0,
            skipped_reasons={},
            trades_by_outcome={},
            live_games=[],
            cancelled_stale=0,
            edge_summary={},
        )
        
        assert isinstance(narrative, str)
        assert len(narrative) > 0
        # Should mention no fixtures
        assert "no fixtures" in narrative.lower() or "no games" in narrative.lower()

    def test_generate_run_narrative_with_skips(self):
        """Test narrative for run with skipped opportunities."""
        generator = NarrativeGenerator()
        
        narrative = generator.generate_run_narrative(
            fixtures_evaluated=10,
            markets_evaluated=20,
            decisions_made=15,
            orders_placed=0,
            orders_filled=0,
            skipped_reasons={
                "edge_too_low": 8,
                "price_stale": 5,
                "max_exposure": 2,
            },
            trades_by_outcome={},
            live_games=[],
            cancelled_stale=0,
            edge_summary={"avg": 0.02, "max": 0.04},
        )
        
        assert isinstance(narrative, str)
        assert len(narrative) > 0
        # Should mention the skip reasons
        assert "edge" in narrative.lower() or "skip" in narrative.lower()

    def test_generate_run_narrative_with_trades(self):
        """Test narrative for run with trading activity."""
        generator = NarrativeGenerator()
        
        narrative = generator.generate_run_narrative(
            fixtures_evaluated=10,
            markets_evaluated=20,
            decisions_made=5,
            orders_placed=3,
            orders_filled=2,
            skipped_reasons={},
            trades_by_outcome={"home_win": 2, "away_win": 1},
            live_games=[],
            cancelled_stale=0,
            edge_summary={"avg": 0.08, "max": 0.12},
        )
        
        assert isinstance(narrative, str)
        assert len(narrative) > 0
        # Should mention the activity
        assert "order" in narrative.lower() or "trade" in narrative.lower() or "fill" in narrative.lower()

    def test_generate_run_narrative_with_live_games(self):
        """Test narrative mentions live games."""
        generator = NarrativeGenerator()
        
        narrative = generator.generate_run_narrative(
            fixtures_evaluated=5,
            markets_evaluated=10,
            decisions_made=2,
            orders_placed=0,
            orders_filled=0,
            skipped_reasons={},
            trades_by_outcome={},
            live_games=["Arsenal vs Chelsea", "Liverpool vs Man City"],
            cancelled_stale=0,
            edge_summary={},
        )
        
        assert isinstance(narrative, str)
        assert "live" in narrative.lower()

    def test_generate_run_narrative_stale_orders(self):
        """Test narrative mentions cancelled stale orders."""
        generator = NarrativeGenerator()
        
        narrative = generator.generate_run_narrative(
            fixtures_evaluated=5,
            markets_evaluated=10,
            decisions_made=2,
            orders_placed=1,
            orders_filled=1,
            skipped_reasons={},
            trades_by_outcome={},
            live_games=[],
            cancelled_stale=3,
            edge_summary={},
        )
        
        assert isinstance(narrative, str)
        assert "cancel" in narrative.lower() or "stale" in narrative.lower()

    def test_generate_position_narrative_no_positions(self):
        """Test narrative for no positions."""
        generator = NarrativeGenerator()
        
        narrative = generator.generate_position_narrative(
            positions=[],
            total_pnl=0,
        )
        
        assert isinstance(narrative, str)
        assert "no" in narrative.lower() or "position" in narrative.lower()

    def test_generate_position_narrative_with_positions(self):
        """Test narrative with open positions."""
        generator = NarrativeGenerator()
        
        positions = [
            {
                "ticker": "TICKER-1",
                "unrealized_pnl": 5.50,
                "quantity": 10,
            },
            {
                "ticker": "TICKER-2",
                "unrealized_pnl": -2.25,
                "quantity": 5,
            },
        ]
        
        narrative = generator.generate_position_narrative(
            positions=positions,
            total_pnl=3.25,
        )
        
        assert isinstance(narrative, str)
        assert "2" in narrative  # 2 positions
        assert "profit" in narrative.lower() or "gain" in narrative.lower()

    def test_generate_position_narrative_all_losing(self):
        """Test narrative when all positions losing."""
        generator = NarrativeGenerator()
        
        positions = [
            {
                "ticker": "TICKER-1",
                "unrealized_pnl": -3.50,
                "quantity": 10,
            },
            {
                "ticker": "TICKER-2",
                "unrealized_pnl": -2.25,
                "quantity": 5,
            },
        ]
        
        narrative = generator.generate_position_narrative(
            positions=positions,
            total_pnl=-5.75,
        )
        
        assert isinstance(narrative, str)
        assert "loss" in narrative.lower()

