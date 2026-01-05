"""Tests for live executor with kill-switch integration."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from footbe_trader.execution.alerts import AlertManager, AlertType
from footbe_trader.execution.kill_switch import KillSwitchConfig, KillSwitchManager
from footbe_trader.execution.live_executor import (
    ExecutionMode,
    KillSwitchTrippedError,
    LiveExecutor,
    LiveExecutorConfig,
    LiveTradingNotEnabledError,
)
from footbe_trader.kalshi.interfaces import OrderData
from footbe_trader.strategy.interfaces import Signal


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_modes_exist(self):
        """Test all expected modes exist."""
        assert ExecutionMode.DRY_RUN
        assert ExecutionMode.PAPER
        assert ExecutionMode.LIVE

    def test_mode_values(self):
        """Test mode values."""
        assert ExecutionMode.DRY_RUN.value == "dry_run"
        assert ExecutionMode.PAPER.value == "paper"
        assert ExecutionMode.LIVE.value == "live"


class TestLiveExecutorConfig:
    """Tests for LiveExecutorConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = LiveExecutorConfig()

        assert config.enable_live_trading is False
        assert config.mode == ExecutionMode.DRY_RUN
        assert config.require_environment_confirmation is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = LiveExecutorConfig(
            enable_live_trading=True,
            mode=ExecutionMode.LIVE,
            require_environment_confirmation=False,
        )

        assert config.enable_live_trading is True
        assert config.mode == ExecutionMode.LIVE

    def test_to_dict(self):
        """Test serialization."""
        config = LiveExecutorConfig()
        data = config.to_dict()

        assert "enable_live_trading" in data
        assert "mode" in data
        assert data["mode"] == "dry_run"

    def test_order_defaults(self):
        """Test order default values."""
        config = LiveExecutorConfig()

        assert config.max_order_quantity == 100
        assert config.max_order_value == 1000.0


class TestLiveTradingNotEnabledError:
    """Tests for LiveTradingNotEnabledError."""

    def test_error_message(self):
        """Test error message."""
        error = LiveTradingNotEnabledError("Test reason")
        assert "Test reason" in str(error)


class TestKillSwitchTrippedError:
    """Tests for KillSwitchTrippedError."""

    def test_error_message(self):
        """Test error message."""
        from footbe_trader.execution.kill_switch import KillSwitchReason
        
        error = KillSwitchTrippedError(KillSwitchReason.DAILY_LOSS_LIMIT, "Lost $500")
        assert "daily_loss_limit" in str(error).lower() or "Lost $500" in str(error)


class TestLiveExecutorInitialization:
    """Tests for LiveExecutor initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        mock_client = MagicMock()
        executor = LiveExecutor(mock_client)

        assert executor.config.mode == ExecutionMode.DRY_RUN
        assert executor.kill_switch is not None
        assert executor.alerts is not None

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        mock_client = MagicMock()
        config = LiveExecutorConfig(mode=ExecutionMode.PAPER)

        executor = LiveExecutor(mock_client, config)

        assert executor.config.mode == ExecutionMode.PAPER

    def test_initialization_live_mode_requires_flags(self):
        """Test that live mode requires explicit flags."""
        mock_client = MagicMock()
        config = LiveExecutorConfig(
            mode=ExecutionMode.LIVE,
            enable_live_trading=False,  # Not enabled
        )

        with pytest.raises(LiveTradingNotEnabledError):
            LiveExecutor(mock_client, config)

    def test_initialization_live_mode_requires_env(self):
        """Test that live mode requires environment confirmation."""
        mock_client = MagicMock()
        config = LiveExecutorConfig(
            mode=ExecutionMode.LIVE,
            enable_live_trading=True,
            require_environment_confirmation=True,
        )

        # Without env var, should raise
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LiveTradingNotEnabledError):
                LiveExecutor(mock_client, config)

    def test_initialization_live_mode_with_env(self):
        """Test that live mode works with env confirmation."""
        mock_client = MagicMock()
        config = LiveExecutorConfig(
            mode=ExecutionMode.LIVE,
            enable_live_trading=True,
            require_environment_confirmation=True,
        )

        with patch.dict(os.environ, {"FOOTBE_ENABLE_LIVE_TRADING": "true"}):
            executor = LiveExecutor(mock_client, config)
            assert executor.is_live is True


class TestLiveExecutorModeChecks:
    """Tests for mode checking properties."""

    def test_dry_run_property(self):
        """Test dry_run property."""
        mock_client = MagicMock()
        config = LiveExecutorConfig(mode=ExecutionMode.DRY_RUN)
        executor = LiveExecutor(mock_client, config)

        assert executor.dry_run is True
        assert executor.is_live is False

    def test_is_live_property(self):
        """Test is_live property."""
        mock_client = MagicMock()
        config = LiveExecutorConfig(
            mode=ExecutionMode.LIVE,
            enable_live_trading=True,
            require_environment_confirmation=False,
        )
        executor = LiveExecutor(mock_client, config)

        assert executor.dry_run is False
        assert executor.is_live is True


class TestLiveExecutorDryRunMode:
    """Tests for dry-run mode execution."""

    @pytest.mark.asyncio
    async def test_dry_run_does_not_send_orders(self):
        """Test that dry-run mode doesn't send real orders."""
        mock_client = AsyncMock()
        config = LiveExecutorConfig(mode=ExecutionMode.DRY_RUN)
        executor = LiveExecutor(mock_client, config)

        signal = Signal(
            market_id="TEST-1",
            side="yes",
            action="buy",
            target_price=0.45,
            quantity=10,
            edge=0.05,
            confidence=0.8,
            direction="long",
        )

        result = await executor.execute(signal)

        # Client should not be called
        mock_client.place_limit_order.assert_not_called()

        # Result should indicate dry-run
        assert result.success is True
        assert result.metadata.get("dry_run") is True
        assert "DRY" in result.order.order_id

    @pytest.mark.asyncio
    async def test_dry_run_creates_order_data(self):
        """Test that dry-run creates proper order data."""
        mock_client = AsyncMock()
        config = LiveExecutorConfig(mode=ExecutionMode.DRY_RUN)
        executor = LiveExecutor(mock_client, config)

        signal = Signal(
            market_id="TEST-1",
            side="yes",
            action="buy",
            target_price=0.45,
            quantity=10,
            edge=0.05,
            confidence=0.8,
            direction="long",
        )

        result = await executor.execute(signal)

        assert result.order is not None
        assert result.order.ticker == "TEST-1"
        assert result.order.quantity == 10


class TestLiveExecutorPaperMode:
    """Tests for paper trading mode."""

    @pytest.mark.asyncio
    async def test_paper_mode_simulates(self):
        """Test that paper mode simulates execution."""
        mock_client = AsyncMock()
        config = LiveExecutorConfig(mode=ExecutionMode.PAPER)
        executor = LiveExecutor(mock_client, config)

        signal = Signal(
            market_id="TEST-1",
            side="yes",
            action="buy",
            target_price=0.45,
            quantity=10,
            edge=0.05,
            confidence=0.8,
            direction="long",
        )

        result = await executor.execute(signal)

        # Should still be simulated
        assert result.success is True


class TestLiveExecutorLiveMode:
    """Tests for live trading mode."""

    @pytest.mark.asyncio
    async def test_live_mode_sends_real_orders(self):
        """Test that live mode sends real orders."""
        mock_client = AsyncMock()
        mock_order = OrderData(
            order_id="LIVE-123",
            ticker="TEST-1",
            side="yes",
            action="buy",
            order_type="limit",
            price=0.45,  # Use float to match executor expectations
            quantity=10,
            filled_quantity=10,
            remaining_quantity=0,
            status="executed",
        )
        mock_client.place_limit_order.return_value = mock_order

        config = LiveExecutorConfig(
            enable_live_trading=True,
            mode=ExecutionMode.LIVE,
            require_environment_confirmation=False,
        )
        executor = LiveExecutor(mock_client, config)

        signal = Signal(
            market_id="TEST-1",
            side="yes",
            action="buy",
            target_price=0.45,
            quantity=10,
            edge=0.05,
            confidence=0.8,
            direction="long",
        )

        result = await executor.execute(signal)

        mock_client.place_limit_order.assert_called_once()
        assert result.success is True
        assert result.order.order_id == "LIVE-123"

    @pytest.mark.asyncio
    async def test_live_mode_handles_errors(self):
        """Test that live mode handles API errors."""
        mock_client = AsyncMock()
        mock_client.place_limit_order.side_effect = Exception("API Error")

        config = LiveExecutorConfig(
            enable_live_trading=True,
            mode=ExecutionMode.LIVE,
            require_environment_confirmation=False,
        )
        executor = LiveExecutor(mock_client, config)

        signal = Signal(
            market_id="TEST-1",
            side="yes",
            action="buy",
            target_price=0.45,
            quantity=10,
            edge=0.05,
            confidence=0.8,
            direction="long",
        )

        result = await executor.execute(signal)

        assert result.success is False
        assert "API Error" in result.error_message


class TestLiveExecutorKillSwitchIntegration:
    """Tests for kill-switch integration."""

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_execution(self):
        """Test that tripped kill-switch blocks execution."""
        mock_client = AsyncMock()
        config = LiveExecutorConfig(mode=ExecutionMode.DRY_RUN)

        executor = LiveExecutor(mock_client, config)
        executor.kill_switch.halt_trading("Test halt")

        signal = Signal(
            market_id="TEST-1",
            side="yes",
            action="buy",
            target_price=0.45,
            quantity=10,
            edge=0.05,
            confidence=0.8,
            direction="long",
        )

        result = await executor.execute(signal)

        assert result.success is False
        assert "kill-switch" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_kill_switch_allows_after_resume(self):
        """Test that execution works after resuming."""
        mock_client = AsyncMock()
        config = LiveExecutorConfig(mode=ExecutionMode.DRY_RUN)

        executor = LiveExecutor(mock_client, config)
        executor.kill_switch.halt_trading("Test halt")
        executor.kill_switch.resume_trading()

        signal = Signal(
            market_id="TEST-1",
            side="yes",
            action="buy",
            target_price=0.45,
            quantity=10,
            edge=0.05,
            confidence=0.8,
            direction="long",
        )

        result = await executor.execute(signal)
        assert result.success is True


class TestLiveExecutorAlertIntegration:
    """Tests for alert system integration."""

    @pytest.mark.asyncio
    async def test_alerts_on_execution(self):
        """Test that alerts are sent on execution."""
        mock_client = AsyncMock()
        config = LiveExecutorConfig(mode=ExecutionMode.DRY_RUN)

        executor = LiveExecutor(mock_client, config)

        signal = Signal(
            market_id="TEST-1",
            side="yes",
            action="buy",
            target_price=0.45,
            quantity=10,
            edge=0.05,
            confidence=0.8,
            direction="long",
        )

        await executor.execute(signal)

        # Should have alert for order
        assert len(executor.alerts.history) >= 1

    @pytest.mark.asyncio
    async def test_alerts_on_kill_switch_block(self):
        """Test that alert is sent when kill-switch blocks."""
        mock_client = AsyncMock()
        config = LiveExecutorConfig(mode=ExecutionMode.DRY_RUN)

        executor = LiveExecutor(mock_client, config)
        executor.kill_switch.halt_trading("Test halt")

        signal = Signal(
            market_id="TEST-1",
            side="yes",
            action="buy",
            target_price=0.45,
            quantity=10,
            edge=0.05,
            confidence=0.8,
            direction="long",
        )

        await executor.execute(signal)

        # Should have kill-switch alert
        alerts = executor.alerts.get_recent_alerts(
            alert_type=AlertType.KILL_SWITCH_TRIPPED
        )
        assert len(alerts) >= 1


class TestLiveExecutorStartStop:
    """Tests for start/stop functionality."""

    def test_start_executor(self):
        """Test starting executor."""
        mock_client = MagicMock()
        executor = LiveExecutor(mock_client)

        executor.start()

        # Should have trading started alert
        started_alerts = [
            a for a in executor.alerts.history
            if a.alert_type == AlertType.TRADING_STARTED
        ]
        assert len(started_alerts) == 1

    def test_stop_executor(self):
        """Test stopping executor."""
        mock_client = MagicMock()
        executor = LiveExecutor(mock_client)

        executor.start()
        executor.stop("test reason")

        # Should have trading stopped alert
        stopped_alerts = [
            a for a in executor.alerts.history
            if a.alert_type == AlertType.TRADING_STOPPED
        ]
        assert len(stopped_alerts) == 1

    def test_start_is_idempotent(self):
        """Test that start can be called multiple times."""
        mock_client = MagicMock()
        executor = LiveExecutor(mock_client)

        executor.start()
        executor.start()  # Second call should be no-op

        started_alerts = [
            a for a in executor.alerts.history
            if a.alert_type == AlertType.TRADING_STARTED
        ]
        assert len(started_alerts) == 1


class TestLiveExecutorExecutionCount:
    """Tests for execution counting."""

    @pytest.mark.asyncio
    async def test_execution_count_increments(self):
        """Test that execution count increments."""
        mock_client = AsyncMock()
        config = LiveExecutorConfig(mode=ExecutionMode.DRY_RUN)
        executor = LiveExecutor(mock_client, config)

        signal = Signal(
            market_id="TEST-1",
            side="yes",
            action="buy",
            target_price=0.45,
            quantity=10,
            edge=0.05,
            confidence=0.8,
            direction="long",
        )

        assert executor._execution_count == 0

        await executor.execute(signal)
        assert executor._execution_count == 1

        await executor.execute(signal)
        assert executor._execution_count == 2
