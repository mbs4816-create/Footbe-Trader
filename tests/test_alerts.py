"""Tests for alert system."""

from datetime import timedelta

import pytest

from footbe_trader.common.time_utils import utc_now
from footbe_trader.execution.alerts import (
    Alert,
    AlertConfig,
    AlertLevel,
    AlertManager,
    AlertType,
    CallbackAlertHandler,
    LogAlertHandler,
)


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_levels_exist(self):
        """Test all expected levels exist."""
        assert AlertLevel.DEBUG
        assert AlertLevel.INFO
        assert AlertLevel.WARNING
        assert AlertLevel.ERROR
        assert AlertLevel.CRITICAL


class TestAlertType:
    """Tests for AlertType enum."""

    def test_types_exist(self):
        """Test key alert types exist."""
        assert AlertType.TRADE_EXECUTED
        assert AlertType.ORDER_PLACED
        assert AlertType.ORDER_FAILED
        assert AlertType.KILL_SWITCH_TRIPPED
        assert AlertType.MAPPING_FAILURE
        assert AlertType.API_ERROR


class TestAlert:
    """Tests for Alert dataclass."""

    def test_creation(self):
        """Test alert creation."""
        alert = Alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Test Trade",
            message="Trade executed",
        )

        assert alert.alert_type == AlertType.TRADE_EXECUTED
        assert alert.level == AlertLevel.INFO
        assert alert.title == "Test Trade"
        assert alert.message == "Trade executed"
        assert alert.acknowledged is False

    def test_to_dict(self):
        """Test alert serialization."""
        alert = Alert(
            alert_type=AlertType.ORDER_PLACED,
            level=AlertLevel.INFO,
            title="Order",
            message="Order placed",
            data={"ticker": "TEST-1"},
        )

        data = alert.to_dict()

        assert data["alert_type"] == "order_placed"
        assert data["level"] == "info"
        assert data["title"] == "Order"
        assert data["data"]["ticker"] == "TEST-1"

    def test_acknowledge(self):
        """Test acknowledging an alert."""
        alert = Alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.ERROR,
            title="Error",
            message="Something failed",
        )

        alert.acknowledge()

        assert alert.acknowledged is True
        assert alert.acknowledged_at is not None


class TestAlertConfig:
    """Tests for AlertConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = AlertConfig()

        assert config.enabled is True
        assert config.log_alerts is True
        assert config.file_alerts is False
        assert config.alert_on_every_trade is True
        assert config.alert_on_kill_switch is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = AlertConfig(
            enabled=False,
            file_alerts=True,
            file_path="/tmp/alerts.log",
        )

        assert config.enabled is False
        assert config.file_alerts is True
        assert config.file_path == "/tmp/alerts.log"

    def test_to_dict(self):
        """Test serialization."""
        config = AlertConfig()
        data = config.to_dict()

        assert "enabled" in data
        assert "log_alerts" in data
        assert "alert_on_kill_switch" in data


class TestLogAlertHandler:
    """Tests for LogAlertHandler."""

    def test_handles_above_min_level(self):
        """Test handler processes alerts at or above min level."""
        handler = LogAlertHandler(min_level=AlertLevel.WARNING)

        # Should handle warning and above
        alert = Alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.ERROR,
            title="Error",
            message="Test",
        )

        # Should not raise
        handler.handle(alert)

    def test_skips_below_min_level(self):
        """Test handler skips alerts below min level."""
        handler = LogAlertHandler(min_level=AlertLevel.WARNING)

        alert = Alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Info",
            message="Test",
        )

        # Should not raise, but also not log (we can't easily verify)
        handler.handle(alert)


class TestCallbackAlertHandler:
    """Tests for CallbackAlertHandler."""

    def test_calls_callback(self):
        """Test callback is called."""
        received_alerts = []

        def callback(alert: Alert) -> None:
            received_alerts.append(alert)

        handler = CallbackAlertHandler(callback)

        alert = Alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Trade",
            message="Test",
        )
        handler.handle(alert)

        assert len(received_alerts) == 1
        assert received_alerts[0] == alert

    def test_respects_min_level(self):
        """Test callback respects minimum level."""
        received_alerts = []

        def callback(alert: Alert) -> None:
            received_alerts.append(alert)

        handler = CallbackAlertHandler(callback, min_level=AlertLevel.ERROR)

        # Info alert should be skipped
        info_alert = Alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Info",
            message="Test",
        )
        handler.handle(info_alert)
        assert len(received_alerts) == 0

        # Error alert should be handled
        error_alert = Alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.ERROR,
            title="Error",
            message="Test",
        )
        handler.handle(error_alert)
        assert len(received_alerts) == 1


class TestAlertManager:
    """Tests for AlertManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = AlertManager()

        assert len(manager.handlers) >= 1  # At least log handler
        assert len(manager.history) == 0

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = AlertConfig(log_alerts=False, file_alerts=False)
        manager = AlertManager(config)

        assert len(manager.handlers) == 0

    def test_add_handler(self):
        """Test adding custom handler."""
        config = AlertConfig(log_alerts=False)
        manager = AlertManager(config)

        received = []
        handler = CallbackAlertHandler(lambda a: received.append(a))
        manager.add_handler(handler)

        manager.send(
            Alert(
                alert_type=AlertType.TRADE_EXECUTED,
                level=AlertLevel.INFO,
                title="Test",
                message="Test",
            )
        )

        assert len(received) == 1

    def test_send_adds_to_history(self):
        """Test that send adds alerts to history."""
        manager = AlertManager()

        manager.send(
            Alert(
                alert_type=AlertType.TRADE_EXECUTED,
                level=AlertLevel.INFO,
                title="Test",
                message="Test",
            )
        )

        assert len(manager.history) == 1

    def test_disabled_manager_skips_send(self):
        """Test that disabled manager doesn't send."""
        config = AlertConfig(enabled=False, log_alerts=False)
        manager = AlertManager(config)

        manager.send(
            Alert(
                alert_type=AlertType.TRADE_EXECUTED,
                level=AlertLevel.INFO,
                title="Test",
                message="Test",
            )
        )

        assert len(manager.history) == 0

    def test_duplicate_suppression(self):
        """Test duplicate alert suppression."""
        config = AlertConfig(suppress_duplicates_seconds=60)
        manager = AlertManager(config)

        alert1 = Alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Same Title",
            message="Test",
        )
        alert2 = Alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Same Title",
            message="Test 2",
        )

        manager.send(alert1)
        manager.send(alert2)  # Should be suppressed

        assert len(manager.history) == 1


class TestAlertManagerConvenienceMethods:
    """Tests for AlertManager convenience methods."""

    def test_alert_trade_executed(self):
        """Test trade execution alert."""
        manager = AlertManager()

        manager.alert_trade_executed(
            ticker="TEST-1",
            side="yes",
            action="buy",
            price=0.45,
            quantity=10,
        )

        assert len(manager.history) == 1
        assert manager.history[0].alert_type == AlertType.TRADE_EXECUTED

    def test_alert_trade_with_pnl(self):
        """Test trade alert with P&L."""
        manager = AlertManager()

        manager.alert_trade_executed(
            ticker="TEST-1",
            side="yes",
            action="sell",
            price=0.55,
            quantity=10,
            pnl=100.0,
        )

        assert "P&L" in manager.history[0].message

    def test_alert_order_placed(self):
        """Test order placed alert."""
        manager = AlertManager()

        manager.alert_order_placed(
            ticker="TEST-1",
            side="yes",
            action="buy",
            price=0.45,
            quantity=10,
            order_id="ORD-123",
        )

        assert len(manager.history) == 1
        assert manager.history[0].alert_type == AlertType.ORDER_PLACED

    def test_alert_order_dry_run(self):
        """Test dry-run order alert."""
        manager = AlertManager()

        manager.alert_order_placed(
            ticker="TEST-1",
            side="yes",
            action="buy",
            price=0.45,
            quantity=10,
            order_id="DRY-123",
            dry_run=True,
        )

        assert "[DRY-RUN]" in manager.history[0].title

    def test_alert_order_failed(self):
        """Test order failure alert."""
        manager = AlertManager()

        manager.alert_order_failed(
            ticker="TEST-1",
            error="Insufficient balance",
        )

        assert len(manager.history) == 1
        assert manager.history[0].alert_type == AlertType.ORDER_FAILED
        assert manager.history[0].level == AlertLevel.ERROR

    def test_alert_kill_switch_tripped(self):
        """Test kill-switch trip alert."""
        manager = AlertManager()

        manager.alert_kill_switch_tripped(
            reason="daily_loss_limit",
            message="Lost $500",
        )

        assert len(manager.history) == 1
        assert manager.history[0].alert_type == AlertType.KILL_SWITCH_TRIPPED
        assert manager.history[0].level == AlertLevel.CRITICAL

    def test_alert_mapping_failure(self):
        """Test mapping failure alert."""
        manager = AlertManager()

        manager.alert_mapping_failure(
            fixture_id=12345,
            reason="No matching markets found",
        )

        assert len(manager.history) == 1
        assert manager.history[0].alert_type == AlertType.MAPPING_FAILURE

    def test_alert_api_error(self):
        """Test API error alert."""
        manager = AlertManager()

        manager.alert_api_error(
            endpoint="/markets",
            error="Connection timeout",
            status_code=500,
        )

        assert len(manager.history) == 1
        assert manager.history[0].alert_type == AlertType.API_ERROR

    def test_alert_trading_started(self):
        """Test trading started alert."""
        manager = AlertManager()

        manager.alert_trading_started(mode="live")

        assert len(manager.history) == 1
        assert manager.history[0].alert_type == AlertType.TRADING_STARTED

    def test_alert_trading_stopped(self):
        """Test trading stopped alert."""
        manager = AlertManager()

        manager.alert_trading_stopped(reason="kill-switch tripped")

        assert len(manager.history) == 1
        assert manager.history[0].alert_type == AlertType.TRADING_STOPPED


class TestAlertManagerHistory:
    """Tests for alert history methods."""

    def test_get_recent_alerts(self):
        """Test getting recent alerts."""
        manager = AlertManager()

        for i in range(10):
            manager.send(
                Alert(
                    alert_type=AlertType.TRADE_EXECUTED,
                    level=AlertLevel.INFO,
                    title=f"Alert {i}",
                    message="Test",
                )
            )

        recent = manager.get_recent_alerts(count=5)
        assert len(recent) == 5
        assert recent[-1].title == "Alert 9"

    def test_get_recent_by_type(self):
        """Test filtering recent alerts by type."""
        manager = AlertManager()

        manager.send(
            Alert(
                alert_type=AlertType.TRADE_EXECUTED,
                level=AlertLevel.INFO,
                title="Info",
                message="Test",
            )
        )
        manager.send(
            Alert(
                alert_type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.ERROR,
                title="Error",
                message="Test",
            )
        )

        errors = manager.get_recent_alerts(alert_type=AlertType.SYSTEM_ERROR)
        assert len(errors) == 1
        assert errors[0].alert_type == AlertType.SYSTEM_ERROR

    def test_get_recent_by_level(self):
        """Test filtering by minimum level."""
        manager = AlertManager()

        manager.send(
            Alert(
                alert_type=AlertType.TRADE_EXECUTED,
                level=AlertLevel.INFO,
                title="Info",
                message="Test",
            )
        )
        manager.send(
            Alert(
                alert_type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.ERROR,
                title="Error",
                message="Test",
            )
        )

        errors = manager.get_recent_alerts(min_level=AlertLevel.ERROR)
        assert len(errors) == 1

    def test_get_unacknowledged(self):
        """Test getting unacknowledged alerts."""
        manager = AlertManager()

        alert1 = Alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Alert 1",
            message="Test",
        )
        alert2 = Alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Alert 2",
            message="Test",
        )

        manager.send(alert1)
        manager.send(alert2)
        manager.history[0].acknowledge()

        unacked = manager.get_unacknowledged_alerts()
        assert len(unacked) == 1
        assert unacked[0].title == "Alert 2"

    def test_acknowledge_all(self):
        """Test acknowledging all alerts."""
        manager = AlertManager()

        for i in range(5):
            manager.send(
                Alert(
                    alert_type=AlertType.TRADE_EXECUTED,
                    level=AlertLevel.INFO,
                    title=f"Alert {i}",
                    message="Test",
                )
            )

        count = manager.acknowledge_all()

        assert count == 5
        assert len(manager.get_unacknowledged_alerts()) == 0

    def test_history_pruning(self):
        """Test that history is pruned to max size."""
        config = AlertConfig(suppress_duplicates_seconds=0)
        manager = AlertManager(config)

        for i in range(1100):
            manager.send(
                Alert(
                    alert_type=AlertType.TRADE_EXECUTED,
                    level=AlertLevel.INFO,
                    title=f"Alert {i}",
                    message="Test",
                )
            )

        # Should be pruned to 1000
        assert len(manager.history) == 1000
