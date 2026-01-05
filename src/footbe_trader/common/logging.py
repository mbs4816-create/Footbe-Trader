"""Structured JSON logging setup."""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from footbe_trader.common.config import LoggingConfig


def _add_log_level(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add log level to event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configure structured logging.

    Args:
        config: Logging configuration. Uses defaults if not provided.
    """
    if config is None:
        config = LoggingConfig()

    # Determine processors based on format
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if config.format == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Add file handler if configured
    handlers: list[logging.Handler] = [handler]
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    root_logger.setLevel(getattr(logging, config.level.upper()))

    # Configure our package logger
    package_logger = logging.getLogger("footbe_trader")
    package_logger.setLevel(getattr(logging, config.level.upper()))


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name. Defaults to 'footbe_trader'.

    Returns:
        Bound logger instance.
    """
    logger_name = name or "footbe_trader"
    return structlog.get_logger(logger_name)
