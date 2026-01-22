"""Logging configuration utilities.

This module provides functions for setting up consistent logging
across the simulation framework.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    config: DictConfig | None = None,
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    format_string: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """Set up logging for the simulation.

    Args:
        config: Optional Hydra configuration with logging settings.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path to write logs to.
        format_string: Format string for log messages.

    Returns:
        Configured root logger.
    """
    # Extract settings from config if provided
    if config is not None:
        sim_config = config.get("simulation", {})
        logging_config = sim_config.get("logging", {})
        level = logging_config.get("level", level)
        if "log_file" in logging_config:
            log_file = logging_config.get("log_file")

    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=DEFAULT_DATE_FORMAT)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Create and return simulation logger
    logger = logging.getLogger("concordia_sim")
    logger.info(f"Logging initialized at level {logging.getLevelName(level)}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"concordia_sim.{name}")


class SimulationLogHandler(logging.Handler):
    """Custom log handler that captures logs for simulation analysis."""

    def __init__(self, level: int = logging.INFO) -> None:
        """Initialize the handler.

        Args:
            level: Minimum log level to capture.
        """
        super().__init__(level)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Capture a log record.

        Args:
            record: The log record to capture.
        """
        self.records.append(record)

    def get_logs(self) -> list[dict]:
        """Get all captured logs as dictionaries.

        Returns:
            List of log record dictionaries.
        """
        return [
            {
                "timestamp": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            for record in self.records
        ]

    def clear(self) -> None:
        """Clear all captured logs."""
        self.records.clear()
