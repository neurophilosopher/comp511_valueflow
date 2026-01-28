"""Logging configuration utilities.

This module provides functions for setting up consistent logging
across the simulation framework, including stdout/stderr capture.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

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


class TeeStdout:
    """Redirect stdout to both console and a file.

    This allows capturing verbose print() output from Concordia's engine
    while still displaying it in the terminal.
    """

    def __init__(self, file_path: Path, original_stdout: TextIO | None = None) -> None:
        """Initialize the Tee.

        Args:
            file_path: Path to the log file to write to.
            original_stdout: The original stdout stream (defaults to sys.stdout).
        """
        self.file_path = file_path
        self.original_stdout: TextIO = (
            original_stdout if original_stdout is not None else sys.__stdout__
        )  # type: ignore[assignment]
        self.file: TextIO | None = None

    def __enter__(self) -> TeeStdout:
        """Start capturing stdout."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.file_path.open("a", encoding="utf-8")
        sys.stdout = self  # type: ignore[assignment]
        return self

    def __exit__(self, *args: object) -> None:
        """Stop capturing and restore original stdout."""
        sys.stdout = self.original_stdout
        if self.file:
            self.file.close()
            self.file = None

    def write(self, data: str) -> int:
        """Write data to both console and file.

        Args:
            data: The string data to write.

        Returns:
            Number of characters written.
        """
        self.original_stdout.write(data)
        if self.file:
            self.file.write(data)
        return len(data)

    def flush(self) -> None:
        """Flush both streams."""
        self.original_stdout.flush()
        if self.file:
            self.file.flush()


def setup_stdout_capture(log_file: str | Path) -> TeeStdout:
    """Set up stdout capture to a log file.

    Args:
        log_file: Path to the log file.

    Returns:
        TeeStdout context manager (use with 'with' statement).
    """
    return TeeStdout(Path(log_file))


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
