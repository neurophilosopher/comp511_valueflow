"""Base engine class for simulation execution.

This module provides the abstract base class for all simulation engines.
Engines control how simulation steps are executed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class BaseEngine(ABC):
    """Abstract base class for simulation engines.

    Engines are responsible for executing simulation steps according to
    a particular strategy (sequential, parallel, distributed, etc.).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the engine.

        Args:
            config: Optional engine configuration.
        """
        self._config = config or {}

    @property
    def config(self) -> dict[str, Any]:
        """Get the engine configuration."""
        return self._config

    @abstractmethod
    def execute_step(
        self,
        entities: list[Any],
        step_fn: Callable[[Any], Any],
    ) -> list[Any]:
        """Execute a single simulation step for all entities.

        Args:
            entities: List of entities to process.
            step_fn: Function to call for each entity.

        Returns:
            List of results from processing each entity.
        """

    @abstractmethod
    def setup(self) -> None:
        """Set up the engine before simulation starts."""

    @abstractmethod
    def teardown(self) -> None:
        """Clean up the engine after simulation ends."""
