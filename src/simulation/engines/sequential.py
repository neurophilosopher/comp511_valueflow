"""Sequential execution engine.

This module provides a simple sequential engine that processes
entities one at a time in order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.simulation.engines.base import BaseEngine

if TYPE_CHECKING:
    from collections.abc import Callable


class SequentialEngine(BaseEngine):
    """Engine that executes simulation steps sequentially.

    Processes each entity one at a time in the order provided.
    This is the simplest execution strategy and useful for
    debugging or when order matters.
    """

    def execute_step(
        self,
        entities: list[Any],
        step_fn: Callable[[Any], Any],
    ) -> list[Any]:
        """Execute a step for all entities sequentially.

        Args:
            entities: List of entities to process.
            step_fn: Function to call for each entity.

        Returns:
            List of results in the same order as entities.
        """
        results = []
        for entity in entities:
            result = step_fn(entity)
            results.append(result)
        return results

    def setup(self) -> None:
        """Set up the sequential engine (no-op)."""
        pass

    def teardown(self) -> None:
        """Clean up the sequential engine (no-op)."""
        pass
