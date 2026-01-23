"""Base component class for entity capabilities.

This module provides the abstract base class for all entity components.
Components are modular pieces that can be attached to entities to provide
specific capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from concordia.typing import entity as entity_lib


class BaseComponent(ABC):
    """Abstract base class for entity components.

    Components provide modular capabilities that can be attached to entities.
    Examples include memory systems, perception handlers, and action selectors.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize the component.

        Args:
            name: Optional name for the component.
        """
        self._name = name or self.__class__.__name__
        self._entity: entity_lib.Entity | None = None

    @property
    def name(self) -> str:
        """Get the component name."""
        return self._name

    @property
    def entity(self) -> entity_lib.Entity | None:
        """Get the entity this component is attached to."""
        return self._entity

    def attach(self, entity: entity_lib.Entity) -> None:
        """Attach this component to an entity.

        Args:
            entity: The entity to attach to.
        """
        self._entity = entity
        self.on_attach()

    def detach(self) -> None:
        """Detach this component from its entity."""
        self.on_detach()
        self._entity = None

    def on_attach(self) -> None:
        """Called when the component is attached to an entity.

        Override this method to perform setup when attached.
        """
        pass

    def on_detach(self) -> None:
        """Called when the component is detached from an entity.

        Override this method to perform cleanup when detached.
        """
        pass

    @abstractmethod
    def update(self, context: dict[str, Any]) -> None:
        """Update the component state.

        Args:
            context: Context information for the update.
        """
