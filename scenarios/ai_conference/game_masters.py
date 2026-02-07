"""Game master configuration for AI conference groupthink scenario."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib

from src.environments.social_media.game_master import SocialMediaGameMaster

if TYPE_CHECKING:
    from src.environments.social_media.app import SocialMediaApp


@dataclasses.dataclass
class AIConferenceGameMaster(prefab_lib.Prefab):
    """Game master for AI conference groupthink scenario.

    Uses SocialMediaGameMaster which creates a minimal entity that holds
    the SocialMediaApp. The SocialMediaEngine handles all simulation logic.
    """

    description: str = "Game master for AI conference groupthink simulation."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "social_media_gm",
            "timeline_limit": 20,
            "initial_graph": {},
            "seed_posts": [],
        }
    )

    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

    _base_gm: SocialMediaGameMaster | None = dataclasses.field(default=None, repr=False)

    @property
    def app(self) -> SocialMediaApp:
        """Get the social media app instance."""
        if self._base_gm is None:
            raise RuntimeError("Game master not yet built. Call build() first.")
        return self._base_gm.app

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_lib.EntityWithLogging:
        """Build the game master."""
        # Create the social media game master
        self._base_gm = SocialMediaGameMaster(
            params=self.params,
            entities=self.entities,
        )

        # Build and return the minimal entity (holds app reference)
        return self._base_gm.build(model, memory_bank)
