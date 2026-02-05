"""Game master configuration for misinformation scenario."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

from src.environments.social_media.concordia_gm import ConcordiaSocialMediaGameMaster

if TYPE_CHECKING:
    from src.environments.social_media.app import SocialMediaApp


@dataclasses.dataclass
class MisinformationGameMaster(prefab_lib.Prefab):
    """Game master for misinformation spread scenario.

    This wraps ConcordiaSocialMediaGameMaster to work with run_experiment.py.
    Uses parallel execution where all agents act each step.
    """

    description: str = "Game master for social media misinformation simulation."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "social_media_gm",
            "timeline_limit": 20,
            "initial_graph": {},
            "seed_posts": [],
        }
    )

    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

    _base_gm: ConcordiaSocialMediaGameMaster | None = dataclasses.field(
        default=None, repr=False
    )

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
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the game master."""
        # Create the Concordia-compatible game master
        self._base_gm = ConcordiaSocialMediaGameMaster(
            params=self.params,
            entities=self.entities,
        )

        # Build and return the entity
        return self._base_gm.build(model, memory_bank)
