"""Social media game master prefab."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib

from src.environments.social_media.app import SocialMediaApp


@dataclasses.dataclass
class SocialMediaGameMaster(prefab_lib.Prefab):
    """Game master for social media simulation.

    This prefab creates a minimal game master that holds the SocialMediaApp
    state. The actual simulation logic is handled by SocialMediaEngine.

    The game master is primarily used for:
    - Holding the SocialMediaApp instance
    - Providing a name for logging
    - Checkpoint state management
    """

    description: str = "Game master for social media simulation."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "social_media_gm",
            "timeline_limit": 20,
            "initial_graph": {},  # user -> list of users they follow
            "seed_posts": [],  # list of {author, content, tags}
        }
    )

    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

    # The app instance (created during build, shared with engine)
    _app: SocialMediaApp | None = dataclasses.field(default=None, repr=False)

    @property
    def app(self) -> SocialMediaApp:
        """Get the social media app instance."""
        if self._app is None:
            raise RuntimeError("Game master not yet built. Call build() first.")
        return self._app

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_lib.EntityWithLogging:
        """Build the social media game master.

        Note: The model and memory_bank are not used for social media GM
        since observations and resolution are deterministic. They are
        accepted for API compatibility with other prefabs.

        Args:
            model: Language model (unused).
            memory_bank: Memory bank (unused).

        Returns:
            A minimal entity for the game master.
        """
        name = str(self.params.get("name", "social_media_gm"))
        timeline_limit = int(self.params.get("timeline_limit", 20))
        initial_graph = self.params.get("initial_graph", {})
        seed_posts = self.params.get("seed_posts", [])

        # Create the social media app
        self._app = SocialMediaApp()

        # Get player names
        player_names = [e.name for e in self.entities]

        # Initialize users (ensure all players exist in the app)
        for player in player_names:
            self._app._ensure_user(player)

        # Set up initial follower graph
        for user, follows in initial_graph.items():
            if isinstance(follows, list | tuple):
                for target in follows:
                    self._app.follow(str(user), str(target))

        # Create seed posts (step 0)
        self._app.current_step = 0
        for seed in seed_posts:
            author = str(seed.get("author", ""))
            content = str(seed.get("content", ""))
            tags = seed.get("tags", [])
            if author and content:
                # Ensure author exists
                self._app._ensure_user(author)
                self._app.post(author, content, tags=tags)

        # Store timeline limit for use by engine
        self._timeline_limit = timeline_limit

        # Create a minimal entity (the actual logic is in SocialMediaEngine)
        # We use a simple no-op act component since the engine handles everything
        return _MinimalGameMasterEntity(name=name, app=self._app)


class _MinimalGameMasterEntity(entity_lib.EntityWithLogging):
    """Minimal game master entity for social media simulation.

    This entity just holds state and provides logging interface.
    The actual simulation logic is in SocialMediaEngine.
    """

    def __init__(self, name: str, app: SocialMediaApp) -> None:
        """Initialize minimal game master.

        Args:
            name: Name of the game master.
            app: Social media app instance.
        """
        self._name = name
        self._app = app
        self._last_log: dict[str, Any] = {}

    @functools.cached_property
    def name(self) -> str:
        """Get entity name."""
        return self._name

    @property
    def app(self) -> SocialMediaApp:
        """Get the social media app instance."""
        return self._app

    def act(self, action_spec: entity_lib.ActionSpec = entity_lib.DEFAULT_ACTION_SPEC) -> str:
        """No-op act - engine handles all logic."""
        return ""

    def observe(self, observation: str) -> None:
        """No-op observe - engine handles all logic."""
        pass

    def get_last_log(self) -> dict[str, Any]:
        """Get last log entry."""
        return self._last_log

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for checkpoints."""
        return {
            "name": self._name,
            "app_state": self._app.to_dict(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> _MinimalGameMasterEntity:
        """Restore from checkpoint state."""
        app = SocialMediaApp.from_dict(state["app_state"])
        return cls(name=state["name"], app=app)
