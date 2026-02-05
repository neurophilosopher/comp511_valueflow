"""Concordia-compatible social media game master.

This module provides a game master that integrates with Concordia's
standard run_loop, enabling social media simulations to run via
run_experiment.py with the standard framework.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib

from src.environments.social_media.app import SocialMediaApp
from src.environments.social_media.engine import execute_action, parse_action


class SocialMediaResolution:
    """Event resolution component using SocialMediaApp.

    Resolves actions by parsing structured action strings and
    executing them on the SocialMediaApp deterministically.
    """

    def __init__(
        self,
        app: SocialMediaApp,
        player_names: Sequence[str],
    ) -> None:
        """Initialize resolution component.

        Args:
            app: The SocialMediaApp instance.
            player_names: Names of all players.
        """
        self._app = app
        self._player_names = list(player_names)
        self._action_results: dict[str, str] = {}

    def __call__(
        self,
        action_attempt: str,
        acting_entity: entity_lib.Entity,
    ) -> str:
        """Resolve an action attempt.

        Args:
            action_attempt: The raw action string from the entity.
            acting_entity: The entity performing the action.

        Returns:
            Result description of the action.
        """
        parsed = parse_action(action_attempt)
        result = execute_action(self._app, acting_entity.name, parsed)
        self._action_results[acting_entity.name] = result.message
        return result.message

    def get_last_result(self, player_name: str) -> str:
        """Get last action result for a player."""
        return self._action_results.get(player_name, "")


class SocialMediaObservation:
    """Observation component using SocialMediaApp feed.

    Generates observations from the player's social media feed
    instead of using LLM-generated descriptions.
    """

    def __init__(
        self,
        app: SocialMediaApp,
        timeline_limit: int = 20,
    ) -> None:
        """Initialize observation component.

        Args:
            app: The SocialMediaApp instance.
            timeline_limit: Maximum posts to show in feed.
        """
        self._app = app
        self._timeline_limit = timeline_limit

    def __call__(self, player_name: str) -> str:
        """Generate observation for a player.

        Args:
            player_name: Name of the player.

        Returns:
            Formatted feed as observation string.
        """
        return self._app.format_timeline(player_name, limit=self._timeline_limit)


class SocialMediaSwitchAct:
    """Custom SwitchAct for social media parallel execution.

    Unlike Concordia's SwitchAct which uses NextActing to select
    one entity, this returns all entities for parallel execution.
    """

    def __init__(
        self,
        app: SocialMediaApp,
        player_names: Sequence[str],
        timeline_limit: int = 20,
    ) -> None:
        """Initialize switch act component.

        Args:
            app: The SocialMediaApp instance.
            player_names: Names of all players.
            timeline_limit: Maximum posts to show in feed.
        """
        self._app = app
        self._player_names = list(player_names)
        self._timeline_limit = timeline_limit
        self._resolution = SocialMediaResolution(app, player_names)
        self._observation = SocialMediaObservation(app, timeline_limit)
        self._current_step = 0

    def get_action_spec(self) -> entity_lib.ActionSpec:
        """Get action specification for entities."""
        return entity_lib.ActionSpec(
            call_to_action=(
                f"Step {self._current_step}: What action do you take?\n"
                "Respond with: ACTION: <type> | TARGET: <id/username/none> | CONTENT: <text/none>"
            ),
            output_type=entity_lib.OutputType.FREE,
        )

    def __call__(
        self,
        entities: Sequence[entity_lib.Entity],
    ) -> str:
        """Called by the entity's act method, returns observation/action spec."""
        # This is a simplified implementation that works with run_loop
        # The actual orchestration happens via get_entity_names_to_act
        return ""

    def get_entity_names_to_act(self) -> Sequence[str]:
        """Return all entity names for parallel execution."""
        return self._player_names

    def make_observation(self, player_name: str) -> str:
        """Generate observation for a player."""
        return self._observation(player_name)

    def resolve(self, action_attempt: str, acting_entity: entity_lib.Entity) -> str:
        """Resolve an action attempt."""
        return self._resolution(action_attempt, acting_entity)

    def advance_step(self) -> None:
        """Advance to the next simulation step."""
        self._current_step += 1
        self._app.current_step = self._current_step


@dataclasses.dataclass
class ConcordiaSocialMediaGameMaster(prefab_lib.Prefab):
    """Concordia-compatible game master for social media simulation.

    This game master integrates with Concordia's standard framework,
    enabling social media simulations to run via run_experiment.py.

    Key differences from standard Concordia GMs:
    - Uses parallel execution (all agents act each step)
    - Observations come from SocialMediaApp feed (not LLM-generated)
    - Actions are resolved deterministically via execute_action
    """

    description: str = "Concordia-compatible social media game master."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "social_media_gm",
            "timeline_limit": 20,
            "initial_graph": {},
            "seed_posts": [],
        }
    )

    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

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
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the game master entity.

        Args:
            model: Language model (unused for deterministic resolution).
            memory_bank: Memory bank for the game master.

        Returns:
            EntityAgentWithLogging instance.
        """
        name = str(self.params.get("name", "social_media_gm"))
        timeline_limit = int(self.params.get("timeline_limit", 20))
        initial_graph = self.params.get("initial_graph", {})
        seed_posts = self.params.get("seed_posts", [])

        # Create the social media app
        self._app = SocialMediaApp()

        # Get player names
        player_names = [e.name for e in self.entities]

        # Initialize users
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
                self._app._ensure_user(author)
                self._app.post(author, content, tags=tags)

        # Build components
        components: dict[str, Any] = {}

        # Memory
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank
        )

        # GM Instructions
        instructions_key = "instructions"
        components[instructions_key] = gm_components.instructions.Instructions()

        # Social media description
        desc_key = "social_media_description"
        components[desc_key] = agent_components.constant.Constant(
            state=(
                "This is a social media simulation. Users can post, reply, like, "
                "boost, follow, and unfollow each other. All users act simultaneously "
                "each step."
            ),
            pre_act_label="\nSimulation setting",
        )

        # Player characters
        players_key = "players"
        components[players_key] = gm_components.instructions.PlayerCharacters(
            player_characters=player_names,
        )

        # Observation to memory
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        # Observation history
        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=100,
        )

        # Custom make observation from feed
        make_obs_key = gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
        components[make_obs_key] = _FeedBasedMakeObservation(
            app=self._app,
            player_names=player_names,
            timeline_limit=timeline_limit,
        )

        # Next actor - all players (parallel)
        next_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
        components[next_key] = _AllPlayersNextActing(player_names=player_names)

        # Event resolution using SocialMediaApp
        resolution_key = gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
        components[resolution_key] = _SocialMediaEventResolution(
            app=self._app,
            player_names=player_names,
        )

        # Create the switch act component
        context_keys = [instructions_key, desc_key, players_key]
        act_component = gm_components.switch_act.SwitchAct(
            model=model,
            entity_names=player_names,
            component_order=list(components.keys()),
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )


class _FeedBasedMakeObservation(entity_component.ContextComponent):
    """Make observation from social media feed instead of LLM.

    Inherits from ContextComponent to satisfy Concordia's entity interface.
    """

    def __init__(
        self,
        app: SocialMediaApp,
        player_names: Sequence[str],
        timeline_limit: int = 20,
    ) -> None:
        """Initialize feed-based observation maker.

        Args:
            app: SocialMediaApp instance.
            player_names: Names of all players.
            timeline_limit: Maximum posts in feed.
        """
        self._app = app
        self._player_names = list(player_names)
        self._timeline_limit = timeline_limit

    def make_observation(self, player_name: str) -> str:
        """Generate observation from feed.

        Args:
            player_name: Player to generate observation for.

        Returns:
            Formatted feed string.
        """
        return self._app.format_timeline(player_name, limit=self._timeline_limit)

    def get_state(self) -> entity_component.ComponentState:
        """Get component state for serialization."""
        return {"timeline_limit": self._timeline_limit}

    def set_state(self, state: entity_component.ComponentState) -> None:
        """Restore component state."""
        self._timeline_limit = int(state.get("timeline_limit", 20))


class _AllPlayersNextActing(entity_component.ContextComponent):
    """Return all players for parallel execution.

    Inherits from ContextComponent to satisfy Concordia's entity interface.
    """

    def __init__(self, player_names: Sequence[str]) -> None:
        """Initialize with player names.

        Args:
            player_names: Names of all players.
        """
        self._player_names = list(player_names)

    def __call__(self) -> Sequence[str]:
        """Return all players.

        Returns:
            List of all player names.
        """
        return self._player_names

    def get_state(self) -> entity_component.ComponentState:
        """Get component state for serialization."""
        return {"player_names": self._player_names}

    def set_state(self, state: entity_component.ComponentState) -> None:
        """Restore component state."""
        player_names = state.get("player_names", [])
        self._player_names = list(player_names) if player_names else []


class _SocialMediaEventResolution(entity_component.ContextComponent):
    """Resolve events using SocialMediaApp instead of LLM.

    Inherits from ContextComponent to satisfy Concordia's entity interface.
    """

    def __init__(
        self,
        app: SocialMediaApp,
        player_names: Sequence[str],
    ) -> None:
        """Initialize event resolution.

        Args:
            app: SocialMediaApp instance.
            player_names: Names of all players.
        """
        self._app = app
        self._player_names = list(player_names)
        self._last_results: dict[str, str] = {}

    def __call__(
        self,
        action_attempt: str,
        acting_entity: entity_lib.Entity,
    ) -> str:
        """Resolve action attempt.

        Args:
            action_attempt: Raw action string.
            acting_entity: Entity performing action.

        Returns:
            Result message.
        """
        parsed = parse_action(action_attempt)
        result = execute_action(self._app, acting_entity.name, parsed)
        self._last_results[acting_entity.name] = result.message

        # Advance step after each resolution
        self._app.current_step += 1

        return result.message

    def get_state(self) -> entity_component.ComponentState:
        """Get component state for serialization."""
        return {
            "last_results": self._last_results,
            "app_state": self._app.to_dict(),
        }

    def set_state(self, state: entity_component.ComponentState) -> None:
        """Restore component state."""
        last_results = state.get("last_results", {})
        self._last_results = dict(last_results) if last_results else {}
        app_state = state.get("app_state")
        if app_state:
            self._app = SocialMediaApp.from_dict(dict(app_state))
