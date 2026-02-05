"""Concordia components for social media game master.

These components integrate the SocialMediaApp with Concordia's component system,
enabling the social media environment to work with run_experiment.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from concordia.components import game_master as gm_components
from concordia.typing import entity as entity_lib

if TYPE_CHECKING:
    from src.environments.social_media.app import SocialMediaApp


class SocialMediaMakeObservation(gm_components.make_observation.MakeObservation):
    """Generate observations from the social media feed.

    Instead of using an LLM to generate observations, this component
    provides the player's feed directly from the SocialMediaApp.
    """

    def __init__(
        self,
        app: SocialMediaApp,
        player_names: Sequence[str],
        timeline_limit: int = 20,
    ) -> None:
        """Initialize the observation component.

        Args:
            app: The SocialMediaApp instance.
            player_names: Names of all players.
            timeline_limit: Maximum posts to show in feed.
        """
        self._app = app
        self._player_names = list(player_names)
        self._timeline_limit = timeline_limit

    def make_observation(self, player_name: str) -> str:
        """Generate observation for a player from their feed.

        Args:
            player_name: Name of the player to generate observation for.

        Returns:
            Formatted feed as observation string.
        """
        return self._app.format_timeline(player_name, limit=self._timeline_limit)


class SocialMediaNextActing(gm_components.next_acting.NextActing):
    """Determine which players should act next.

    For parallel execution, all players act each step.
    """

    def __init__(self, player_names: Sequence[str]) -> None:
        """Initialize next acting component.

        Args:
            player_names: Names of all players.
        """
        self._player_names = list(player_names)

    def __call__(self) -> Sequence[str]:
        """Return all players (parallel execution).

        Returns:
            List of all player names.
        """
        return self._player_names


class SocialMediaEventResolution(gm_components.event_resolution.EventResolution):
    """Resolve player actions using the SocialMediaApp.

    Instead of using an LLM to interpret and resolve actions,
    this component parses structured actions and executes them
    deterministically on the SocialMediaApp.
    """

    def __init__(
        self,
        app: SocialMediaApp,
        player_names: Sequence[str],
        notify_observers: bool = True,
    ) -> None:
        """Initialize event resolution component.

        Args:
            app: The SocialMediaApp instance.
            player_names: Names of all players.
            notify_observers: Whether to notify other players of actions.
        """
        # Import here to avoid circular imports
        from src.environments.social_media.engine import execute_action, parse_action

        self._app = app
        self._player_names = list(player_names)
        self._notify_observers = notify_observers
        self._parse_action = parse_action
        self._execute_action = execute_action
        self._last_results: dict[str, str] = {}

    def resolve(
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
        parsed = self._parse_action(action_attempt)
        result = self._execute_action(self._app, acting_entity.name, parsed)
        self._last_results[acting_entity.name] = result.message
        return result.message

    def get_resolution_for_observers(self, player_name: str) -> str:
        """Get action result for observers.

        Args:
            player_name: Name of the player who acted.

        Returns:
            Action result message.
        """
        return self._last_results.get(player_name, "")


class SocialMediaActComponent:
    """Act component for social media game master.

    Orchestrates the simulation step by:
    1. Determining which entities act (all of them, parallel)
    2. Generating observations from feeds
    3. Resolving actions on the SocialMediaApp
    """

    def __init__(
        self,
        app: SocialMediaApp,
        player_names: Sequence[str],
        timeline_limit: int = 20,
    ) -> None:
        """Initialize the act component.

        Args:
            app: The SocialMediaApp instance.
            player_names: Names of all players.
            timeline_limit: Maximum posts to show in feed.
        """
        self._app = app
        self._player_names = list(player_names)
        self._timeline_limit = timeline_limit

        # Import here to avoid circular imports
        from src.environments.social_media.engine import execute_action, parse_action

        self._parse_action = parse_action
        self._execute_action = execute_action

        self._current_step = 0
        self._last_log: dict[str, Any] = {}

    def get_action_spec(self) -> entity_lib.ActionSpec:
        """Get action specification for entities.

        Returns:
            ActionSpec with current step context.
        """
        return entity_lib.ActionSpec(
            call_to_action=(
                f"Step {self._current_step}: What action do you take on social media?\n"
                "Respond with: ACTION: <type> | TARGET: <id/username/none> | CONTENT: <text/none>"
            ),
            output_type=entity_lib.OutputType.FREE,
        )

    def __call__(
        self,
        entities: Sequence[entity_lib.Entity],
    ) -> tuple[Sequence[str], dict[str, Any]]:
        """Execute a simulation step.

        Args:
            entities: All entities in the simulation.

        Returns:
            Tuple of (acting entity names, step log data).
        """
        # Advance the app step
        self._app.current_step = self._current_step

        # Generate observations for all players
        observations: dict[str, str] = {}
        for name in self._player_names:
            observations[name] = self._app.format_timeline(name, limit=self._timeline_limit)

        self._last_log = {
            "step": self._current_step,
            "observations": observations,
            "actions": {},
            "results": {},
        }

        self._current_step += 1

        # Return all players (parallel execution)
        return self._player_names, self._last_log

    def resolve_action(self, entity_name: str, action: str) -> str:
        """Resolve an entity's action.

        Args:
            entity_name: Name of the acting entity.
            action: Raw action string.

        Returns:
            Result message.
        """
        parsed = self._parse_action(action)
        result = self._execute_action(self._app, entity_name, parsed)

        self._last_log["actions"][entity_name] = parsed
        self._last_log["results"][entity_name] = result.message

        return result.message

    def get_last_log(self) -> dict[str, Any]:
        """Get the last step's log data."""
        return self._last_log
