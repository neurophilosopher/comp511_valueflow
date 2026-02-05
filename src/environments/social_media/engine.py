"""Social media simulation engine with parallel agent execution."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import termcolor
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from concordia.utils import concurrency

if TYPE_CHECKING:
    from src.environments.social_media.app import SocialMediaApp

_PRINT_COLOR = "cyan"


@dataclass
class ActionResult:
    """Result of parsing and executing an action."""

    success: bool
    action_type: str
    message: str
    post_id: int | None = None  # For post/reply/boost actions


def parse_action(raw_action: str) -> dict[str, str]:
    """Parse structured action format.

    Expected format: ACTION: <type> | TARGET: <id> | CONTENT: <text>

    Args:
        raw_action: Raw action string from agent.

    Returns:
        Dictionary with action_type, target, content keys.
    """
    result = {"action_type": "skip", "target": "none", "content": "none"}

    # Try to parse structured format
    action_match = re.search(r"ACTION:\s*(\w+)", raw_action, re.IGNORECASE)
    target_match = re.search(r"TARGET:\s*([^\|]+)", raw_action, re.IGNORECASE)
    content_match = re.search(r"CONTENT:\s*(.+?)(?:\||$)", raw_action, re.IGNORECASE | re.DOTALL)

    if action_match:
        result["action_type"] = action_match.group(1).strip().lower()
    if target_match:
        result["target"] = target_match.group(1).strip()
    if content_match:
        result["content"] = content_match.group(1).strip()

    return result


def execute_action(
    app: SocialMediaApp,
    user: str,
    parsed: dict[str, str],
) -> ActionResult:
    """Execute a parsed action on the social media app.

    Args:
        app: The social media app instance.
        user: User performing the action.
        parsed: Parsed action dictionary.

    Returns:
        ActionResult with success status and message.
    """
    action_type = parsed["action_type"]
    target = parsed["target"]
    content = parsed["content"]

    try:
        if action_type == "post":
            if content == "none" or not content:
                return ActionResult(False, action_type, "Post requires content")
            post_id = app.post(user, content)
            return ActionResult(True, action_type, f"Posted #{post_id}", post_id)

        elif action_type == "reply":
            if content == "none" or not content:
                return ActionResult(False, action_type, "Reply requires content")
            try:
                target_id = int(target)
            except (ValueError, TypeError):
                return ActionResult(False, action_type, f"Invalid post ID: {target}")
            post_id = app.post(user, content, reply_to=target_id)
            return ActionResult(True, action_type, f"Replied #{post_id} to #{target_id}", post_id)

        elif action_type == "like":
            try:
                target_id = int(target)
            except (ValueError, TypeError):
                return ActionResult(False, action_type, f"Invalid post ID: {target}")
            if app.like(user, target_id):
                return ActionResult(True, action_type, f"Liked #{target_id}")
            return ActionResult(True, action_type, f"Already liked #{target_id}")

        elif action_type == "boost":
            try:
                target_id = int(target)
            except (ValueError, TypeError):
                return ActionResult(False, action_type, f"Invalid post ID: {target}")
            post_id = app.boost(user, target_id)
            return ActionResult(True, action_type, f"Boosted #{target_id} as #{post_id}", post_id)

        elif action_type == "follow":
            if target == "none" or not target:
                return ActionResult(False, action_type, "Follow requires target user")
            if app.follow(user, target):
                return ActionResult(True, action_type, f"Now following {target}")
            return ActionResult(True, action_type, f"Already following {target}")

        elif action_type == "unfollow":
            if target == "none" or not target:
                return ActionResult(False, action_type, "Unfollow requires target user")
            if app.unfollow(user, target):
                return ActionResult(True, action_type, f"Unfollowed {target}")
            return ActionResult(True, action_type, f"Was not following {target}")

        elif action_type == "skip":
            return ActionResult(True, action_type, "Skipped this step")

        else:
            return ActionResult(False, action_type, f"Unknown action type: {action_type}")

    except ValueError as e:
        return ActionResult(False, action_type, str(e))


class SocialMediaEngine(engine_lib.Engine):
    """Engine for parallel social media simulation.

    All agents act simultaneously each step:
    1. Each agent receives their timeline as observation
    2. Each agent selects an action (parallel)
    3. All actions are executed on the app
    4. Checkpoint

    The app instance is obtained from the game master at runtime.
    """

    def __init__(self) -> None:
        """Initialize engine.

        The SocialMediaApp is obtained from the game master at the start
        of run_loop, not at initialization time.
        """
        self.app: SocialMediaApp | None = None
        self._action_prompt = (
            "Choose ONE action. Format: ACTION: <type> | TARGET: <id or username or none> | CONTENT: <text or none>\n"
            "Actions: post, reply, like, boost, follow, unfollow, skip\n"
            "Examples:\n"
            "  ACTION: post | TARGET: none | CONTENT: Hello everyone!\n"
            "  ACTION: reply | TARGET: 42 | CONTENT: I agree with this\n"
            "  ACTION: like | TARGET: 42 | CONTENT: none\n"
            "  ACTION: boost | TARGET: 42 | CONTENT: none\n"
            "  ACTION: follow | TARGET: Alice | CONTENT: none\n"
            "  ACTION: skip | TARGET: none | CONTENT: none"
        )

    def make_observation(
        self,
        game_master: entity_lib.Entity,
        entity: entity_lib.Entity,
    ) -> str:
        raise NotImplementedError(
            "SocialMediaEngine uses run_loop directly; make_observation is not called."
        )

    def next_acting(
        self,
        game_master: entity_lib.Entity,
        entities: Sequence[entity_lib.Entity],
    ) -> tuple[entity_lib.Entity, entity_lib.ActionSpec]:
        raise NotImplementedError(
            "SocialMediaEngine uses run_loop directly; next_acting is not called."
        )

    def resolve(
        self,
        game_master: entity_lib.Entity,
        event: str,
    ) -> None:
        raise NotImplementedError(
            "SocialMediaEngine uses run_loop directly; resolve is not called."
        )

    def terminate(self, game_master: entity_lib.Entity) -> bool:
        raise NotImplementedError(
            "SocialMediaEngine uses run_loop directly; terminate is not called."
        )

    def next_game_master(
        self,
        game_master: entity_lib.Entity,
        game_masters: Sequence[entity_lib.Entity],
    ) -> entity_lib.Entity:
        raise NotImplementedError(
            "SocialMediaEngine uses run_loop directly; next_game_master is not called."
        )

    def run_loop(
        self,
        game_masters: Sequence[entity_lib.Entity],
        entities: Sequence[entity_lib.Entity],
        premise: str = "",
        max_steps: int = 100,
        verbose: bool = False,
        log: list[Mapping[str, Any]] | None = None,
        checkpoint_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Run the social media simulation loop.

        Args:
            game_masters: List of game masters (first one used for GM name).
            entities: List of agent entities.
            premise: Initial premise (observed by all agents).
            max_steps: Maximum simulation steps.
            verbose: Print detailed output.
            log: List to append log entries to.
            checkpoint_callback: Called after each step with step number.
        """
        if not entities:
            raise ValueError("No entities provided.")

        if not game_masters:
            raise ValueError("No game masters provided.")

        # Extract app from game master (find the one with app attribute)
        gm = None
        for candidate in game_masters:
            if hasattr(candidate, "app"):
                self.app = candidate.app
                gm = candidate
                break
            elif hasattr(candidate, "_app"):
                self.app = candidate._app
                gm = candidate
                break

        if gm is None:
            gm_names = [g.name for g in game_masters]
            raise ValueError(f"No game master with 'app' attribute found. Game masters: {gm_names}")

        # Validate app was set
        if self.app is None:
            raise RuntimeError("Failed to extract app from game master")

        # Capture app in local variable for use in nested function
        app = self.app

        steps = 0
        gm_name = gm.name

        # Initial premise observation
        if premise:
            for entity in entities:
                entity.observe(premise)

        while steps < max_steps:
            app.current_step = steps

            if verbose:
                print(termcolor.colored(f"\n=== Step {steps} ===", _PRINT_COLOR))

            # Create action spec for this step
            action_spec = entity_lib.ActionSpec(
                call_to_action=self._action_prompt,
                output_type=entity_lib.OutputType.FREE,
            )

            def _entity_step(
                entity: entity_lib.Entity,
                spec: entity_lib.ActionSpec = action_spec,
            ) -> dict[str, Any]:
                """Process one entity: observe -> act -> resolve."""
                # Generate observation (timeline)
                observation = app.format_timeline(entity.name)

                if verbose:
                    print(termcolor.colored(f"\n{entity.name} observes:", _PRINT_COLOR))
                    # Truncate for display
                    obs_lines = observation.split("\n")[:10]
                    print("\n".join(obs_lines))
                    if len(observation.split("\n")) > 10:
                        print("...")

                entity.observe(observation)

                # Get action from agent
                raw_action = entity.act(spec)

                if verbose:
                    print(termcolor.colored(f"{entity.name} action: {raw_action}", _PRINT_COLOR))

                # Parse and execute action
                parsed = parse_action(raw_action)
                result = execute_action(app, entity.name, parsed)

                if verbose:
                    status = "OK" if result.success else "FAILED"
                    print(termcolor.colored(f"  -> [{status}] {result.message}", _PRINT_COLOR))

                return {
                    "entity": entity.name,
                    "raw_action": raw_action,
                    "parsed": parsed,
                    "result": {
                        "success": result.success,
                        "action_type": result.action_type,
                        "message": result.message,
                        "post_id": result.post_id,
                    },
                }

            # Run all entities in parallel
            tasks = {entity.name: functools.partial(_entity_step, entity) for entity in entities}
            step_results = concurrency.run_tasks(tasks)

            # Log this step
            if log is not None:
                log_entry = {
                    "Step": steps,
                    "Summary": f"Step {steps} {gm_name}",
                    gm_name: {"actions": step_results},
                }
                for entity_name, entity_result in step_results.items():
                    log_entry[f"Entity [{entity_name}]"] = entity_result
                log.append(log_entry)

            steps += 1

            if checkpoint_callback is not None:
                checkpoint_callback(steps)

    def get_app_state(self) -> dict[str, Any]:
        """Get serializable app state for checkpoints."""
        if self.app is None:
            raise RuntimeError("App not available - run_loop has not been called")
        return self.app.to_dict()
