"""Custom engine utilities and fixes for Concordia compatibility.

This module provides fixes for Concordia engine components that need
customization for better LLM compatibility.
"""

from __future__ import annotations

import json
import re

from absl import logging
from concordia.typing import entity as entity_lib

_TYPE_SKIP_THIS_STEP = "type: __SKIP_THIS_STEP__"


def _legacy_action_spec_parser(
    next_action_spec_string: str,
) -> entity_lib.ActionSpec:
    """Parse the next action spec string using the legacy format.

    This exists for backward compatibility with the old string format:
    "prompt: <call_to_action>;;type: <free|choice> [options: opt1, opt2, ...]"

    Args:
        next_action_spec_string: The string representation of the next action spec.

    Returns:
        The parsed action spec.

    Raises:
        RuntimeError: If the next action spec string is invalid.
    """
    if "type: free" in next_action_spec_string:
        splits = next_action_spec_string.split(";;")

        if splits and "prompt: " in splits[0]:
            call_to_action = splits[0].split("prompt: ", 1)[1]
        else:
            call_to_action = entity_lib.DEFAULT_CALL_TO_ACTION

        return entity_lib.ActionSpec(
            call_to_action=call_to_action,
            output_type=entity_lib.OutputType.FREE,
        )

    elif "type: choice" in next_action_spec_string:
        splits = next_action_spec_string.split(";;")

        if "prompt: " in splits[0]:
            call_to_action = splits[0].split("prompt: ", 1)[1]
        else:
            call_to_action = entity_lib.DEFAULT_CALL_TO_ACTION

        if "options: " not in next_action_spec_string:
            return entity_lib.ActionSpec(
                call_to_action=call_to_action,
                output_type=entity_lib.OutputType.FREE,
            )

        options_str = next_action_spec_string.split("options: ", 1)[1]
        parts = re.split(r"(?<!\\),", options_str)
        options = tuple(
            dict.fromkeys(part.replace(r"\,", ",").strip() for part in parts if part.strip())
        )
        return entity_lib.ActionSpec(
            call_to_action=call_to_action,
            output_type=entity_lib.OutputType.CHOICE,
            options=options,
        )
    elif _TYPE_SKIP_THIS_STEP in next_action_spec_string:
        return entity_lib.skip_this_step_action_spec()
    else:
        raise RuntimeError(f'Invalid next action spec string: "{next_action_spec_string}"')


def action_spec_parser(next_action_spec_string: str) -> entity_lib.ActionSpec:
    """Parse the next action spec string into an action spec.

    Supports both JSON format (preferred) and legacy string format (for backward
    compatibility).

    This is a fixed version that handles cases where LLMs return valid JSON
    that is not a dict (e.g., a string or array).

    Args:
        next_action_spec_string: The string representation of the next action spec.

    Returns:
        The parsed action spec.
    """
    try:
        spec_dict = json.loads(next_action_spec_string)
        # Ensure we got a dict, not a string or other JSON type
        if not isinstance(spec_dict, dict):
            raise ValueError(f"Expected dict from JSON, got {type(spec_dict).__name__}")
        return entity_lib.action_spec_from_dict(spec_dict)
    except (json.JSONDecodeError, ValueError) as e:
        logging.warning(
            "Using legacy action spec parser. JSON parsing failed: %s. " "Input was: %s...",
            str(e),
            next_action_spec_string[:100],
        )
        return _legacy_action_spec_parser(next_action_spec_string)


def patch_concordia_parser() -> None:
    """Monkey-patch Concordia's action_spec_parser with our fixed version.

    This should be called during simulator initialization before any
    simulations are run.
    """
    from concordia.environment import engine as concordia_engine

    concordia_engine.action_spec_parser = action_spec_parser
    logging.info("Patched Concordia action_spec_parser with fixed version")
