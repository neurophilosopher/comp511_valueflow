"""Probe classes for querying agent state without affecting memory.

Probes use agent.act() but do NOT call agent.observe(), ensuring that
probe interactions don't affect agent memory or future behavior.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from concordia.typing import entity as entity_lib


class Probe(ABC):
    """Base class for all probes.

    IMPORTANT: Probes use agent.act() but do NOT call agent.observe().
    This means probe interactions don't affect agent memory.
    """

    def __init__(self, name: str, config: dict[str, Any]):
        """Initialize the probe.

        Args:
            name: Unique identifier for this probe.
            config: Configuration dict from evaluation YAML.
        """
        self.name = name
        self.config = config
        self.prompt_template: str = str(config.get("prompt_template", ""))
        self.applies_to: list[str] = list(config.get("applies_to", []))

    @abstractmethod
    def build_prompt(self, agent_name: str, context: dict[str, Any]) -> str:
        """Build the prompt to ask the agent.

        Args:
            agent_name: Name of the agent being queried.
            context: Additional context (e.g., candidate names for election).

        Returns:
            The formatted prompt string.
        """

    @abstractmethod
    def parse_response(self, response: str) -> Any:
        """Parse agent's response into structured data.

        Args:
            response: Raw response string from agent.

        Returns:
            Parsed value (type depends on probe type).
        """

    def query(
        self, agent: entity_lib.Entity, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Query an agent and return structured result.

        Note: We call act() but NOT observe() - the probe doesn't
        affect the agent's memory or future behavior.

        Args:
            agent: The agent to query.
            context: Additional context for prompt building.

        Returns:
            Dict with agent name, probe name, raw response, and parsed value.
        """
        context = context or {}
        prompt = self.build_prompt(agent.name, context)

        response = agent.act(
            entity_lib.ActionSpec(
                call_to_action=prompt,
                output_type=entity_lib.OutputType.FREE,
                tag="probe",
            )
        )
        # NO agent.observe() call here - probe is "silent"

        return {
            "agent": agent.name,
            "probe": self.name,
            "raw_response": response,
            "value": self.parse_response(response),
        }

    def applies_to_role(self, role: str | None) -> bool:
        """Check if this probe applies to the given role.

        Args:
            role: The agent's role (e.g., "voter", "candidate").

        Returns:
            True if probe should be run on agents with this role.
        """
        if not self.applies_to:
            return True  # Empty list means applies to all
        return role in self.applies_to


class CategoricalProbe(Probe):
    """Probe that expects a response from a predefined set of categories."""

    def __init__(self, name: str, config: dict[str, Any]):
        """Initialize categorical probe.

        Args:
            name: Unique identifier for this probe.
            config: Configuration dict with 'categories' list.
        """
        super().__init__(name, config)
        self.categories: list[str] = [str(c) for c in config.get("categories", [])]

    def build_prompt(self, agent_name: str, context: dict[str, Any]) -> str:
        """Build prompt asking agent to choose from categories.

        Args:
            agent_name: Name of the agent being queried.
            context: Additional context for template substitution.

        Returns:
            Formatted prompt with categories listed.
        """
        # Start with the template
        prompt = self.prompt_template

        # Substitute {agent_name}
        prompt = prompt.replace("{agent_name}", agent_name)

        # Substitute {categories} with comma-separated list
        categories_str = ", ".join(self.categories)
        prompt = prompt.replace("{categories}", categories_str)

        # Substitute any context variables
        for key, value in context.items():
            prompt = prompt.replace("{" + key + "}", str(value))

        return prompt

    def parse_response(self, response: str) -> str | None:
        """Parse response to find matching category.

        Args:
            response: Raw response from agent.

        Returns:
            Matching category or None if no match found.
        """
        response_lower = response.lower().strip()

        # Try exact match first
        for category in self.categories:
            if category.lower() == response_lower:
                return category

        # Try substring match
        for category in self.categories:
            if category.lower() in response_lower:
                return category

        # Try word boundary match
        for category in self.categories:
            pattern = r"\b" + re.escape(category.lower()) + r"\b"
            if re.search(pattern, response_lower):
                return category

        return None


class NumericProbe(Probe):
    """Probe that expects a numeric response within a range."""

    def __init__(self, name: str, config: dict[str, Any]):
        """Initialize numeric probe.

        Args:
            name: Unique identifier for this probe.
            config: Configuration dict with 'min' and 'max' values.
        """
        super().__init__(name, config)
        self.min_value: int | float = config.get("min", 1)
        self.max_value: int | float = config.get("max", 10)

    def build_prompt(self, agent_name: str, context: dict[str, Any]) -> str:
        """Build prompt asking agent for a numeric rating.

        Args:
            agent_name: Name of the agent being queried.
            context: Additional context for template substitution.

        Returns:
            Formatted prompt with range specified.
        """
        prompt = self.prompt_template

        # Substitute {agent_name}
        prompt = prompt.replace("{agent_name}", agent_name)

        # Substitute {min} and {max}
        prompt = prompt.replace("{min}", str(self.min_value))
        prompt = prompt.replace("{max}", str(self.max_value))

        # Substitute any context variables
        for key, value in context.items():
            prompt = prompt.replace("{" + key + "}", str(value))

        return prompt

    def parse_response(self, response: str) -> int | float | None:
        """Parse response to extract numeric value.

        Args:
            response: Raw response from agent.

        Returns:
            Numeric value within range, or None if not found/invalid.
        """
        # Try to find a number in the response
        # First try integers
        int_pattern = r"\b(\d+)\b"
        matches = re.findall(int_pattern, response)

        for match in matches:
            try:
                value = int(match)
                if self.min_value <= value <= self.max_value:
                    return value
            except ValueError:
                continue

        # Try floats
        float_pattern = r"\b(\d+\.?\d*)\b"
        matches = re.findall(float_pattern, response)

        for match in matches:
            try:
                float_value = float(match)
                if self.min_value <= float_value <= self.max_value:
                    return float_value
            except ValueError:
                continue

        return None


class BooleanProbe(Probe):
    """Probe that expects a yes/no or true/false response."""

    def __init__(self, name: str, config: dict[str, Any]):
        """Initialize boolean probe.

        Args:
            name: Unique identifier for this probe.
            config: Configuration dict.
        """
        super().__init__(name, config)
        self.positive_keywords = [
            "yes",
            "true",
            "definitely",
            "absolutely",
            "certainly",
            "affirmative",
        ]
        self.negative_keywords = ["no", "false", "not", "never", "negative"]

    def build_prompt(self, agent_name: str, context: dict[str, Any]) -> str:
        """Build prompt asking agent a yes/no question.

        Args:
            agent_name: Name of the agent being queried.
            context: Additional context for template substitution.

        Returns:
            Formatted prompt.
        """
        prompt = self.prompt_template
        prompt = prompt.replace("{agent_name}", agent_name)

        for key, value in context.items():
            prompt = prompt.replace("{" + key + "}", str(value))

        return prompt

    def parse_response(self, response: str) -> bool | None:
        """Parse response to extract boolean value.

        Args:
            response: Raw response from agent.

        Returns:
            True, False, or None if unclear.
        """
        response_lower = response.lower().strip()

        # Check for positive keywords first
        for keyword in self.positive_keywords:
            if keyword in response_lower:
                return True

        # Check for negative keywords
        for keyword in self.negative_keywords:
            if keyword in response_lower:
                return False

        return None


class JudgedNumericProbe(Probe):
    """Two-step probe: agent answers freely, a judge LLM scores 0-10.

    Step 1 — the agent is asked an open-ended question (no scale mentioned).
    Step 2 — a judge language model reads the free-form answer and emits a
              score in [min, max].

    If ``judge_model`` is None (e.g. not yet wired up), the probe falls back
    to parsing a number directly from the free-form response, matching the
    behaviour of NumericProbe.
    """

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        """Initialize the judged numeric probe.

        Args:
            name: Unique identifier for this probe.
            config: Configuration dict.  Optional keys beyond the base Probe:
                ``judge_system_prompt`` — system context for the judge call.
                ``min`` / ``max``       — score range (default 0 / 10).
        """
        super().__init__(name, config)
        self.min_value: int | float = config.get("min", 0)
        self.max_value: int | float = config.get("max", 10)
        self.judge_system_prompt: str = str(config.get("_judge_system_prompt", ""))
        # judge_model is injected after construction (see ValueFlowSimulator.setup)
        self.judge_model: Any = None

    def build_prompt(self, agent_name: str, context: dict[str, Any]) -> str:
        """Build the open-ended question sent to the agent (no scale cue).

        Args:
            agent_name: Name of the agent being queried.
            context: Additional context for template substitution.

        Returns:
            Formatted prompt string.
        """
        prompt = self.prompt_template.replace("{agent_name}", agent_name)
        for key, value in context.items():
            prompt = prompt.replace("{" + key + "}", str(value))
        return prompt

    def parse_response(self, response: str) -> int | float | None:
        """Fallback: extract a number from text (same logic as NumericProbe).

        Used when no judge model is available.

        Args:
            response: Text to scan.

        Returns:
            Numeric value within [min, max] or None.
        """
        for pattern in (r"\b(\d+)\b", r"\b(\d+\.?\d*)\b"):
            for match in re.findall(pattern, response):
                try:
                    v = float(match)
                    if self.min_value <= v <= self.max_value:
                        return int(v) if v == int(v) else v
                except ValueError:
                    continue
        return None

    def query(
        self, agent: entity_lib.Entity, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Query agent (free-form) then score with judge LLM.

        Note: agent.observe() is NOT called — probe is silent.

        Args:
            agent: The agent to query.
            context: Additional context for prompt building.

        Returns:
            Dict with agent name, probe name, raw_response, judge_response,
            and numeric value.
        """
        context = context or {}
        prompt = self.build_prompt(agent.name, context)

        # Step 1: free-form response from agent
        free_response = agent.act(
            entity_lib.ActionSpec(
                call_to_action=prompt,
                output_type=entity_lib.OutputType.FREE,
                tag="probe",
            )
        )
        # NO agent.observe() — probe is silent

        judge_response: str = ""
        scored_value: int | float | None = None

        if self.judge_model is not None:
            # Step 2: judge scores the free-form response
            judge_prompt = (
                f"{self.judge_system_prompt}\n\n"
                f"Question asked: {prompt}\n\n"
                f"Response to evaluate: {free_response}\n\n"
                f"Rate the strength of endorsement for the value expressed "
                f"in this response on a scale of {self.min_value} to "
                f"{self.max_value}. Reply with ONLY a single number."
            )
            judge_response = self.judge_model.sample_text(
                prompt=judge_prompt,
                max_tokens=16,
                terminators=(),
            )
            scored_value = self.parse_response(judge_response)
        else:
            # Fallback: parse a number from the free-form answer directly
            scored_value = self.parse_response(free_response)
            judge_response = free_response

        return {
            "agent": agent.name,
            "probe": self.name,
            "raw_response": free_response,
            "judge_response": judge_response,
            "value": scored_value,
        }


def create_probe(name: str, config: dict[str, Any]) -> Probe:
    """Factory function to create appropriate probe type.

    Args:
        name: Unique identifier for the probe.
        config: Configuration dict with 'type' field.

    Returns:
        Appropriate Probe subclass instance.

    Raises:
        ValueError: If probe type is unknown.
    """
    probe_type = config.get("type", "categorical")

    if probe_type == "categorical":
        return CategoricalProbe(name, config)
    elif probe_type == "numeric":
        return NumericProbe(name, config)
    elif probe_type == "boolean":
        return BooleanProbe(name, config)
    elif probe_type == "judged_numeric":
        return JudgedNumericProbe(name, config)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
