"""Testing utilities.

This module provides mock classes and helper functions for testing
the simulation framework without external API calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from concordia.language_model import language_model


class MockLanguageModel(language_model.LanguageModel):
    """A mock language model for testing without API calls.

    This model returns deterministic responses based on the prompt,
    making tests reproducible.
    """

    def __init__(
        self,
        default_response: str = "I observe my surroundings and consider my options.",
        response_map: dict[str, str] | None = None,
    ) -> None:
        """Initialize the mock model.

        Args:
            default_response: Default response when no match is found.
            response_map: Optional mapping of prompt substrings to responses.
        """
        self._default_response = default_response
        self._response_map = response_map or {}
        self._call_count = 0
        self._call_history: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        """Get the number of times the model was called."""
        return self._call_count

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """Get the history of all calls to the model."""
        return self._call_history

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 5000,
        terminators: tuple = (),
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        timeout: float = 60,
        seed: int | None = None,
    ) -> str:
        """Return a mock response based on the prompt.

        Args:
            prompt: The prompt text.
            max_tokens: Maximum tokens (ignored in mock).
            terminators: Terminator strings (ignored in mock).
            temperature: Temperature (ignored in mock).
            top_p: Top-p sampling (ignored in mock).
            top_k: Top-k sampling (ignored in mock).
            timeout: Timeout (ignored in mock).
            seed: Random seed (ignored in mock).

        Returns:
            Mock response string.
        """
        self._call_count += 1
        self._call_history.append(
            {
                "type": "sample_text",
                "prompt": prompt,
                "max_tokens": max_tokens,
            }
        )

        # Check for matching prompts in user-provided map
        for key, response in self._response_map.items():
            if key.lower() in prompt.lower():
                return response

        # Handle Concordia v2 specific prompts
        prompt_lower = prompt.lower()

        # Termination check - say No to continue simulation
        if "terminate" in prompt_lower or "should the simulation end" in prompt_lower:
            # Terminate after a few steps to avoid infinite loop
            if self._call_count > 20:
                return "Yes"
            return "No"

        # Event resolution - return JSON format expected by Concordia v2
        if "suggested event" in prompt_lower or "resolve" in prompt_lower:
            return '{"events": [{"description": "The action was completed successfully.", "public": true}]}'

        # Observation generation
        if "what does" in prompt_lower and "observe" in prompt_lower:
            return "Nothing unusual happens. The environment remains calm."

        # Action selection
        if "what would" in prompt_lower and "do" in prompt_lower:
            return "waits and observes the surroundings."

        # Situation perception
        if "situation" in prompt_lower:
            return "The situation is calm and under control."

        # Best action
        if "best action" in prompt_lower or "best option" in prompt_lower:
            return "The best action is to continue observing."

        # Available options
        if "options" in prompt_lower or "available" in prompt_lower:
            return "The available options are: wait, observe, or take action."

        return self._default_response

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, Mapping[str, Any]]:
        """Return appropriate choice from the options.

        Args:
            prompt: The prompt text.
            responses: Available response options.
            seed: Random seed (ignored in mock).

        Returns:
            Tuple of (index, chosen response, empty metadata).
        """
        self._call_count += 1
        self._call_history.append(
            {
                "type": "sample_choice",
                "prompt": prompt,
                "responses": list(responses),
            }
        )

        prompt_lower = prompt.lower()

        # For termination/finished checks (detected by prompt content), say "No" until enough steps
        # Concordia uses letters (a, b) for choices, so we find which maps to "No"
        termination_keywords = [
            "terminate",
            "finished",
            "should the simulation end",
            "game over",
            "end the",
        ]
        is_termination_check = any(kw in prompt_lower for kw in termination_keywords)

        if is_termination_check:
            # Find which letter corresponds to "No" by looking at the prompt
            no_idx = None
            yes_idx = None
            for i, letter in enumerate(responses):
                # Check if this letter maps to "No"
                if f"({letter}) no" in prompt_lower:
                    no_idx = i
                elif f"({letter}) yes" in prompt_lower:
                    yes_idx = i

            # Terminate after enough calls to allow simulation to run
            if self._call_count > 10 and yes_idx is not None:
                return yes_idx, responses[yes_idx], {}
            elif no_idx is not None:
                return no_idx, responses[no_idx], {}

        # Default: return first option
        return 0, responses[0], {}

    def reset_history(self) -> None:
        """Reset call count and history."""
        self._call_count = 0
        self._call_history.clear()


def mock_embedder(text: str) -> np.ndarray:
    """Mock embedder that returns deterministic vectors.

    Args:
        text: Text to embed.

    Returns:
        Deterministic embedding vector based on text hash.
    """
    np.random.seed(hash(text) % (2**32))
    return np.random.randn(768).astype(np.float32)


class MockMemoryBank:
    """Mock memory bank for testing memory operations."""

    def __init__(self) -> None:
        """Initialize the mock memory bank."""
        self._memories: list[dict[str, Any]] = []

    def add_memory(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a memory.

        Args:
            text: Memory text.
            metadata: Optional metadata.
        """
        self._memories.append(
            {
                "text": text,
                "metadata": metadata or {},
            }
        )

    def get_memories(self, query: str, num_results: int = 5) -> list[str]:
        """Get memories matching a query.

        Args:
            query: Search query.
            num_results: Maximum number of results.

        Returns:
            List of memory texts.
        """
        # Simple substring matching for testing
        results = [m["text"] for m in self._memories if query.lower() in m["text"].lower()]
        return results[:num_results]

    def get_all_memories(self) -> list[str]:
        """Get all stored memories.

        Returns:
            List of all memory texts.
        """
        return [m["text"] for m in self._memories]

    def clear(self) -> None:
        """Clear all memories."""
        self._memories.clear()


def create_test_config() -> dict[str, Any]:
    """Create a minimal test configuration.

    Returns:
        Configuration dictionary suitable for testing.
    """
    return {
        "experiment": {
            "name": "test_experiment",
            "seed": 42,
            "output_dir": "./test_outputs",
        },
        "simulation": {
            "name": "sequential",
            "engine": {"type": "sequential"},
            "execution": {
                "max_steps": 5,
                "verbose": False,
                "checkpoint": {"enabled": False},
            },
            "logging": {
                "save_html": False,
                "save_raw": False,
                "level": "WARNING",
            },
        },
        "model": {
            "name": "mock",
            "provider": "mock",
            "model_name": "mock-model",
            "parameters": {},
        },
        "scenario": {
            "name": "test_scenario",
            "premise": "This is a test scenario.",
            "agents": {
                "entities": [
                    {
                        "name": "TestAgent",
                        "role": "test",
                        "prefab": "basic_entity",
                        "params": {"goal": "Test goal"},
                    }
                ],
            },
            "game_master": {
                "prefab": "basic_game_master",
                "name": "test_narrator",
            },
            "prefabs": {
                "basic_entity": "src.entities.agents.basic_entity.BasicEntity",
                "basic_game_master": "src.entities.game_masters.basic_gm.BasicGameMaster",
            },
        },
        "evaluation": {
            "name": "basic_metrics",
        },
    }
