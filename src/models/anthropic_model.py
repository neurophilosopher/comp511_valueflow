"""Anthropic language model for use with the simulation framework.

This module provides an AnthropicLanguageModel class that wraps the Anthropic API
for use with Concordia-based simulations.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Sequence
from typing import Any

from concordia.language_model import language_model

_MAX_MULTIPLE_CHOICE_ATTEMPTS = 5


class AnthropicLanguageModel(language_model.LanguageModel):
    """Anthropic language model wrapper for Concordia simulations."""

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        """Initialize the Anthropic language model.

        Args:
            model_name: The Anthropic model to use (e.g., 'claude-3-5-sonnet-20241022').
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments (ignored, for config compatibility).
        """
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self._api_key = api_key
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic support requires the anthropic package. "
                "Install with: pip install anthropic"
            ) from e

        self._client = anthropic.Anthropic(api_key=self._api_key)

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        top_p: float = language_model.DEFAULT_TOP_P,
        top_k: int = language_model.DEFAULT_TOP_K,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        """Generate text from the Anthropic model.

        Args:
            prompt: The prompt text.
            max_tokens: Maximum tokens to generate.
            terminators: Stop sequences.
            temperature: Sampling temperature. Uses instance default if 1.0.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            timeout: Timeout in seconds.
            seed: Random seed (not supported by Anthropic, ignored).

        Returns:
            Generated text string.
        """
        import anthropic

        # Use instance temperature if default is passed
        effective_temp = (
            self._temperature if temperature == language_model.DEFAULT_TEMPERATURE else temperature
        )
        effective_max_tokens = min(max_tokens, self._max_tokens)

        system_prompt = (
            "You always continue sentences provided by the user and "
            "you never repeat what the user already said."
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]

        # Build request parameters
        request_params: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": effective_temp,
            "system": system_prompt,
        }

        # Add top_p if not default
        if top_p != language_model.DEFAULT_TOP_P:
            request_params["top_p"] = top_p

        # Add top_k if specified
        if top_k != language_model.DEFAULT_TOP_K:
            request_params["top_k"] = top_k

        # Add stop sequences if provided
        if terminators:
            request_params["stop_sequences"] = list(terminators)

        # Retry loop for transient errors
        has_result = False
        response = None
        while not has_result:
            try:
                response = self._client.messages.create(**request_params)
                has_result = True
            except anthropic.APIError as e:
                print(f"Anthropic API returned an API Error: {e}")
            except anthropic.APIConnectionError as e:
                print(f"Failed to connect to Anthropic API: {e}")
            except anthropic.RateLimitError as e:
                print(f"Anthropic API request exceeded rate limit: {e}")
                import time

                time.sleep(1)

        if response is None:
            return ""

        # Extract text from response
        content = response.content
        if content and len(content) > 0:
            return str(content[0].text)
        return ""

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict[str, Any]]:
        """Choose from a set of responses.

        Args:
            prompt: The prompt text.
            responses: Available response options.
            seed: Random seed (not supported by Anthropic, ignored).

        Returns:
            Tuple of (index, chosen response, metadata).
        """
        choice_prompt = (
            prompt
            + "\nRespond EXACTLY with one of the following strings:\n"
            + "\n".join(responses)
            + "."
        )

        for attempt in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
            # Increase temperature with each attempt
            temperature = min(1.0, 0.0 + attempt * 0.2)

            sample = self.sample_text(
                choice_prompt,
                temperature=temperature,
            )

            # Try to find an exact match
            answer = sample.strip()
            try:
                idx = responses.index(answer)
                return idx, responses[idx], {"attempts": attempt + 1}
            except ValueError:
                # Try partial matching
                for i, resp in enumerate(responses):
                    if resp.lower() in answer.lower() or answer.lower() in resp.lower():
                        return i, responses[i], {"attempts": attempt + 1, "partial_match": True}
                continue

        # Fallback: return first option
        return 0, responses[0], {"attempts": _MAX_MULTIPLE_CHOICE_ATTEMPTS, "fallback": True}
