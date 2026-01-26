"""OpenAI language model for use with the simulation framework.

This module provides an OpenAILanguageModel class that wraps the OpenAI API
for use with Concordia-based simulations.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Sequence
from typing import Any

from concordia.language_model import language_model

_MAX_MULTIPLE_CHOICE_ATTEMPTS = 5


class OpenAILanguageModel(language_model.LanguageModel):
    """OpenAI language model wrapper for Concordia simulations."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI language model.

        Args:
            model_name: The OpenAI model to use (e.g., 'gpt-4o-mini', 'gpt-4o').
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments (ignored, for config compatibility).
        """
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self._api_key = api_key
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAI support requires the openai package. " "Install with: pip install openai"
            ) from e

        self._client = openai.OpenAI(api_key=self._api_key)

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
        """Generate text from the OpenAI model.

        Args:
            prompt: The prompt text.
            max_tokens: Maximum tokens to generate.
            terminators: Stop sequences.
            temperature: Sampling temperature. Uses instance default if 1.0.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter (ignored by OpenAI).
            timeout: Timeout in seconds.
            seed: Random seed for reproducibility.

        Returns:
            Generated text string.
        """
        import openai

        # Use instance temperature if default is passed
        effective_temp = (
            self._temperature if temperature == language_model.DEFAULT_TEMPERATURE else temperature
        )
        effective_max_tokens = min(max_tokens, self._max_tokens)

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You always continue sentences provided by the user and "
                    "you never repeat what the user already said."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Build request parameters
        request_params: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": effective_temp,
            "top_p": top_p,
            "timeout": timeout,
        }

        # Add stop sequences if provided (OpenAI max 4)
        if terminators:
            request_params["stop"] = list(terminators)[:4]

        if seed is not None:
            request_params["seed"] = seed

        # Retry loop for transient errors
        has_result = False
        response = None
        while not has_result:
            try:
                response = self._client.chat.completions.create(**request_params)
                has_result = True
            except openai.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
            except openai.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}")
            except openai.RateLimitError as e:
                print(f"OpenAI API request exceeded rate limit: {e}")
                import time

                time.sleep(1)

        if response is None:
            return ""

        content = response.choices[0].message.content
        return str(content) if content else ""

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
            seed: Random seed for reproducibility.

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
                seed=seed,
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
