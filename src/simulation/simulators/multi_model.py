"""Multi-model simulator implementation.

This module provides the MultiModelSimulator that supports assigning
different language models to different entities in the simulation.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
from concordia.language_model import language_model
from omegaconf import DictConfig, OmegaConf

from src.simulation.simulators.base import BaseSimulator


class MultiModelSimulator(BaseSimulator):
    """Simulator supporting multiple language models.

    This simulator can assign different LLMs (GPT-4, Claude, Ollama, etc.)
    to different entities, enabling heterogeneous agent simulations.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the multi-model simulator.

        Args:
            config: Hydra configuration.
        """
        super().__init__(config)
        self._models_cache: dict[str, language_model.LanguageModel] = {}

    def create_models(self) -> dict[str, language_model.LanguageModel]:
        """Create language model instances from the model registry.

        Returns:
            Dictionary mapping model names to LanguageModel instances.
        """
        model_config = self._config.model

        # Handle single-model configuration
        if "model_registry" not in model_config:
            model = self._create_single_model(model_config)
            return {model_config.name: model}

        # Handle multi-model registry
        models: dict[str, language_model.LanguageModel] = {}
        registry = model_config.model_registry

        for model_name, model_spec in registry.items():
            model_spec_dict = OmegaConf.to_container(model_spec, resolve=True)
            models[model_name] = self._create_model_from_spec(model_spec_dict)

        return models

    def _create_single_model(
        self, model_config: DictConfig
    ) -> language_model.LanguageModel:
        """Create a single model from config.

        Args:
            model_config: Configuration for a single model.

        Returns:
            LanguageModel instance.
        """
        spec = OmegaConf.to_container(model_config, resolve=True)
        return self._create_model_from_spec(spec)

    def _create_model_from_spec(
        self, spec: dict[str, Any]
    ) -> language_model.LanguageModel:
        """Create a language model from a specification dictionary.

        Args:
            spec: Model specification with provider, model_name, parameters, etc.

        Returns:
            LanguageModel instance.

        Raises:
            ValueError: If the provider is not supported.
        """
        provider = spec.get("provider", "").lower()

        if provider == "openai":
            return self._create_openai_model(spec)
        elif provider == "anthropic":
            return self._create_anthropic_model(spec)
        elif provider == "ollama":
            return self._create_ollama_model(spec)
        elif provider == "mock":
            return self._create_mock_model(spec)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def _create_openai_model(
        self, spec: dict[str, Any]
    ) -> language_model.LanguageModel:
        """Create an OpenAI model.

        Args:
            spec: Model specification.

        Returns:
            OpenAI LanguageModel instance.
        """
        try:
            from concordia.language_model import gpt_model
        except ImportError as e:
            raise ImportError(
                "OpenAI support requires the concordia package with gpt_model. "
                "Install with: pip install concordia[openai]"
            ) from e

        api_config = spec.get("api", {})
        params = spec.get("parameters", {})

        api_key = api_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")

        return gpt_model.GptLanguageModel(
            model_name=spec.get("model_name", "gpt-4-turbo"),
            api_key=api_key,
            **params,
        )

    def _create_anthropic_model(
        self, spec: dict[str, Any]
    ) -> language_model.LanguageModel:
        """Create an Anthropic Claude model.

        Args:
            spec: Model specification.

        Returns:
            Anthropic LanguageModel instance.
        """
        try:
            from concordia.language_model import anthropic_model
        except ImportError as e:
            raise ImportError(
                "Anthropic support requires concordia with anthropic_model. "
                "Install with: pip install concordia anthropic"
            ) from e

        api_config = spec.get("api", {})
        params = spec.get("parameters", {})

        api_key = api_config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found in config or environment")

        return anthropic_model.AnthropicLanguageModel(
            model_name=spec.get("model_name", "claude-3-5-sonnet-20241022"),
            api_key=api_key,
            **params,
        )

    def _create_ollama_model(
        self, spec: dict[str, Any]
    ) -> language_model.LanguageModel:
        """Create an Ollama local model.

        Args:
            spec: Model specification.

        Returns:
            Ollama LanguageModel instance.
        """
        try:
            from concordia.language_model import ollama_model
        except ImportError as e:
            raise ImportError(
                "Ollama support requires concordia with ollama_model. "
                "Install with: pip install concordia ollama"
            ) from e

        api_config = spec.get("api", {})
        params = spec.get("parameters", {})

        host = api_config.get("host") or os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434"
        )

        return ollama_model.OllamaLanguageModel(
            model_name=spec.get("model_name", "llama3.1:8b"),
            host=host,
            **params,
        )

    def _create_mock_model(
        self, spec: dict[str, Any]
    ) -> language_model.LanguageModel:
        """Create a mock model for testing.

        Args:
            spec: Model specification.

        Returns:
            Mock LanguageModel instance.
        """
        from src.utils.testing import MockLanguageModel
        return MockLanguageModel()

    def create_embedder(self) -> Callable[[str], np.ndarray]:
        """Create the sentence embedder.

        Returns:
            Function that embeds text into vectors.
        """
        try:
            import sentence_transformers

            model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")

            def embedder(text: str) -> np.ndarray:
                return model.encode(text, show_progress_bar=False)

            return embedder

        except ImportError:
            # Fall back to mock embedder for testing
            print("Warning: sentence-transformers not available, using mock embedder")
            return self._mock_embedder

    def _mock_embedder(self, text: str) -> np.ndarray:
        """Mock embedder for testing without sentence-transformers.

        Args:
            text: Text to embed.

        Returns:
            Random but deterministic embedding vector.
        """
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(768).astype(np.float32)
