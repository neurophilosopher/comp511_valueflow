"""Tests for MultiModelSimulator class."""

from __future__ import annotations

import pytest
from omegaconf import DictConfig

from src.simulation.simulators.multi_model import MultiModelSimulator


class TestMultiModelSimulator:
    """Tests for MultiModelSimulator functionality."""

    def test_create_models_single_model(self, test_config: DictConfig):
        """Test creating models for single model config."""
        simulator = MultiModelSimulator(test_config)
        models = simulator.create_models()

        assert "mock" in models
        assert len(models) == 1

    def test_create_models_multi_model(self, multi_model_config: DictConfig):
        """Test creating models for multi-model config."""
        simulator = MultiModelSimulator(multi_model_config)
        models = simulator.create_models()

        assert "mock1" in models
        assert "mock2" in models
        assert len(models) == 2

    def test_create_embedder(self, test_config: DictConfig):
        """Test creating embedder."""
        simulator = MultiModelSimulator(test_config)
        embedder = simulator.create_embedder()

        # Test that embedder works
        result = embedder("test text")
        assert result.shape == (768,)

    def test_mock_embedder_deterministic(self, test_config: DictConfig):
        """Test that mock embedder is deterministic."""
        simulator = MultiModelSimulator(test_config)

        # Force use of mock embedder
        embedding1 = simulator._mock_embedder("test")
        embedding2 = simulator._mock_embedder("test")

        assert (embedding1 == embedding2).all()

    def test_mock_embedder_different_texts(self, test_config: DictConfig):
        """Test that mock embedder gives different results for different texts."""
        simulator = MultiModelSimulator(test_config)

        embedding1 = simulator._mock_embedder("text one")
        embedding2 = simulator._mock_embedder("text two")

        # Should be different (with very high probability)
        assert not (embedding1 == embedding2).all()

    @pytest.mark.integration
    def test_full_setup(self, test_config: DictConfig):
        """Test full simulator setup."""
        simulator = MultiModelSimulator(test_config)
        simulator.setup()

        assert simulator.simulation is not None
        assert len(simulator.simulation.entities) >= 1

    def test_create_mock_model(self, test_config: DictConfig):
        """Test creating a mock model."""
        simulator = MultiModelSimulator(test_config)
        spec = {"provider": "mock", "model_name": "test"}
        model = simulator._create_mock_model(spec)

        response = model.sample_text("test prompt")
        assert isinstance(response, str)

    def test_unsupported_provider_raises(self, test_config: DictConfig):
        """Test that unsupported provider raises ValueError."""
        simulator = MultiModelSimulator(test_config)
        spec = {"provider": "unknown_provider", "model_name": "test"}

        with pytest.raises(ValueError, match="Unsupported model provider"):
            simulator._create_model_from_spec(spec)
