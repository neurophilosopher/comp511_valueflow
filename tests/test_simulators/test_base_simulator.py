"""Tests for BaseSimulator class."""

from __future__ import annotations

import pytest
from omegaconf import DictConfig

from src.simulation.simulators.base import BaseSimulator


class ConcreteSimulator(BaseSimulator):
    """Concrete implementation of BaseSimulator for testing."""

    def create_models(self):
        from src.utils.testing import MockLanguageModel

        return {"mock": MockLanguageModel()}

    def create_embedder(self):
        from src.utils.testing import mock_embedder

        return mock_embedder


class TestBaseSimulator:
    """Tests for BaseSimulator functionality."""

    def test_init_stores_config(self, test_config: DictConfig):
        """Test that __init__ stores the configuration."""
        simulator = ConcreteSimulator(test_config)
        assert simulator.config == test_config
        assert simulator.simulation is None

    def test_get_entity_model_mapping_single_model(self, test_config: DictConfig):
        """Test entity model mapping for single model configs."""
        simulator = ConcreteSimulator(test_config)
        mapping = simulator.get_entity_model_mapping()

        assert "_default_" in mapping
        assert mapping["_default_"] == "mock"

    def test_get_entity_model_mapping_multi_model(self, multi_model_config: DictConfig):
        """Test entity model mapping for multi-model configs."""
        simulator = ConcreteSimulator(multi_model_config)
        mapping = simulator.get_entity_model_mapping()

        assert mapping.get("Alice") == "mock1"
        assert mapping.get("Bob") == "mock2"
        assert mapping.get("_default_") == "mock1"

    def test_build_instances(self, test_config: DictConfig):
        """Test building instances from config."""
        simulator = ConcreteSimulator(test_config)
        instances = simulator.build_instances()

        # Should have buyer, seller, and game master
        assert len(instances) >= 2

        # Check entity names
        entity_names = [inst.params.get("name") for inst in instances]
        assert "TestBuyer" in entity_names
        assert "TestSeller" in entity_names

    def test_build_config(self, test_config: DictConfig):
        """Test building Concordia Config object."""
        simulator = ConcreteSimulator(test_config)
        config = simulator.build_config()

        assert config.default_premise == "This is a test scenario."
        assert config.default_max_steps == 5
        assert len(config.instances) >= 2

    def test_run_without_setup_raises(self, test_config: DictConfig):
        """Test that running without setup raises RuntimeError."""
        simulator = ConcreteSimulator(test_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            simulator.run()
