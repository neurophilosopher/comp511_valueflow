"""Tests for BaseSimulator class."""

from __future__ import annotations

import pytest
from omegaconf import DictConfig

from src.simulation.simulators.base import BaseSimulator

# Import marketplace_config from scenario-specific conftest
pytest_plugins = ["scenarios.marketplace.conftest"]


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

        assert mapping.get("Agent1") == "mock1"
        assert mapping.get("Agent2") == "mock2"
        assert mapping.get("_default_") == "mock1"

    def test_build_instances(self, test_config: DictConfig):
        """Test building instances from config."""
        simulator = ConcreteSimulator(test_config)
        instances = simulator.build_instances()

        # Should have entities and game master
        assert len(instances) >= 2

        # Check entity names
        entity_names = [inst.params.get("name") for inst in instances]
        assert "Agent1" in entity_names
        assert "Agent2" in entity_names

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

    def test_build_instances_with_shared_memories(self, test_config: DictConfig):
        """Test that shared memories create an initializer instance."""
        # Add shared_memories to config
        test_config.scenario.shared_memories = [
            "This is a shared memory.",
            "All agents know this.",
        ]

        simulator = ConcreteSimulator(test_config)
        instances = simulator.build_instances()

        # Should have initializer at the start
        initializer = instances[0]
        assert initializer.prefab == "formative_memories_initializer__GameMaster"
        assert initializer.params["shared_memories"] == [
            "This is a shared memory.",
            "All agents know this.",
        ]

    def test_flatten_shared_memories(self, test_config: DictConfig):
        """Test that nested lists in shared_memories are flattened."""
        # Add nested shared_memories to config
        test_config.scenario.shared_memories = [
            "Simple string.",
            ["Nested item 1", "Nested item 2"],
            "Another string.",
        ]

        simulator = ConcreteSimulator(test_config)
        instances = simulator.build_instances()

        # Check flattened memories in initializer
        initializer = instances[0]
        expected = ["Simple string.", "Nested item 1", "Nested item 2", "Another string."]
        assert initializer.params["shared_memories"] == expected

    def test_build_agent_knowledge_no_builder(self, test_config: DictConfig):
        """Test build_agent_knowledge returns empty list when no builder configured."""
        simulator = ConcreteSimulator(test_config)

        # No builders configured in test_config
        knowledge = simulator.build_agent_knowledge("TestAgent", "participant", {})
        assert knowledge == []

    def test_build_agent_knowledge_with_builder(self, marketplace_config: DictConfig):
        """Test build_agent_knowledge calls the configured builder.

        Uses marketplace_config fixture which has the knowledge builder configured.
        """
        simulator = ConcreteSimulator(marketplace_config)

        params = {"budget": 500, "strategy": "value_seeker"}
        knowledge = simulator.build_agent_knowledge("TestBuyer", "buyer", params)

        # Should return knowledge from the builder
        assert isinstance(knowledge, list)
        assert len(knowledge) > 0

    def test_player_specific_context_from_goals(self, test_config: DictConfig):
        """Test that agent goals are added to player_specific_context."""
        test_config.scenario.shared_memories = ["Shared memory."]

        simulator = ConcreteSimulator(test_config)
        instances = simulator.build_instances()

        initializer = instances[0]
        context = initializer.params["player_specific_context"]

        # Check goals are captured for generic agents
        assert context.get("Agent1") == "Complete the test objective"
        assert context.get("Agent2") == "Assist with the test"
