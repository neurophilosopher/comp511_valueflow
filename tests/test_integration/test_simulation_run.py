"""Integration tests for full simulation runs."""

from __future__ import annotations

import pytest
from concordia.typing import prefab as prefab_lib

from src.entities.agents.basic_entity import BasicEntity
from src.entities.game_masters.basic_gm import BasicGameMaster
from src.simulation.simulation import Simulation
from src.utils.testing import MockLanguageModel


class TestSimulationSetup:
    """Tests for simulation setup (without running the full event loop)."""

    @pytest.mark.integration
    def test_simulation_setup(self, mock_model, embedder):
        """Test that simulation can be set up correctly."""
        prefabs = {
            "basic_entity": BasicEntity(),
            "basic_gm": BasicGameMaster(),
        }

        instances = [
            prefab_lib.InstanceConfig(
                prefab="basic_entity",
                role=prefab_lib.Role.ENTITY,
                params={"name": "TestAgent", "goal": "Test"},
            ),
            prefab_lib.InstanceConfig(
                prefab="basic_gm",
                role=prefab_lib.Role.GAME_MASTER,
                params={"name": "narrator"},
            ),
        ]

        config = prefab_lib.Config(
            prefabs=prefabs,
            instances=instances,
            default_premise="Test simulation",
            default_max_steps=2,
        )

        simulation = Simulation(
            config=config,
            models={"mock": mock_model},
            entity_to_model={"_default_": "mock"},
            embedder=embedder,
        )

        assert len(simulation.entities) == 1
        assert len(simulation.game_masters) == 1
        assert simulation.entities[0].name == "TestAgent"
        assert simulation.game_masters[0].name == "narrator"

    @pytest.mark.integration
    def test_multi_entity_setup(self, mock_model, embedder):
        """Test simulation setup with multiple entities."""
        prefabs = {
            "basic_entity": BasicEntity(),
            "basic_gm": BasicGameMaster(),
        }

        instances = [
            prefab_lib.InstanceConfig(
                prefab="basic_entity",
                role=prefab_lib.Role.ENTITY,
                params={"name": "Alice", "goal": "Explore"},
            ),
            prefab_lib.InstanceConfig(
                prefab="basic_entity",
                role=prefab_lib.Role.ENTITY,
                params={"name": "Bob", "goal": "Discover"},
            ),
            prefab_lib.InstanceConfig(
                prefab="basic_gm",
                role=prefab_lib.Role.GAME_MASTER,
                params={"name": "narrator"},
            ),
        ]

        config = prefab_lib.Config(
            prefabs=prefabs,
            instances=instances,
            default_premise="Two adventurers meet.",
            default_max_steps=3,
        )

        simulation = Simulation(
            config=config,
            models={"mock": mock_model},
            entity_to_model={"_default_": "mock"},
            embedder=embedder,
        )

        assert len(simulation.entities) == 2
        entity_names = [e.name for e in simulation.entities]
        assert "Alice" in entity_names
        assert "Bob" in entity_names

    @pytest.mark.integration
    def test_multi_model_setup(self, embedder):
        """Test simulation setup with multiple models."""
        model1 = MockLanguageModel(default_response="Model 1 response")
        model2 = MockLanguageModel(default_response="Model 2 response")

        prefabs = {
            "basic_entity": BasicEntity(),
            "basic_gm": BasicGameMaster(),
        }

        instances = [
            prefab_lib.InstanceConfig(
                prefab="basic_entity",
                role=prefab_lib.Role.ENTITY,
                params={"name": "Alice"},
            ),
            prefab_lib.InstanceConfig(
                prefab="basic_entity",
                role=prefab_lib.Role.ENTITY,
                params={"name": "Bob"},
            ),
            prefab_lib.InstanceConfig(
                prefab="basic_gm",
                role=prefab_lib.Role.GAME_MASTER,
                params={"name": "narrator"},
            ),
        ]

        config = prefab_lib.Config(
            prefabs=prefabs,
            instances=instances,
            default_premise="Test",
            default_max_steps=2,
        )

        simulation = Simulation(
            config=config,
            models={"model1": model1, "model2": model2},
            entity_to_model={"Alice": "model1", "Bob": "model2", "_default_": "model1"},
            embedder=embedder,
        )

        assert len(simulation.entities) == 2
        assert len(simulation.game_masters) == 1

    @pytest.mark.integration
    def test_checkpoint_data_structure(self, mock_model, embedder):
        """Test checkpoint data structure without running simulation."""
        prefabs = {
            "basic_entity": BasicEntity(),
            "basic_gm": BasicGameMaster(),
        }

        instances = [
            prefab_lib.InstanceConfig(
                prefab="basic_entity",
                role=prefab_lib.Role.ENTITY,
                params={"name": "CheckpointAgent"},
            ),
            prefab_lib.InstanceConfig(
                prefab="basic_gm",
                role=prefab_lib.Role.GAME_MASTER,
                params={"name": "narrator"},
            ),
        ]

        config = prefab_lib.Config(
            prefabs=prefabs,
            instances=instances,
            default_premise="Test",
            default_max_steps=2,
        )

        simulation = Simulation(
            config=config,
            models={"mock": mock_model},
            entity_to_model={"_default_": "mock"},
            embedder=embedder,
        )

        # Create checkpoint without running simulation
        checkpoint = simulation.make_checkpoint_data()

        assert "entities" in checkpoint
        assert "game_masters" in checkpoint
        assert "raw_log" in checkpoint
        assert "CheckpointAgent" in checkpoint["entities"]

    @pytest.mark.integration
    def test_entity_can_observe(self, mock_model, embedder):
        """Test that entities can receive observations."""
        prefabs = {
            "basic_entity": BasicEntity(),
            "basic_gm": BasicGameMaster(),
        }

        instances = [
            prefab_lib.InstanceConfig(
                prefab="basic_entity",
                role=prefab_lib.Role.ENTITY,
                params={"name": "ObserverAgent"},
            ),
            prefab_lib.InstanceConfig(
                prefab="basic_gm",
                role=prefab_lib.Role.GAME_MASTER,
                params={"name": "narrator"},
            ),
        ]

        config = prefab_lib.Config(
            prefabs=prefabs,
            instances=instances,
            default_premise="Test",
            default_max_steps=2,
        )

        simulation = Simulation(
            config=config,
            models={"mock": mock_model},
            entity_to_model={"_default_": "mock"},
            embedder=embedder,
        )

        # Test that entity can receive observations
        entity = simulation.entities[0]
        entity.observe("Something happened in the environment.")
        # Should not raise

    @pytest.mark.integration
    def test_entity_can_act(self, mock_model, embedder):
        """Test that entities can generate actions."""
        prefabs = {
            "basic_entity": BasicEntity(),
            "basic_gm": BasicGameMaster(),
        }

        instances = [
            prefab_lib.InstanceConfig(
                prefab="basic_entity",
                role=prefab_lib.Role.ENTITY,
                params={"name": "ActingAgent"},
            ),
            prefab_lib.InstanceConfig(
                prefab="basic_gm",
                role=prefab_lib.Role.GAME_MASTER,
                params={"name": "narrator"},
            ),
        ]

        config = prefab_lib.Config(
            prefabs=prefabs,
            instances=instances,
            default_premise="Test",
            default_max_steps=2,
        )

        simulation = Simulation(
            config=config,
            models={"mock": mock_model},
            entity_to_model={"_default_": "mock"},
            embedder=embedder,
        )

        # Test that entity can act
        entity = simulation.entities[0]
        action = entity.act()
        assert isinstance(action, str)
