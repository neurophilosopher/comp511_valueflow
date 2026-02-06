"""Integration tests for full simulation runs."""

from __future__ import annotations

import json
from pathlib import Path

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

    @pytest.mark.integration
    def test_add_entity_wrong_role_raises(self, mock_model, embedder):
        """Test that add_entity rejects non-ENTITY role."""
        prefabs = {
            "basic_entity": BasicEntity(),
            "basic_gm": BasicGameMaster(),
        }

        config = prefab_lib.Config(
            prefabs=prefabs,
            instances=[
                prefab_lib.InstanceConfig(
                    prefab="basic_gm",
                    role=prefab_lib.Role.GAME_MASTER,
                    params={"name": "narrator"},
                ),
            ],
            default_premise="Test",
            default_max_steps=2,
        )

        simulation = Simulation(
            config=config,
            models={"mock": mock_model},
            entity_to_model={"_default_": "mock"},
            embedder=embedder,
        )

        gm_instance = prefab_lib.InstanceConfig(
            prefab="basic_entity",
            role=prefab_lib.Role.GAME_MASTER,
            params={"name": "ShouldFail"},
        )

        with pytest.raises(ValueError, match="ENTITY"):
            simulation.add_entity(gm_instance)

    @pytest.mark.integration
    def test_add_game_master_wrong_role_raises(self, mock_model, embedder):
        """Test that add_game_master rejects ENTITY role."""
        prefabs = {
            "basic_entity": BasicEntity(),
            "basic_gm": BasicGameMaster(),
        }

        config = prefab_lib.Config(
            prefabs=prefabs,
            instances=[
                prefab_lib.InstanceConfig(
                    prefab="basic_entity",
                    role=prefab_lib.Role.ENTITY,
                    params={"name": "Alice"},
                ),
                prefab_lib.InstanceConfig(
                    prefab="basic_gm",
                    role=prefab_lib.Role.GAME_MASTER,
                    params={"name": "narrator"},
                ),
            ],
            default_premise="Test",
            default_max_steps=2,
        )

        simulation = Simulation(
            config=config,
            models={"mock": mock_model},
            entity_to_model={"_default_": "mock"},
            embedder=embedder,
        )

        entity_instance = prefab_lib.InstanceConfig(
            prefab="basic_gm",
            role=prefab_lib.Role.ENTITY,
            params={"name": "ShouldFail"},
        )

        with pytest.raises(ValueError, match="GAME_MASTER"):
            simulation.add_game_master(entity_instance)

    @pytest.mark.integration
    def test_duplicate_entity_skipped(self, mock_model, embedder):
        """Test that adding duplicate entity is skipped."""
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

        assert len(simulation.entities) == 1

        # Try adding duplicate
        dup_instance = prefab_lib.InstanceConfig(
            prefab="basic_entity",
            role=prefab_lib.Role.ENTITY,
            params={"name": "Alice"},
        )
        simulation.add_entity(dup_instance)

        # Still 1 entity
        assert len(simulation.entities) == 1

    @pytest.mark.integration
    def test_get_model_fallback_to_default(self, embedder):
        """Test _get_model_for_entity falls back through: specific -> default -> first."""
        model1 = MockLanguageModel(default_response="m1")
        model2 = MockLanguageModel(default_response="m2")

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
            entity_to_model={"Alice": "model1", "_default_": "model2"},
            embedder=embedder,
        )

        # Specific mapping
        assert simulation._get_model_for_entity("Alice") is model1
        # Falls back to _default_
        assert simulation._get_model_for_entity("Unknown") is model2

    @pytest.mark.integration
    def test_save_checkpoint_to_disk(self, mock_model, embedder, tmp_path: Path):
        """Test save_checkpoint writes JSON file."""
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

        checkpoint_dir = str(tmp_path / "checkpoints")
        simulation.save_checkpoint(step=1, checkpoint_path=checkpoint_dir)

        checkpoint_file = tmp_path / "checkpoints" / "step_1_checkpoint.json"
        assert checkpoint_file.exists()

        data = json.loads(checkpoint_file.read_text())
        assert "entities" in data
        assert "game_masters" in data
        assert "raw_log" in data

    @pytest.mark.integration
    def test_get_raw_log_and_accessors(self, mock_model, embedder):
        """Test get_raw_log, get_entities, get_game_masters."""
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

        assert simulation.get_raw_log() == []
        assert len(simulation.get_entities()) == 1
        assert len(simulation.get_game_masters()) == 1
        assert simulation.get_entity_prefab_config("Alice") is not None
        assert simulation.get_entity_prefab_config("NonExistent") is None
