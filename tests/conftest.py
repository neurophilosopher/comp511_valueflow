"""Pytest configuration and fixtures for the test suite.

This module provides common fixtures used across all tests, including:
- Mock language models
- Mock embedders
- Mock memory banks
- Generic test configurations (scenario-agnostic)

Scenario-specific fixtures should be placed in:
- scenarios/<scenario_name>/conftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.testing import MockLanguageModel, mock_embedder


@pytest.fixture
def mock_model() -> MockLanguageModel:
    """Provide a mock language model for testing.

    Returns:
        MockLanguageModel instance.
    """
    return MockLanguageModel()


@pytest.fixture
def mock_model_with_responses() -> MockLanguageModel:
    """Provide a mock language model with generic responses.

    Returns:
        MockLanguageModel with response mapping.
    """
    response_map = {
        "situation": "I observe my current surroundings.",
        "options": "I can take various actions or observe others.",
        "best": "I should consider my options carefully.",
        "goal": "I am working towards my objective.",
    }
    return MockLanguageModel(
        default_response="I observe my surroundings.",
        response_map=response_map,
    )


@pytest.fixture
def embedder():
    """Provide a mock embedder function.

    Returns:
        Mock embedder function.
    """
    return mock_embedder


@pytest.fixture
def mock_memory_bank(embedder):
    """Provide a mock memory bank for testing.

    Args:
        embedder: Mock embedder fixture.

    Returns:
        AssociativeMemoryBank instance.
    """
    from concordia.associative_memory import basic_associative_memory

    return basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=embedder,
    )


@pytest.fixture
def test_config() -> DictConfig:
    """Provide a minimal, scenario-agnostic test configuration.

    Uses generic BasicEntity and BasicGameMaster prefabs from src/entities/.
    For scenario-specific configs, see scenarios/<name>/conftest.py.

    Returns:
        DictConfig with test settings.
    """
    config_dict = {
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
                        "name": "Agent1",
                        "role": "participant",
                        "prefab": "basic_entity",
                        "params": {"goal": "Complete the test objective"},
                    },
                    {
                        "name": "Agent2",
                        "role": "participant",
                        "prefab": "basic_entity",
                        "params": {"goal": "Assist with the test"},
                    },
                ],
            },
            "game_master": {
                "prefab": "basic_game_master",
                "name": "test_narrator",
                "params": {},
            },
            "prefabs": {
                "basic_entity": {
                    "_target_": "src.entities.agents.basic_entity.BasicEntity",
                },
                "basic_game_master": {
                    "_target_": "src.entities.game_masters.basic_gm.BasicGameMaster",
                },
            },
        },
        "environment": {
            "name": "generic_world",
        },
        "evaluation": {
            "name": "basic_metrics",
        },
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def multi_model_config() -> DictConfig:
    """Provide a multi-model configuration (scenario-agnostic).

    Returns:
        DictConfig with multi-model settings.
    """
    config_dict = {
        "experiment": {
            "name": "multi_model_test",
            "seed": 42,
            "output_dir": "./test_outputs/multi_model",
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
            "name": "multi_model",
            "model_registry": {
                "mock1": {
                    "provider": "mock",
                    "model_name": "mock-model-1",
                    "parameters": {},
                },
                "mock2": {
                    "provider": "mock",
                    "model_name": "mock-model-2",
                    "parameters": {},
                },
            },
            "entity_model_mapping": {
                "Agent1": "mock1",
                "Agent2": "mock2",
                "_default_": "mock1",
            },
            "default_model": "mock1",
        },
        "scenario": {
            "name": "test_scenario",
            "premise": "Multi-model test scenario.",
            "agents": {
                "entities": [
                    {
                        "name": "Agent1",
                        "role": "participant",
                        "prefab": "basic_entity",
                        "params": {"goal": "Test with model 1"},
                    },
                    {
                        "name": "Agent2",
                        "role": "participant",
                        "prefab": "basic_entity",
                        "params": {"goal": "Test with model 2"},
                    },
                ],
            },
            "game_master": {
                "prefab": "basic_game_master",
                "name": "narrator",
            },
            "prefabs": {
                "basic_entity": {
                    "_target_": "src.entities.agents.basic_entity.BasicEntity",
                },
                "basic_game_master": {
                    "_target_": "src.entities.game_masters.basic_gm.BasicGameMaster",
                },
            },
        },
        "evaluation": {
            "name": "basic_metrics",
        },
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Provide a temporary output directory.

    Args:
        tmp_path: pytest's tmp_path fixture.

    Returns:
        Path to temporary directory.
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
