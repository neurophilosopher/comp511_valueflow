"""Pytest configuration and fixtures for the test suite.

This module provides common fixtures used across all tests, including:
- Mock language models
- Mock embedders
- Mock memory banks
- Test configurations
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
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
    """Provide a mock language model with custom responses.

    Returns:
        MockLanguageModel with response mapping.
    """
    response_map = {
        "situation": "I am in a marketplace looking at various goods.",
        "options": "I can buy items, negotiate prices, or observe other traders.",
        "best": "I should carefully evaluate the available items.",
        "goal": "My goal is to find good deals within my budget.",
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
    """Provide a minimal test configuration.

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
                "buyers": [
                    {
                        "name": "TestBuyer",
                        "prefab": "buyer_agent",
                        "params": {"goal": "Test goal", "budget": 100},
                    }
                ],
                "sellers": [
                    {
                        "name": "TestSeller",
                        "prefab": "seller_agent",
                        "params": {"goal": "Sell items", "inventory": []},
                    }
                ],
            },
            "game_master": {
                "prefab": "market_game_master",
                "name": "test_gm",
                "params": {},
            },
            "prefabs": {
                "buyer_agent": "scenarios.marketplace.agents.BuyerAgent",
                "seller_agent": "scenarios.marketplace.agents.SellerAgent",
                "market_game_master": "scenarios.marketplace.game_masters.MarketGameMaster",
            },
        },
        "evaluation": {
            "name": "basic_metrics",
        },
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def marketplace_config() -> DictConfig:
    """Provide a full marketplace scenario configuration.

    Returns:
        DictConfig with marketplace settings.
    """
    config_dict = {
        "experiment": {
            "name": "marketplace_test",
            "seed": 42,
            "output_dir": "./test_outputs/marketplace",
        },
        "simulation": {
            "name": "sequential",
            "engine": {"type": "sequential"},
            "execution": {
                "max_steps": 10,
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
            "name": "marketplace",
            "premise": "A bustling marketplace where traders gather.",
            "agents": {
                "buyers": [
                    {
                        "name": "Alice",
                        "prefab": "buyer_agent",
                        "params": {
                            "goal": "Find good deals",
                            "budget": 500,
                            "strategy": "value_seeker",
                            "preferred_categories": ["electronics"],
                        },
                    },
                ],
                "sellers": [
                    {
                        "name": "Bob",
                        "prefab": "seller_agent",
                        "params": {
                            "goal": "Sell inventory",
                            "inventory": [
                                {"item": "Widget", "category": "electronics", "base_price": 100, "quantity": 5}
                            ],
                            "pricing_strategy": "competitive",
                        },
                    },
                ],
                "auctioneer": {
                    "name": "Max",
                    "prefab": "auctioneer_agent",
                    "params": {
                        "goal": "Facilitate trades",
                        "auction_style": "english",
                        "commission_rate": 0.05,
                    },
                },
            },
            "game_master": {
                "prefab": "market_game_master",
                "name": "market_master",
                "params": {
                    "market_rules": ["All trades must be fair"],
                },
            },
            "prefabs": {
                "buyer_agent": "scenarios.marketplace.agents.BuyerAgent",
                "seller_agent": "scenarios.marketplace.agents.SellerAgent",
                "auctioneer_agent": "scenarios.marketplace.agents.AuctioneerAgent",
                "market_game_master": "scenarios.marketplace.game_masters.MarketGameMaster",
            },
        },
        "evaluation": {
            "name": "basic_metrics",
        },
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def multi_model_config() -> DictConfig:
    """Provide a multi-model configuration.

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
                "Alice": "mock1",
                "Bob": "mock2",
                "_default_": "mock1",
            },
            "default_model": "mock1",
        },
        "scenario": {
            "name": "test_scenario",
            "premise": "Multi-model test scenario.",
            "agents": {
                "buyers": [
                    {
                        "name": "Alice",
                        "prefab": "basic_entity",
                        "params": {"goal": "Test"},
                    }
                ],
                "sellers": [
                    {
                        "name": "Bob",
                        "prefab": "basic_entity",
                        "params": {"goal": "Test"},
                    }
                ],
            },
            "game_master": {
                "prefab": "basic_game_master",
                "name": "narrator",
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
