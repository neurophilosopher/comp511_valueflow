"""Pytest fixtures for marketplace scenario tests.

This module provides marketplace-specific test configurations and fixtures.
"""

from __future__ import annotations

import pytest
from omegaconf import DictConfig, OmegaConf


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
            "roles": [
                {"name": "buyer", "description": "Agents who purchase goods"},
                {"name": "seller", "description": "Agents who sell goods"},
                {
                    "name": "auctioneer",
                    "description": "Agent who facilitates trades",
                    "singular": True,
                },
            ],
            "builders": {
                "knowledge": {
                    "module": "scenarios.marketplace.knowledge",
                    "function": "build_market_knowledge",
                },
                "events": {
                    "module": "scenarios.marketplace.events",
                    "function": "generate_market_events",
                },
            },
            "agents": {
                "entities": [
                    {
                        "name": "Alice",
                        "role": "buyer",
                        "prefab": "buyer_agent",
                        "params": {
                            "goal": "Find good deals",
                            "budget": 500,
                            "strategy": "value_seeker",
                            "preferred_categories": ["electronics"],
                        },
                    },
                    {
                        "name": "Bob",
                        "role": "seller",
                        "prefab": "seller_agent",
                        "params": {
                            "goal": "Sell inventory",
                            "inventory": [
                                {
                                    "item": "Widget",
                                    "category": "electronics",
                                    "base_price": 100,
                                    "quantity": 5,
                                }
                            ],
                            "pricing_strategy": "competitive",
                        },
                    },
                    {
                        "name": "Max",
                        "role": "auctioneer",
                        "prefab": "auctioneer_agent",
                        "params": {
                            "goal": "Facilitate trades",
                            "auction_style": "english",
                            "commission_rate": 0.05,
                        },
                    },
                ],
            },
            "game_master": {
                "prefab": "market_game_master",
                "name": "market_master",
                "params": {
                    "market_rules": ["All trades must be fair"],
                },
            },
            "prefabs": {
                "buyer_agent": {
                    "_target_": "scenarios.marketplace.agents.BuyerAgent",
                },
                "seller_agent": {
                    "_target_": "scenarios.marketplace.agents.SellerAgent",
                },
                "auctioneer_agent": {
                    "_target_": "scenarios.marketplace.agents.AuctioneerAgent",
                },
                "market_game_master": {
                    "_target_": "scenarios.marketplace.game_masters.MarketGameMaster",
                },
            },
        },
        "evaluation": {
            "name": "basic_metrics",
        },
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def marketplace_mock_responses() -> dict[str, str]:
    """Provide marketplace-specific mock responses for testing.

    Returns:
        Dictionary mapping prompts to mock responses.
    """
    return {
        "situation": "I am in a marketplace looking at various goods.",
        "options": "I can buy items, negotiate prices, or observe other traders.",
        "best": "I should carefully evaluate the available items.",
        "goal": "My goal is to find good deals within my budget.",
    }
