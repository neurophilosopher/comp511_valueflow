"""Pytest fixtures for election scenario tests.

This module provides election-specific test configurations and fixtures.
"""

from __future__ import annotations

import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def election_config() -> DictConfig:
    """Provide a full election scenario configuration.

    Returns:
        DictConfig with election settings.
    """
    config_dict = {
        "experiment": {
            "name": "election_test",
            "seed": 42,
            "output_dir": "./test_outputs/election",
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
            "name": "election",
            "premise": "A local election is approaching in the community.",
            "roles": [
                {"name": "voter", "description": "Citizens who will vote"},
                {"name": "candidate", "description": "Politicians running for office"},
                {
                    "name": "news",
                    "description": "News outlet providing coverage",
                    "singular": True,
                },
            ],
            "builders": {
                "knowledge": {
                    "module": "scenarios.election.knowledge",
                    "function": "build_election_knowledge",
                },
                "events": {
                    "module": "scenarios.election.events",
                    "function": "generate_election_events",
                },
            },
            "agents": {
                "entities": [
                    {
                        "name": "Thomas Morrison",
                        "role": "candidate",
                        "prefab": "candidate_agent",
                        "params": {
                            "goal": "Win the election",
                            "partisan_type": "conservative",
                            "policy_proposals": ["Lower taxes"],
                            "campaign_style": "traditional",
                        },
                    },
                    {
                        "name": "Elena Rodriguez",
                        "role": "candidate",
                        "prefab": "candidate_agent",
                        "params": {
                            "goal": "Win the election",
                            "partisan_type": "progressive",
                            "policy_proposals": ["Invest in education"],
                            "campaign_style": "grassroots",
                        },
                    },
                    {
                        "name": "Maria Chen",
                        "role": "voter",
                        "prefab": "voter_agent",
                        "params": {
                            "goal": "Make an informed vote",
                            "persona_context": "Small business owner",
                            "communication_style": "pragmatic",
                            "initial_lean": "undecided",
                        },
                    },
                    {
                        "name": "James Wright",
                        "role": "voter",
                        "prefab": "voter_agent",
                        "params": {
                            "goal": "Support education",
                            "persona_context": "Retired teacher",
                            "communication_style": "thoughtful",
                            "initial_lean": "progressive",
                        },
                    },
                    {
                        "name": "Local News",
                        "role": "news",
                        "prefab": "news_agent",
                        "params": {
                            "goal": "Provide fair coverage",
                            "outlet_style": "local_journalism",
                            "headlines": ["Election race heats up"],
                        },
                    },
                ],
            },
            "game_master": {
                "prefab": "election_game_master",
                "name": "election_master",
                "params": {
                    "candidates": ["Thomas Morrison", "Elena Rodriguez"],
                    "election_rules": ["All citizens may vote"],
                },
            },
            "prefabs": {
                "voter_agent": {
                    "_target_": "scenarios.election.agents.VoterAgent",
                },
                "candidate_agent": {
                    "_target_": "scenarios.election.agents.CandidateAgent",
                },
                "news_agent": {
                    "_target_": "scenarios.election.agents.NewsAgent",
                },
                "election_game_master": {
                    "_target_": "scenarios.election.game_masters.ElectionGameMaster",
                },
            },
        },
        "evaluation": {
            "name": "basic_metrics",
        },
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def election_mock_responses() -> dict[str, str]:
    """Provide election-specific mock responses for testing.

    Returns:
        Dictionary mapping prompts to mock responses.
    """
    return {
        "situation": "I am considering the upcoming election and the candidates.",
        "options": "I can discuss issues with neighbors, attend events, or research candidates.",
        "best": "I should learn more about the candidates' policy positions.",
        "goal": "My goal is to make an informed decision in the election.",
        "vote_intention": "I am still weighing the options carefully.",
    }
