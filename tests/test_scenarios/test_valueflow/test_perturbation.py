"""Tests for ValueFlow perturbation injection (Phase 3).

Covers:
- build_perturbation_persona() template substitution
- ValueFlowSimulator.build_instances() persona injection
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from concordia.typing import prefab as prefab_lib
from omegaconf import OmegaConf

from scenarios.valueflow.game_masters import build_perturbation_persona
from scenarios.valueflow.simulator import ValueFlowSimulator
from src.simulation.simulators.multi_model import MultiModelSimulator

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_instance(name: str, persona: str = "neutral persona") -> prefab_lib.InstanceConfig:
    return prefab_lib.InstanceConfig(
        prefab="valueflow_agent",
        role=prefab_lib.Role.ENTITY,
        params={"name": name, "persona": persona},
    )


def _make_config(
    enabled: bool = True,
    perturbed_agent_index: int = 0,
    target_value: str = "social_power",
    target_value_type: str = "power",
    strength: int = 9,
) -> object:
    """Build a minimal OmegaConf config for ValueFlowSimulator."""
    return OmegaConf.create(
        {
            "experiment": {"name": "test", "seed": 0, "output_dir": "/tmp/vf_test"},
            "simulation": {
                "name": "sequential",
                "execution": {
                    "max_steps": 3,
                    "verbose": False,
                    "checkpoint": {"enabled": False},
                },
                "logging": {"save_html": False, "save_raw": False, "level": "WARNING"},
            },
            "model": {"name": "mock", "provider": "mock", "model_name": "mock"},
            "scenario": {
                "name": "valueflow",
                "premise": "Test premise.",
                "agents": {
                    "entities": [
                        {
                            "name": "Agent_0",
                            "role": "participant",
                            "prefab": "valueflow_agent",
                            "params": {},
                        },
                        {
                            "name": "Agent_1",
                            "role": "participant",
                            "prefab": "valueflow_agent",
                            "params": {},
                        },
                        {
                            "name": "Agent_2",
                            "role": "participant",
                            "prefab": "valueflow_agent",
                            "params": {},
                        },
                    ]
                },
                "game_master": {"prefab": "valueflow_game_master", "name": "gm", "params": {}},
                "prefabs": {},
                "topology": {"type": "chain"},
                "interaction": {"num_rounds": 2},
                "perturbation": {
                    "enabled": enabled,
                    "perturbed_agent_index": perturbed_agent_index,
                    "target_value": target_value,
                    "target_value_type": target_value_type,
                    "strength": strength,
                    "persona_override": (
                        "You hold extremely strong views about {value_name}. "
                        "Strength: {strength}/10."
                    ),
                },
                "data": {},
            },
            "environment": {"engine": "valueflow"},
        }
    )


# ── build_perturbation_persona ────────────────────────────────────────────────


class TestBuildPerturbationPersona:
    def test_substitutes_value_name(self) -> None:
        config = {
            "target_value": "social_power",
            "target_value_type": "power",
            "strength": 9,
            "persona_override": "You value {value_name} above all.",
        }
        result = build_perturbation_persona(config)
        assert "social power" in result

    def test_substitutes_strength(self) -> None:
        config = {
            "target_value": "freedom",
            "target_value_type": "self_direction",
            "strength": 8,
            "persona_override": "Intensity: {strength}/10.",
        }
        result = build_perturbation_persona(config)
        assert "8" in result

    def test_substitutes_value_type(self) -> None:
        config = {
            "target_value": "equality",
            "target_value_type": "universalism",
            "strength": 7,
            "persona_override": "Category: {value_type}.",
        }
        result = build_perturbation_persona(config)
        assert "universalism" in result

    def test_underscore_to_space_in_value_name(self) -> None:
        config = {
            "target_value": "family_security",
            "target_value_type": "security",
            "strength": 9,
            "persona_override": "{value_name}",
        }
        result = build_perturbation_persona(config)
        assert "family security" in result

    def test_empty_template_returns_empty(self) -> None:
        config = {
            "target_value": "helpful",
            "target_value_type": "benevolence",
            "strength": 5,
            "persona_override": "",
        }
        assert build_perturbation_persona(config) == ""

    def test_values_data_overrides_description(self) -> None:
        config = {
            "target_value": "social_power",
            "target_value_type": "power",
            "strength": 9,
            "persona_override": "{value_description}",
        }
        values_data = {
            "value_types": {"power": {"description": "control and dominance over others"}}
        }
        result = build_perturbation_persona(config, values_data)
        assert "control and dominance over others" in result


# ── ValueFlowSimulator.build_instances ───────────────────────────────────────


class TestValueFlowSimulatorPerturbation:
    """Tests for perturbation injection via ValueFlowSimulator.build_instances()."""

    def _base_instances(self) -> list[prefab_lib.InstanceConfig]:
        return [
            _make_instance("Agent_0", "original persona 0"),
            _make_instance("Agent_1", "original persona 1"),
            _make_instance("Agent_2", "original persona 2"),
        ]

    def test_perturbation_disabled_leaves_personas_unchanged(self) -> None:
        config = _make_config(enabled=False)
        sim = ValueFlowSimulator(config)
        instances = self._base_instances()

        with patch.object(MultiModelSimulator, "build_instances", return_value=instances):
            result = sim.build_instances()

        for inst in result:
            if inst.role == prefab_lib.Role.ENTITY:
                original = next(
                    i for i in instances if i.params.get("name") == inst.params.get("name")
                )
                assert inst.params["persona"] == original.params["persona"]

    def test_perturbation_modifies_target_agent_persona(self) -> None:
        config = _make_config(enabled=True, perturbed_agent_index=0)
        sim = ValueFlowSimulator(config)
        instances = self._base_instances()

        with patch.object(MultiModelSimulator, "build_instances", return_value=instances):
            result = sim.build_instances()

        agent0 = next(i for i in result if i.params.get("name") == "Agent_0")
        assert agent0.params["persona"] != "original persona 0"
        assert "social power" in agent0.params["persona"]

    def test_non_target_agents_personas_unchanged(self) -> None:
        config = _make_config(enabled=True, perturbed_agent_index=0)
        sim = ValueFlowSimulator(config)
        instances = self._base_instances()

        with patch.object(MultiModelSimulator, "build_instances", return_value=instances):
            result = sim.build_instances()

        for name, original_persona in [
            ("Agent_1", "original persona 1"),
            ("Agent_2", "original persona 2"),
        ]:
            inst = next(i for i in result if i.params.get("name") == name)
            assert inst.params["persona"] == original_persona

    def test_perturbation_index_1_modifies_agent_1(self) -> None:
        config = _make_config(enabled=True, perturbed_agent_index=1)
        sim = ValueFlowSimulator(config)
        instances = self._base_instances()

        with patch.object(MultiModelSimulator, "build_instances", return_value=instances):
            result = sim.build_instances()

        agent1 = next(i for i in result if i.params.get("name") == "Agent_1")
        agent0 = next(i for i in result if i.params.get("name") == "Agent_0")
        assert agent1.params["persona"] != "original persona 1"
        assert agent0.params["persona"] == "original persona 0"

    def test_out_of_range_index_raises(self) -> None:
        config = _make_config(enabled=True, perturbed_agent_index=99)
        sim = ValueFlowSimulator(config)

        with (
            patch.object(
                MultiModelSimulator, "build_instances", return_value=self._base_instances()
            ),
            pytest.raises(ValueError, match="out of range"),
        ):
            sim.build_instances()

    def test_instance_count_unchanged(self) -> None:
        config = _make_config(enabled=True, perturbed_agent_index=0)
        sim = ValueFlowSimulator(config)
        instances = self._base_instances()

        with patch.object(MultiModelSimulator, "build_instances", return_value=instances):
            result = sim.build_instances()

        assert len(result) == len(instances)

    def test_prefab_and_role_preserved_on_modified_instance(self) -> None:
        config = _make_config(enabled=True, perturbed_agent_index=0)
        sim = ValueFlowSimulator(config)
        instances = self._base_instances()

        with patch.object(MultiModelSimulator, "build_instances", return_value=instances):
            result = sim.build_instances()

        agent0 = next(i for i in result if i.params.get("name") == "Agent_0")
        assert agent0.prefab == "valueflow_agent"
        assert agent0.role == prefab_lib.Role.ENTITY
