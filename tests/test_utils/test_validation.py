"""Tests for configuration validation utilities."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.utils.validation import (
    ConfigValidationError,
    validate_config,
    validate_entity_model_mapping,
)


def _minimal_valid_config() -> dict:
    """Return a minimal valid configuration dict."""
    return {
        "simulation": {
            "execution": {"max_steps": 10, "checkpoint": {"enabled": False}},
            "logging": {"level": "WARNING"},
        },
        "model": {
            "name": "mock",
            "provider": "mock",
            "model_name": "mock-model",
        },
        "scenario": {
            "name": "test",
            "premise": "A test.",
            "agents": {
                "entities": [
                    {"name": "Alice", "prefab": "basic_entity"},
                ],
            },
            "game_master": {"prefab": "basic_gm", "name": "narrator"},
            "prefabs": {
                "basic_entity": {"_target_": "some.module.Entity"},
                "basic_gm": {"_target_": "some.module.GM"},
            },
        },
    }


class TestValidateConfig:
    """Tests for top-level validate_config."""

    def test_valid_config_returns_no_errors(self) -> None:
        config = OmegaConf.create(_minimal_valid_config())
        warnings = validate_config(config)
        assert isinstance(warnings, list)

    def test_missing_simulation_section_raises(self) -> None:
        cfg = _minimal_valid_config()
        del cfg["simulation"]
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="simulation"):
            validate_config(config)

    def test_missing_model_section_raises(self) -> None:
        cfg = _minimal_valid_config()
        del cfg["model"]
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="model"):
            validate_config(config)

    def test_missing_scenario_section_raises(self) -> None:
        cfg = _minimal_valid_config()
        del cfg["scenario"]
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="scenario"):
            validate_config(config)


class TestValidateSimulationConfig:
    """Tests for simulation config validation."""

    def test_missing_execution_raises(self) -> None:
        cfg = _minimal_valid_config()
        del cfg["simulation"]["execution"]
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="execution"):
            validate_config(config)

    def test_negative_max_steps_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["simulation"]["execution"]["max_steps"] = -1
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="max_steps"):
            validate_config(config)

    def test_missing_max_steps_warns(self) -> None:
        cfg = _minimal_valid_config()
        del cfg["simulation"]["execution"]["max_steps"]
        config = OmegaConf.create(cfg)

        warnings = validate_config(config)
        assert any("max_steps" in w for w in warnings)

    def test_invalid_log_level_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["simulation"]["logging"]["level"] = "VERBOSE"
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="log level"):
            validate_config(config)


class TestValidateModelConfig:
    """Tests for model config validation."""

    def test_missing_provider_raises(self) -> None:
        cfg = _minimal_valid_config()
        del cfg["model"]["provider"]
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="provider"):
            validate_config(config)

    def test_missing_model_name_raises(self) -> None:
        cfg = _minimal_valid_config()
        del cfg["model"]["model_name"]
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="model_name"):
            validate_config(config)

    def test_target_pattern_valid(self) -> None:
        cfg = _minimal_valid_config()
        cfg["model"] = {
            "name": "test",
            "_target_": "some.module.Model",
        }
        config = OmegaConf.create(cfg)
        # Should not raise
        validate_config(config)

    def test_target_pattern_invalid_path_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["model"] = {
            "name": "test",
            "_target_": "nodots",
        }
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="_target_"):
            validate_config(config)

    def test_multi_model_registry(self) -> None:
        cfg = _minimal_valid_config()
        cfg["model"] = {
            "name": "multi",
            "model_registry": {
                "model1": {"provider": "mock", "model_name": "m1"},
                "model2": {"provider": "mock", "model_name": "m2"},
            },
            "default_model": "model1",
        }
        config = OmegaConf.create(cfg)
        # Should not raise
        validate_config(config)

    def test_multi_model_empty_registry_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["model"] = {
            "name": "multi",
            "model_registry": {},
        }
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="empty"):
            validate_config(config)

    def test_registry_missing_provider_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["model"] = {
            "name": "multi",
            "model_registry": {
                "m1": {"model_name": "test"},  # missing provider
            },
        }
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="provider"):
            validate_config(config)


class TestValidateScenarioConfig:
    """Tests for scenario config validation."""

    def test_missing_agents_raises(self) -> None:
        cfg = _minimal_valid_config()
        del cfg["scenario"]["agents"]
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="agents"):
            validate_config(config)

    def test_empty_entities_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["scenario"]["agents"]["entities"] = []
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="entity"):
            validate_config(config)

    def test_entity_missing_name_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["scenario"]["agents"]["entities"] = [{"prefab": "basic_entity"}]
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="name"):
            validate_config(config)

    def test_entity_missing_prefab_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["scenario"]["agents"]["entities"] = [{"name": "Alice"}]
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="prefab"):
            validate_config(config)

    def test_missing_prefabs_section_raises(self) -> None:
        cfg = _minimal_valid_config()
        del cfg["scenario"]["prefabs"]
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="prefabs"):
            validate_config(config)

    def test_invalid_prefab_target_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["scenario"]["prefabs"]["basic_entity"] = {"_target_": "nodots"}
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="_target_"):
            validate_config(config)

    def test_missing_gm_prefab_raises(self) -> None:
        cfg = _minimal_valid_config()
        del cfg["scenario"]["game_master"]["prefab"]
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="game_master.prefab"):
            validate_config(config)

    def test_missing_scenario_name_warns(self) -> None:
        cfg = _minimal_valid_config()
        del cfg["scenario"]["name"]
        config = OmegaConf.create(cfg)

        warnings = validate_config(config)
        assert any("name" in w for w in warnings)

    def test_undefined_role_warns(self) -> None:
        cfg = _minimal_valid_config()
        cfg["scenario"]["roles"] = [{"name": "buyer"}, {"name": "seller"}]
        cfg["scenario"]["agents"]["entities"][0]["role"] = "astronaut"
        config = OmegaConf.create(cfg)

        warnings = validate_config(config)
        assert any("astronaut" in w for w in warnings)


class TestValidateKnowledgeConfig:
    """Tests for knowledge/builder config validation."""

    def test_valid_builders(self) -> None:
        cfg = _minimal_valid_config()
        cfg["scenario"]["builders"] = {
            "knowledge": {
                "module": "some.module",
                "function": "build_knowledge",
            },
        }
        config = OmegaConf.create(cfg)
        validate_config(config)  # should not raise

    def test_builder_missing_module_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["scenario"]["builders"] = {
            "knowledge": {"function": "build_knowledge"},
        }
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="module"):
            validate_config(config)

    def test_builder_missing_function_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["scenario"]["builders"] = {
            "knowledge": {"module": "some.module"},
        }
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="function"):
            validate_config(config)

    def test_shared_memories_not_list_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["scenario"]["shared_memories"] = "not a list"
        config = OmegaConf.create(cfg)

        with pytest.raises(ConfigValidationError, match="shared_memories"):
            validate_config(config)


class TestValidateEntityModelMapping:
    """Tests for validate_entity_model_mapping."""

    def test_valid_mapping(self) -> None:
        mapping = {"Alice": "model1", "Bob": "model2", "_default_": "model1"}
        warnings = validate_entity_model_mapping(
            mapping,
            available_models=["model1", "model2"],
            entity_names=["Alice", "Bob"],
        )
        assert warnings == []

    def test_unknown_model_warns(self) -> None:
        mapping = {"Alice": "nonexistent", "_default_": "model1"}
        warnings = validate_entity_model_mapping(
            mapping,
            available_models=["model1"],
            entity_names=["Alice"],
        )
        assert any("nonexistent" in w for w in warnings)

    def test_entity_without_mapping_or_default_warns(self) -> None:
        mapping = {"Alice": "model1"}  # No _default_, Bob unmapped
        warnings = validate_entity_model_mapping(
            mapping,
            available_models=["model1"],
            entity_names=["Alice", "Bob"],
        )
        assert any("Bob" in w for w in warnings)

    def test_default_covers_unmapped_entities(self) -> None:
        mapping = {"_default_": "model1"}
        warnings = validate_entity_model_mapping(
            mapping,
            available_models=["model1"],
            entity_names=["Alice", "Bob"],
        )
        assert warnings == []
