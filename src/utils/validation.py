"""Configuration validation utilities.

This module provides functions for validating Hydra configurations
to ensure they have all required fields and valid values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import ListConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""


def validate_config(config: DictConfig) -> list[str]:
    """Validate a Hydra configuration.

    Args:
        config: The configuration to validate.

    Returns:
        List of warning messages (empty if no warnings).

    Raises:
        ConfigValidationError: If required fields are missing or invalid.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Validate top-level structure
    required_sections = ["simulation", "model", "scenario"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required config section: {section}")

    if errors:
        raise ConfigValidationError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # Validate simulation config
    _validate_simulation_config(config.simulation, errors, warnings)

    # Validate model config
    _validate_model_config(config.model, errors, warnings)

    # Validate scenario config
    _validate_scenario_config(config.scenario, errors, warnings)

    if errors:
        raise ConfigValidationError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return warnings


def _validate_simulation_config(
    sim_config: DictConfig,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate simulation configuration section.

    Args:
        sim_config: Simulation configuration.
        errors: List to append errors to.
        warnings: List to append warnings to.
    """
    # Check execution settings
    if "execution" not in sim_config:
        errors.append("simulation.execution section is required")
        return

    exec_config = sim_config.execution

    if "max_steps" not in exec_config:
        warnings.append("simulation.execution.max_steps not set, using default")
    elif exec_config.max_steps <= 0:
        errors.append("simulation.execution.max_steps must be positive")

    # Check logging settings
    if "logging" in sim_config:
        log_config = sim_config.logging
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if "level" in log_config and log_config.level.upper() not in valid_levels:
            errors.append(f"Invalid log level: {log_config.level}")


def _validate_model_config(
    model_config: DictConfig,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate model configuration section.

    Supports both new _target_ pattern and legacy provider pattern.

    Args:
        model_config: Model configuration.
        errors: List to append errors to.
        warnings: List to append warnings to.
    """
    # New _target_ pattern (single model)
    if "_target_" in model_config:
        _validate_target_model_spec(model_config, "model", errors, warnings)
        return

    # Check for single model or multi-model setup
    if "model_registry" in model_config:
        # Multi-model configuration
        registry = model_config.model_registry
        if not registry:
            errors.append("model.model_registry is empty")
            return

        for model_name, model_spec in registry.items():
            # New _target_ pattern in registry
            if "_target_" in model_spec:
                _validate_target_model_spec(
                    model_spec, f"model.model_registry.{model_name}", errors, warnings
                )
            else:
                # Legacy provider pattern
                if "provider" not in model_spec:
                    errors.append(f"model.model_registry.{model_name}.provider is required")
                elif model_spec.provider.lower() not in [
                    "openai",
                    "anthropic",
                    "ollama",
                    "mock",
                ]:
                    warnings.append(
                        f"Unknown model provider '{model_spec.provider}' for {model_name}"
                    )

                if "model_name" not in model_spec:
                    errors.append(f"model.model_registry.{model_name}.model_name is required")

    else:
        # Legacy single model configuration
        if "provider" not in model_config:
            errors.append("model.provider is required")

        if "model_name" not in model_config:
            errors.append("model.model_name is required")


def _validate_target_model_spec(
    model_spec: DictConfig,
    path: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a model specification using the _target_ pattern.

    Args:
        model_spec: Model specification with _target_.
        path: Config path for error messages.
        errors: List to append errors to.
        warnings: List to append warnings to.
    """
    target = model_spec.get("_target_", "")

    # Check that _target_ is a valid import path
    if not target or "." not in target:
        errors.append(f"{path}._target_ must be a valid fully-qualified class path")
        return

    # Warn about potentially missing dependencies for known model types
    known_targets = {
        "concordia.language_model.gpt_model.GptLanguageModel": "openai",
        "concordia.language_model.anthropic_model.AnthropicLanguageModel": "anthropic",
        "concordia.language_model.ollama_model.OllamaLanguageModel": "ollama",
        "src.utils.testing.MockLanguageModel": "mock",
        "src.models.local_model.LocalLanguageModel": "local",
    }

    if target not in known_targets:
        warnings.append(f"{path}._target_ '{target}' is not a known model class")


def _validate_scenario_config(
    scenario_config: DictConfig,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate scenario configuration section.

    Args:
        scenario_config: Scenario configuration.
        errors: List to append errors to.
        warnings: List to append warnings to.
    """
    if "name" not in scenario_config:
        warnings.append("scenario.name not set")

    if "premise" not in scenario_config:
        warnings.append("scenario.premise not set, simulation may lack context")

    # Validate knowledge system config (optional)
    _validate_knowledge_config(scenario_config, errors, warnings)

    # Check agents configuration
    if "agents" not in scenario_config:
        errors.append("scenario.agents section is required")
        return

    agents_config = scenario_config.agents
    entities = agents_config.get("entities", [])

    # Generic validation for any entity
    for i, entity in enumerate(entities):
        if "name" not in entity:
            errors.append(f"scenario.agents.entities[{i}].name is required")
        if "prefab" not in entity:
            errors.append(f"scenario.agents.entities[{i}].prefab is required")

    if len(entities) == 0:
        errors.append("At least one entity must be defined in scenario.agents.entities")

    # Optional: validate roles if defined in scenario
    if "roles" in scenario_config:
        defined_roles = {r.name for r in scenario_config.roles}
        for entity in entities:
            if "role" in entity and entity.role not in defined_roles:
                warnings.append(f"Entity '{entity.name}' has undefined role '{entity.role}'")

    # Check game master configuration
    if "game_master" not in scenario_config:
        warnings.append("scenario.game_master not set, using default")
    else:
        gm_config = scenario_config.game_master
        if "prefab" not in gm_config:
            errors.append("scenario.game_master.prefab is required")

    # Check prefabs mapping
    if "prefabs" not in scenario_config:
        errors.append("scenario.prefabs mapping is required")
    else:
        _validate_prefabs_config(scenario_config.prefabs, errors, warnings)


def _validate_knowledge_config(
    scenario_config: DictConfig,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate knowledge system configuration.

    Args:
        scenario_config: Scenario configuration.
        errors: List to append errors to.
        warnings: List to append warnings to.
    """
    # Validate setting config (optional)
    if "setting" in scenario_config:
        setting = scenario_config.setting
        if "name" not in setting:
            warnings.append("scenario.setting.name not set")
        if "background" in setting and not isinstance(
            setting.background, list | tuple | ListConfig
        ):
            errors.append("scenario.setting.background must be a list of strings")

    # Validate event config (optional)
    if "event" in scenario_config:
        event = scenario_config.event
        if "name" not in event:
            warnings.append("scenario.event.name not set")

    # Validate shared_memories (optional)
    if "shared_memories" in scenario_config:
        memories = scenario_config.shared_memories
        if not isinstance(memories, list | tuple | ListConfig):
            errors.append("scenario.shared_memories must be a list")

    # Validate initial_observations (optional)
    if "initial_observations" in scenario_config:
        observations = scenario_config.initial_observations
        if not isinstance(observations, list | tuple | ListConfig):
            errors.append("scenario.initial_observations must be a list")

    # Validate builders config (optional)
    if "builders" in scenario_config:
        builders = scenario_config.builders
        if "knowledge" in builders:
            kb = builders.knowledge
            if "module" not in kb:
                errors.append("scenario.builders.knowledge.module is required")
            if "function" not in kb:
                errors.append("scenario.builders.knowledge.function is required")
        if "events" in builders:
            eb = builders.events
            if "module" not in eb:
                errors.append("scenario.builders.events.module is required")
            if "function" not in eb:
                errors.append("scenario.builders.events.function is required")


def _validate_prefabs_config(
    prefabs_config: DictConfig,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate prefabs configuration section.

    Supports both new _target_ pattern and legacy string path pattern.

    Args:
        prefabs_config: Prefabs configuration.
        errors: List to append errors to.
        warnings: List to append warnings to.
    """
    for prefab_name, prefab_spec in prefabs_config.items():
        # New _target_ pattern
        if hasattr(prefab_spec, "get") and "_target_" in prefab_spec:
            target = prefab_spec.get("_target_", "")
            if not target or "." not in target:
                errors.append(
                    f"scenario.prefabs.{prefab_name}._target_ must be a valid "
                    "fully-qualified class path"
                )
        # Legacy string path pattern
        elif isinstance(prefab_spec, str):
            if "." not in prefab_spec:
                errors.append(
                    f"scenario.prefabs.{prefab_name} must be a valid fully-qualified class path"
                )
        else:
            errors.append(
                f"scenario.prefabs.{prefab_name} must be either a _target_ dict "
                "or a string class path"
            )


def validate_entity_model_mapping(
    mapping: dict[str, str],
    available_models: list[str],
    entity_names: list[str],
) -> list[str]:
    """Validate entity-to-model mapping.

    Args:
        mapping: Mapping from entity names to model names.
        available_models: List of available model names.
        entity_names: List of entity names that need models.

    Returns:
        List of warning messages.
    """
    warnings: list[str] = []

    for entity_name, model_name in mapping.items():
        if entity_name == "_default_":
            continue
        if model_name not in available_models:
            warnings.append(f"Entity '{entity_name}' mapped to unknown model '{model_name}'")

    # Check for entities without explicit mapping
    default_model = mapping.get("_default_")
    for entity_name in entity_names:
        if entity_name not in mapping and not default_model:
            warnings.append(f"Entity '{entity_name}' has no model mapping and no default is set")

    return warnings
