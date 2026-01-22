"""Configuration helper utilities.

This module provides helper functions for working with Hydra configurations.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig


def get_model_config(config: DictConfig, model_name: str | None = None) -> dict[str, Any]:
    """Get model configuration, optionally for a specific model.

    Args:
        config: Full Hydra configuration.
        model_name: Optional model name for multi-model configs.

    Returns:
        Model configuration dictionary.

    Raises:
        KeyError: If model_name is not found in registry.
    """
    model_config = config.model

    # Single model configuration
    if "model_registry" not in model_config:
        return {
            "provider": model_config.provider,
            "model_name": model_config.model_name,
            "parameters": dict(model_config.get("parameters", {})),
            "api": dict(model_config.get("api", {})),
        }

    # Multi-model configuration
    if model_name is None:
        model_name = model_config.default_model

    if model_name not in model_config.model_registry:
        raise KeyError(f"Model '{model_name}' not found in model_registry")

    spec = model_config.model_registry[model_name]
    return {
        "provider": spec.provider,
        "model_name": spec.model_name,
        "parameters": dict(spec.get("parameters", {})),
        "api": dict(spec.get("api", {})),
    }


def get_scenario_config(config: DictConfig) -> dict[str, Any]:
    """Get scenario configuration as a dictionary.

    Args:
        config: Full Hydra configuration.

    Returns:
        Scenario configuration dictionary.
    """
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config.scenario, resolve=True)


def resolve_prefab_path(prefab_path: str) -> type:
    """Resolve a prefab class from its module path.

    Args:
        prefab_path: Fully qualified class path (e.g., 'module.ClassName').

    Returns:
        The prefab class.

    Raises:
        ImportError: If module cannot be imported.
        AttributeError: If class is not found in module.
    """
    module_path, class_name = prefab_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_checkpoint_path(config: DictConfig) -> Path | None:
    """Get checkpoint path from configuration.

    Args:
        config: Full Hydra configuration.

    Returns:
        Checkpoint path or None if disabled.
    """
    sim_config = config.simulation
    checkpoint_config = sim_config.execution.get("checkpoint", {})

    if not checkpoint_config.get("enabled", False):
        return None

    path = checkpoint_config.get("path")
    return Path(path) if path else None


def get_output_paths(config: DictConfig) -> dict[str, Path]:
    """Get all output file paths from configuration.

    Args:
        config: Full Hydra configuration.

    Returns:
        Dictionary of output type to path.
    """
    paths: dict[str, Path] = {}

    sim_config = config.simulation
    logging_config = sim_config.get("logging", {})

    if logging_config.get("save_html", False):
        html_path = logging_config.get("html_path")
        if html_path:
            paths["html"] = Path(html_path)

    if logging_config.get("save_raw", False):
        raw_path = logging_config.get("raw_path")
        if raw_path:
            paths["raw_log"] = Path(raw_path)

    # Evaluation results
    eval_config = config.get("evaluation", {})
    if eval_config.get("save_results", False):
        results_path = eval_config.get("results_path")
        if results_path:
            paths["evaluation"] = Path(results_path)

    return paths


def merge_agent_params(
    base_params: dict[str, Any],
    override_params: dict[str, Any],
) -> dict[str, Any]:
    """Merge base agent parameters with overrides.

    Args:
        base_params: Base parameter dictionary.
        override_params: Override parameter dictionary.

    Returns:
        Merged parameters.
    """
    result = dict(base_params)

    for key, value in override_params.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively merge nested dicts
            result[key] = merge_agent_params(result[key], value)
        else:
            result[key] = value

    return result


def get_entity_names(config: DictConfig) -> list[str]:
    """Get all entity names from scenario configuration.

    Args:
        config: Full Hydra configuration.

    Returns:
        List of entity names.
    """
    names: list[str] = []
    agents_config = config.scenario.get("agents", {})

    for agent_type in ["buyers", "sellers"]:
        for agent in agents_config.get(agent_type, []):
            names.append(agent.name)

    if "auctioneer" in agents_config:
        names.append(agents_config.auctioneer.name)

    return names


def get_game_master_name(config: DictConfig) -> str:
    """Get game master name from configuration.

    Args:
        config: Full Hydra configuration.

    Returns:
        Game master name.
    """
    gm_config = config.scenario.get("game_master", {})
    return gm_config.get("name", "narrator")
