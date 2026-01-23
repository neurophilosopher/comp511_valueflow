"""Base simulator wrapping Hydra configuration.

This module provides the BaseSimulator class that serves as the foundation
for all simulator implementations, handling Hydra config integration.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib
from omegaconf import DictConfig, OmegaConf

from src.simulation.simulation import Simulation


class BaseSimulator(ABC):
    """Base simulator class that integrates Hydra configuration with Concordia.

    This abstract class handles:
    - Loading and validating Hydra configuration
    - Building prefabs from config
    - Creating the Simulation instance
    - Managing the simulation lifecycle
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the simulator with Hydra configuration.

        Args:
            config: Hydra DictConfig containing all simulation settings.
        """
        self._config = config
        self._simulation: Simulation | None = None
        self._model: language_model.LanguageModel | None = None
        self._embedder: Callable[[str], np.ndarray] | None = None

    @property
    def config(self) -> DictConfig:
        """Get the Hydra configuration."""
        return self._config

    @property
    def simulation(self) -> Simulation | None:
        """Get the Simulation instance."""
        return self._simulation

    @abstractmethod
    def create_models(self) -> dict[str, language_model.LanguageModel]:
        """Create language model instances from configuration.

        Returns:
            Dictionary mapping model names to LanguageModel instances.
        """

    @abstractmethod
    def create_embedder(self) -> Callable[[str], np.ndarray]:
        """Create the text embedder function.

        Returns:
            Function that embeds text into vectors.
        """

    def build_prefabs(self) -> dict[str, prefab_lib.Prefab]:
        """Build prefab instances from scenario configuration.

        Returns:
            Dictionary mapping prefab names to Prefab instances.
        """
        scenario_config = self._config.scenario
        prefabs: dict[str, prefab_lib.Prefab] = {}

        # Load prefabs from scenario config
        prefab_mappings = scenario_config.get("prefabs", {})

        for prefab_name, prefab_path in prefab_mappings.items():
            prefab_class = self._load_class(prefab_path)
            prefabs[prefab_name] = prefab_class()

        return prefabs

    def build_instances(self) -> list[prefab_lib.InstanceConfig]:
        """Build instance configurations from scenario.

        Returns:
            List of InstanceConfig objects for all entities.
        """
        scenario_config = self._config.scenario
        instances: list[prefab_lib.InstanceConfig] = []

        # Build agent instances
        agents_config = scenario_config.get("agents", {})

        # Generic entity processing - works for any scenario
        for entity in agents_config.get("entities", []):
            entity_params = OmegaConf.to_container(entity.get("params", {}), resolve=True)
            entity_params["name"] = entity.name
            # Pass scenario role to prefab if defined
            if "role" in entity:
                entity_params["scenario_role"] = entity.role

            instances.append(
                prefab_lib.InstanceConfig(
                    prefab=entity.prefab,
                    role=prefab_lib.Role.ENTITY,
                    params=entity_params,
                )
            )

        # Build game master instance
        gm_config = scenario_config.get("game_master", {})
        if gm_config:
            instances.append(
                prefab_lib.InstanceConfig(
                    prefab=gm_config.prefab,
                    role=prefab_lib.Role.GAME_MASTER,
                    params=OmegaConf.to_container(gm_config.get("params", {}), resolve=True)
                    | {"name": gm_config.name},
                )
            )

        return instances

    def get_entity_model_mapping(self) -> dict[str, str]:
        """Get entity to model mapping from configuration.

        Returns:
            Dictionary mapping entity names to model names.
        """
        model_config = self._config.model

        # For single-model configs, all entities use the same model
        if "model_registry" not in model_config:
            return {"_default_": model_config.name}

        # For multi-model configs, use the mapping
        mapping = OmegaConf.to_container(
            model_config.get("entity_model_mapping", {}),
            resolve=True,
        )
        return dict(mapping) if mapping else {"_default_": model_config.default_model}

    def build_config(self) -> prefab_lib.Config:
        """Build the Concordia Config object.

        Returns:
            Concordia Config with prefabs, instances, and defaults.
        """
        scenario_config = self._config.scenario
        simulation_config = self._config.simulation

        return prefab_lib.Config(
            prefabs=self.build_prefabs(),
            instances=self.build_instances(),
            default_premise=scenario_config.get("premise", ""),
            default_max_steps=simulation_config.execution.get("max_steps", 100),
        )

    def setup(self) -> None:
        """Set up the simulation by creating all components.

        This method creates models, embedder, and the Simulation instance.
        """
        models = self.create_models()
        self._embedder = self.create_embedder()

        concordia_config = self.build_config()
        entity_model_mapping = self.get_entity_model_mapping()

        self._simulation = Simulation(
            config=concordia_config,
            models=models,
            entity_to_model=entity_model_mapping,
            embedder=self._embedder,
            hydra_config=self._config,
        )

    def run(self) -> str | list[Mapping[str, Any]]:
        """Run the simulation.

        Returns:
            HTML log or raw log depending on configuration.

        Raises:
            RuntimeError: If setup() has not been called.
        """
        if self._simulation is None:
            raise RuntimeError("Simulation not initialized. Call setup() first.")

        simulation_config = self._config.simulation
        scenario_config = self._config.scenario

        checkpoint_path = None
        if simulation_config.execution.checkpoint.enabled:
            checkpoint_path = simulation_config.execution.checkpoint.path

        return self._simulation.play(
            premise=scenario_config.get("premise"),
            max_steps=simulation_config.execution.max_steps,
            checkpoint_path=checkpoint_path,
            return_html_log=simulation_config.logging.save_html,
        )

    def _load_class(self, class_path: str) -> type:
        """Dynamically load a class from a module path.

        Args:
            class_path: Fully qualified class path (e.g., 'module.submodule.ClassName').

        Returns:
            The loaded class.

        Raises:
            ImportError: If the module or class cannot be found.
        """
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return cast(type, getattr(module, class_name))
