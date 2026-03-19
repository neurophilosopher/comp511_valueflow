"""Base simulator wrapping Hydra configuration.

This module provides the BaseSimulator class that serves as the foundation
for all simulator implementations, handling Hydra config integration.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import concordia.prefabs.game_master as game_master_prefabs
import numpy as np
from concordia.environment import engine as engine_lib
from concordia.environment.engines import sequential
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.evaluation import ProbeRunner
from src.simulation.engines import patch_concordia_parser
from src.simulation.simulation import Simulation

logger = logging.getLogger(__name__)


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
        self._probe_runner: ProbeRunner | None = None

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

        Merges Concordia's built-in prefabs with scenario-specific prefabs.
        Supports both new _target_ pattern and legacy string path pattern.

        Returns:
            Dictionary mapping prefab names to Prefab instances.
        """
        # Load Concordia's built-in prefabs (includes formative_memories_initializer__GameMaster)
        concordia_prefabs = helper_functions.get_package_classes(game_master_prefabs)

        # Load scenario-specific prefabs
        scenario_config = self._config.scenario
        scenario_prefabs: dict[str, prefab_lib.Prefab] = {}

        prefab_mappings = scenario_config.get("prefabs", {})

        for prefab_name, prefab_config in prefab_mappings.items():
            # New _target_ pattern
            if isinstance(prefab_config, DictConfig) and "_target_" in prefab_config:
                scenario_prefabs[prefab_name] = instantiate(prefab_config)
            # Legacy string path pattern
            elif isinstance(prefab_config, str):
                prefab_class = self._load_class(prefab_config)
                scenario_prefabs[prefab_name] = prefab_class()
            else:
                raise ValueError(
                    f"Invalid prefab config for {prefab_name}: "
                    f"expected _target_ dict or string path, got {type(prefab_config)}"
                )

        # Merge: scenario prefabs override Concordia prefabs
        return {**concordia_prefabs, **scenario_prefabs}

    def build_agent_knowledge(
        self,
        agent_name: str,
        agent_role: str,
        params: dict[str, Any],
    ) -> list[str]:
        """Dynamically build agent knowledge using scenario builders.

        Calls the knowledge builder function specified in scenario config
        to generate role-specific and agent-specific knowledge.

        Args:
            agent_name: Name of the agent.
            agent_role: Role of the agent as defined in the scenario config.
            params: Agent parameters from config.

        Returns:
            List of knowledge strings for the agent's memory.
        """
        builders = self._config.scenario.get("builders", {})
        knowledge_builder = builders.get("knowledge", {})

        if not knowledge_builder:
            return []

        try:
            module = importlib.import_module(knowledge_builder["module"])
            func = getattr(module, knowledge_builder["function"])
            result: list[str] = func(agent_name, agent_role, params)
            return result
        except (ImportError, AttributeError, KeyError) as e:
            # Log warning but don't fail - knowledge is optional
            import logging

            logging.getLogger(__name__).warning(f"Failed to build knowledge for {agent_name}: {e}")
            return []

    def _flatten_shared_memories(self, memories: list[Any]) -> list[str]:
        """Flatten nested memory lists into a single list of strings.

        Handles OmegaConf lists and nested lists from variable interpolation.

        Args:
            memories: List that may contain strings or nested lists.

        Returns:
            Flat list of memory strings.
        """
        result: list[str] = []
        for item in memories:
            if isinstance(item, list | ListConfig):
                result.extend(self._flatten_shared_memories(list(item)))
            elif isinstance(item, str):
                result.append(item)
            else:
                # Convert other types to string
                result.append(str(item))
        return result

    def build_instances(self) -> list[prefab_lib.InstanceConfig]:
        """Build instance configurations from scenario.

        Creates entity instances, builds knowledge for each agent,
        and creates an initializer game master to inject memories.

        Returns:
            List of InstanceConfig objects for all entities.
        """
        scenario_config = self._config.scenario
        instances: list[prefab_lib.InstanceConfig] = []

        # Build agent instances
        agents_config = scenario_config.get("agents", {})
        entities = agents_config.get("entities", [])

        # Track player-specific memories and context for initializer
        player_specific_memories: dict[str, list[str]] = {}
        player_specific_context: dict[str, str] = {}

        # Generic entity processing - works for any scenario
        for entity in entities:
            entity_params = cast(
                dict[str, Any],
                OmegaConf.to_container(entity.get("params", {}), resolve=True),
            )
            entity_params["name"] = entity.name
            entity_role = entity.get("role", "")

            # Pass scenario role to prefab if defined
            if entity_role:
                entity_params["scenario_role"] = entity_role

            instances.append(
                prefab_lib.InstanceConfig(
                    prefab=entity.prefab,
                    role=prefab_lib.Role.ENTITY,
                    params=entity_params,
                )
            )

            # Build agent-specific knowledge using the knowledge builder
            knowledge = self.build_agent_knowledge(
                entity.name,
                entity_role,
                entity_params,
            )
            if knowledge:
                player_specific_memories[entity.name] = knowledge

            # Build player-specific context from goal or other params
            if "goal" in entity_params:
                player_specific_context[entity.name] = str(entity_params["goal"])

        # Build game master instance
        gm_config = scenario_config.get("game_master", {})
        gm_name = gm_config.get("name", "game_master") if gm_config else "game_master"

        if gm_config:
            gm_params = cast(
                dict[str, Any],
                OmegaConf.to_container(gm_config.get("params", {}), resolve=True),
            )
            instances.append(
                prefab_lib.InstanceConfig(
                    prefab=gm_config.prefab,
                    role=prefab_lib.Role.GAME_MASTER,
                    params=gm_params | {"name": gm_name},
                )
            )

        # Build shared memories from config (with variable resolution)
        raw_memories = scenario_config.get("shared_memories", [])
        if isinstance(raw_memories, DictConfig | ListConfig):
            raw_shared_memories = cast(
                list[Any],
                OmegaConf.to_container(raw_memories, resolve=True),
            )
        else:
            raw_shared_memories = list(raw_memories) if raw_memories else []
        shared_memories = self._flatten_shared_memories(raw_shared_memories)

        # Create initializer game master instance (inserted at start)
        # This uses Concordia's built-in formative_memories_initializer prefab
        if shared_memories or player_specific_memories:
            initializer_instance = prefab_lib.InstanceConfig(
                prefab="formative_memories_initializer__GameMaster",
                role=prefab_lib.Role.INITIALIZER,
                params={
                    "name": "initial setup",
                    "next_game_master_name": gm_name,
                    "shared_memories": shared_memories,
                    "player_specific_memories": player_specific_memories,
                    "player_specific_context": player_specific_context,
                },
            )
            # Insert at the beginning so it runs first
            instances.insert(0, initializer_instance)

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

    def get_agent_role_mapping(self) -> dict[str, str]:
        """Get agent name to role mapping from configuration.

        Used by ProbeRunner to determine which probes apply to which agents.

        Returns:
            Dictionary mapping agent names to their scenario roles.
        """
        role_mapping: dict[str, str] = {}
        agents_config = self._config.scenario.get("agents", {})
        entities = agents_config.get("entities", [])

        for entity in entities:
            role = entity.get("role", "")
            if role:
                role_mapping[entity.name] = role

        return role_mapping

    def _create_probe_runner(self) -> ProbeRunner | None:
        """Create ProbeRunner if evaluation config exists.

        Returns:
            ProbeRunner instance or None if no evaluation config.
        """
        evaluation_config = self._config.get("evaluation")
        if not evaluation_config:
            logger.debug("No evaluation config found, skipping probe setup")
            return None

        metrics = evaluation_config.get("metrics", {})
        if not metrics:
            logger.debug("No metrics defined in evaluation config")
            return None

        # Get output directory from experiment config (Hydra sets this)
        output_dir = self._config.experiment.get("output_dir", "outputs")
        output_path = Path(output_dir)

        role_mapping = self.get_agent_role_mapping()

        probe_runner = ProbeRunner(
            config=evaluation_config,
            output_dir=output_path,
            role_mapping=role_mapping,
        )

        logger.info(f"Created ProbeRunner with {len(probe_runner.probes)} probes")
        return probe_runner

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

    def _create_engine(self) -> engine_lib.Engine:
        """Create simulation engine based on environment configuration.

        Returns:
            Engine instance (Sequential, Simultaneous, or SocialMedia).
        """
        env_config = self._config.get("environment", {})
        engine_type = env_config.get("engine", "sequential")

        if engine_type == "social_media":
            from src.environments.social_media.engine import SocialMediaEngine

            return SocialMediaEngine()
        elif engine_type == "valueflow":
            from scenarios.valueflow.engine import ValueFlowEngine

            topology_config = OmegaConf.to_container(
                self._config.scenario.get("topology", {}), resolve=True
            )
            interaction_config = OmegaConf.to_container(
                self._config.scenario.get("interaction", {}), resolve=True
            )
            return ValueFlowEngine(
                topology_config=topology_config or {},
                interaction_config=interaction_config or {},
            )
        elif engine_type == "simultaneous":
            from concordia.environment.engines import simultaneous

            return simultaneous.Simultaneous()
        else:
            # Default to sequential
            return sequential.Sequential()

    def setup(self) -> None:
        """Set up the simulation by creating all components.

        This method creates models, embedder, ProbeRunner, and the Simulation instance.
        """
        # Apply Concordia patches for LLM compatibility
        patch_concordia_parser()

        models = self.create_models()
        self._embedder = self.create_embedder()

        # Create ProbeRunner if evaluation config exists
        self._probe_runner = self._create_probe_runner()

        # Create engine based on scenario config
        engine = self._create_engine()

        concordia_config = self.build_config()
        entity_model_mapping = self.get_entity_model_mapping()

        self._simulation = Simulation(
            config=concordia_config,
            models=models,
            entity_to_model=entity_model_mapping,
            embedder=self._embedder,
            engine=engine,
            hydra_config=self._config,
            probe_runner=self._probe_runner,
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
        return cast("type", getattr(module, class_name))
