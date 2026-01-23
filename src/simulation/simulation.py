"""Core Simulation class adapted from mastodon-sim for Concordia v2.

This module provides the main Simulation API that manages entities, game masters,
and the simulation lifecycle with support for multi-model configurations.
"""

from __future__ import annotations

import copy
import functools
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.environment import engine as engine_lib
from concordia.environment.engines import sequential
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.typing import simulation as simulation_lib
from concordia.utils import helper_functions as helper_functions_lib
from concordia.utils import html as html_lib
from omegaconf import DictConfig

Config: TypeAlias = prefab_lib.Config
Role: TypeAlias = prefab_lib.Role


class Simulation(simulation_lib.Simulation):
    """Multi-model simulation supporting Hydra configuration.

    This simulation class extends Concordia's base simulation with:
    - Multi-model support (different LLMs for different entities)
    - Hydra configuration integration
    - Enhanced checkpointing and logging
    """

    def __init__(
        self,
        config: Config,
        models: dict[str, language_model.LanguageModel],
        entity_to_model: dict[str, str],
        embedder: Callable[[str], np.ndarray],
        engine: engine_lib.Engine | None = None,
        hydra_config: DictConfig | None = None,
    ) -> None:
        """Initialize the simulation.

        Args:
            config: Concordia prefab configuration with prefabs and instances.
            models: Dictionary mapping model names to LanguageModel instances.
            entity_to_model: Dictionary mapping entity names to model names.
            embedder: Function to embed text into vectors.
            engine: Simulation engine (defaults to Sequential).
            hydra_config: Optional Hydra configuration for additional settings.
        """
        self._config: Config = config
        self._models = models
        self._entity_to_model = entity_to_model
        self._embedder = embedder
        self._engine = engine or sequential.Sequential()
        self._hydra_config = hydra_config

        self.game_masters: list[entity_lib.Entity] = []
        self.entities: list[entity_lib.Entity] = []
        self._raw_log: list[Mapping[str, Any]] = []
        self._entity_to_prefab_config: dict[str, prefab_lib.InstanceConfig] = {}
        self._checkpoints_path: str | None = None
        self._checkpoint_counter = 0
        self._get_state_callback: Callable[[dict[str, Any]], None] | None = None

        # Shared memory bank for game masters
        self.game_master_memory_bank = associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder,
        )

        # Build instances from config
        self._build_from_config()

    def _build_from_config(self) -> None:
        """Build entities and game masters from configuration."""
        all_data = self._config.instances

        gm_configs = [cfg for cfg in all_data if cfg.role == Role.GAME_MASTER]
        entity_configs = [cfg for cfg in all_data if cfg.role == Role.ENTITY]
        initializer_configs = [cfg for cfg in all_data if cfg.role == Role.INITIALIZER]

        # Build entities first (they don't need references to other entities)
        for entity_config in entity_configs:
            self.add_entity(entity_config)

        # Build game masters (initializers first, then regular GMs)
        for gm_config in initializer_configs + gm_configs:
            self.add_game_master(gm_config)

    def _get_model_for_entity(self, entity_name: str) -> language_model.LanguageModel:
        """Get the language model for a given entity.

        Args:
            entity_name: Name of the entity.

        Returns:
            The LanguageModel instance for this entity.

        Raises:
            KeyError: If entity or its model is not found.
        """
        model_name = self._entity_to_model.get(entity_name)
        if model_name is None:
            # Try default model
            model_name = self._entity_to_model.get("_default_")
            if model_name is None:
                # Use first available model
                model_name = next(iter(self._models.keys()))

        if model_name not in self._models:
            raise KeyError(
                f"Model '{model_name}' for entity '{entity_name}' not found. "
                f"Available models: {list(self._models.keys())}"
            )

        return self._models[model_name]

    def get_raw_log(self) -> list[Mapping[str, Any]]:
        """Get a copy of the raw simulation log."""
        return copy.deepcopy(self._raw_log)

    def get_entity_prefab_config(self, entity_name: str) -> prefab_lib.InstanceConfig | None:
        """Get the prefab config for a given entity name."""
        return self._entity_to_prefab_config.get(entity_name)

    def get_game_masters(self) -> list[entity_lib.Entity]:
        """Get a copy of the game masters list."""
        return copy.copy(self.game_masters)

    def get_entities(self) -> list[entity_lib.Entity]:
        """Get a copy of the entities list."""
        return copy.copy(self.entities)

    def add_game_master(
        self,
        instance_config: prefab_lib.InstanceConfig,
        state: entity_component.EntityState | None = None,
    ) -> None:
        """Add a game master to the simulation.

        Args:
            instance_config: Configuration for the game master instance.
            state: Optional pre-loaded state for the game master.
        """
        if instance_config.role not in [Role.GAME_MASTER, Role.INITIALIZER]:
            raise ValueError("Instance config role must be GAME_MASTER or INITIALIZER")

        # Deep copy prefab to avoid mutation
        gm_prefab = copy.deepcopy(self._config.prefabs[instance_config.prefab])
        gm_prefab.params = instance_config.params
        gm_prefab.entities = self.entities  # Give GM access to entities

        gm_name = gm_prefab.params.get("name", "game_master")

        game_master = gm_prefab.build(
            model=self._get_model_for_entity(gm_name),
            memory_bank=self.game_master_memory_bank,
        )

        # Check for duplicates
        if any(gm.name == game_master.name for gm in self.game_masters):
            print(f"Game master {game_master.name} already exists, skipping.")
            return

        if state:
            game_master.set_state(state)

        self._entity_to_prefab_config[game_master.name] = instance_config
        self.game_masters.append(game_master)

    def add_entity(
        self,
        instance_config: prefab_lib.InstanceConfig,
        state: entity_component.EntityState | None = None,
    ) -> None:
        """Add an entity to the simulation.

        Args:
            instance_config: Configuration for the entity instance.
            state: Optional pre-loaded state for the entity.
        """
        if instance_config.role != Role.ENTITY:
            raise ValueError("Instance config role must be ENTITY")

        # Deep copy prefab to avoid mutation
        entity_prefab = copy.deepcopy(self._config.prefabs[instance_config.prefab])
        entity_prefab.params = instance_config.params

        entity_name = entity_prefab.params.get("name", "entity")

        # Each entity gets its own memory bank
        memory_bank = associative_memory.AssociativeMemoryBank(
            sentence_embedder=self._embedder,
        )

        entity = entity_prefab.build(
            model=self._get_model_for_entity(entity_name),
            memory_bank=memory_bank,
        )

        # Check for duplicates
        if any(e.name == entity.name for e in self.entities):
            print(f"Entity {entity.name} already exists, skipping.")
            return

        # Handle pre-loaded memory state
        memory_state = instance_config.params.get("memory_state")
        if memory_state:
            try:
                memory_component = entity.get_component("__memory__")
                memory_component.set_state(memory_state)
                print(f"Loaded pre-existing memories for {entity.name}")
            except (KeyError, TypeError, ValueError) as e:
                print(f"Error setting pre-loaded memory for {entity.name}: {e}")
                raise

        if state:
            entity.set_state(state)

        self.entities.append(entity)
        self._entity_to_prefab_config[entity.name] = instance_config

        # Update game masters with new entity list
        for game_master in self.game_masters:
            if hasattr(game_master, "entities"):
                game_master.entities = self.entities

    def play(
        self,
        premise: str | None = None,
        max_steps: int | None = None,
        raw_log: list[Mapping[str, Any]] | None = None,
        get_state_callback: Callable[[dict[str, Any]], None] | None = None,
        checkpoint_path: str | None = None,
        return_html_log: bool = True,
    ) -> str | list[Mapping[str, Any]]:
        """Run the simulation.

        Args:
            premise: Initial premise/scenario description.
            max_steps: Maximum number of simulation steps.
            raw_log: Existing log to append to.
            get_state_callback: Callback for checkpoint state.
            checkpoint_path: Path to save checkpoints.
            return_html_log: If True, return HTML log; else return raw log.

        Returns:
            HTML log string or raw log list depending on return_html_log.
        """
        if premise is None:
            premise = self._config.default_premise
        if max_steps is None:
            max_steps = self._config.default_max_steps

        if raw_log is None:
            raw_log = self._raw_log
        else:
            self._raw_log = raw_log

        self._get_state_callback = get_state_callback

        checkpoint_callback = functools.partial(
            self.save_checkpoint, checkpoint_path=checkpoint_path
        )

        # Order game masters: initializers first
        initializers = [
            gm
            for gm in self.game_masters
            if self._entity_to_prefab_config[gm.name].role == Role.INITIALIZER
        ]
        other_gms = [
            gm
            for gm in self.game_masters
            if self._entity_to_prefab_config[gm.name].role == Role.GAME_MASTER
        ]
        sorted_game_masters = initializers + other_gms

        # Run the simulation loop
        self._engine.run_loop(
            game_masters=sorted_game_masters,
            entities=self.entities,
            premise=premise,
            max_steps=max_steps,
            verbose=True,
            log=raw_log,
            checkpoint_callback=checkpoint_callback,
        )

        if not return_html_log:
            return copy.deepcopy(raw_log)

        return self._generate_html_log(raw_log)

    def _generate_html_log(self, raw_log: list[Mapping[str, Any]]) -> str:
        """Generate HTML visualization of the simulation log."""
        player_logs = []
        player_log_names = []

        scores = helper_functions_lib.find_data_in_nested_structure(raw_log, "Player Scores")

        for player in self.entities:
            if (
                not isinstance(player, entity_component.EntityWithComponents)
                or player.get_component("__memory__") is None
            ):
                continue

            entity_memory = player.get_component("__memory__")
            entity_memories = entity_memory.get_all_memories_as_text()
            player_html = html_lib.PythonObjectToHTMLConverter(entity_memories).convert()
            player_logs.append(player_html)
            player_log_names.append(f"{player.name}")

        gm_memories = self.game_master_memory_bank.get_all_memories_as_text()
        gm_html = html_lib.PythonObjectToHTMLConverter(gm_memories).convert()
        player_logs.append(gm_html)
        player_log_names.append("Game Master Memories")

        summary = ""
        if scores:
            summary = f"Player Scores: {scores[-1]}"

        results_log = html_lib.PythonObjectToHTMLConverter(copy.deepcopy(raw_log)).convert()
        tabbed_html = html_lib.combine_html_pages(
            [results_log, *player_logs],
            ["Game Master log", *player_log_names],
            summary=summary,
            title="Simulation Log",
        )
        return cast(str, html_lib.finalise_html(tabbed_html))

    def make_checkpoint_data(self) -> dict[str, Any]:
        """Create checkpoint data dictionary."""
        entities_data: dict[str, Any] = {}
        game_masters_data: dict[str, Any] = {}

        checkpoint_data: dict[str, Any] = {
            "entities": entities_data,
            "game_masters": game_masters_data,
            "raw_log": copy.deepcopy(self._raw_log),
            "checkpoint_counter": self._checkpoint_counter,
        }

        # Save entity states
        for entity in self.entities:
            if not isinstance(entity, entity_component.EntityWithComponents):
                continue
            prefab_config = self.get_entity_prefab_config(entity.name)
            if not prefab_config:
                print(f"Warning: Prefab config not found for entity {entity.name}")
                continue

            entities_data[entity.name] = {
                "prefab_type": prefab_config.prefab,
                "entity_params": prefab_config.params,
                "components": entity.get_state(),
            }

        # Save game master states
        for gm in self.game_masters:
            if not isinstance(gm, entity_component.EntityWithComponents):
                continue
            prefab_config = self.get_entity_prefab_config(gm.name)
            if not prefab_config:
                print(f"Warning: Prefab config not found for game master {gm.name}")
                continue

            game_masters_data[gm.name] = {
                "prefab_type": prefab_config.prefab,
                "entity_params": prefab_config.params,
                "role": prefab_config.role.name,
                "components": gm.get_state(),
            }

        self._checkpoint_counter += 1
        return checkpoint_data

    def save_checkpoint(self, step: int, checkpoint_path: str | None) -> None:
        """Save checkpoint at the current step.

        Args:
            step: Current simulation step.
            checkpoint_path: Directory to save checkpoint.
        """
        checkpoint_data = self.make_checkpoint_data()

        if self._get_state_callback:
            self._get_state_callback(checkpoint_data)

        if not checkpoint_path:
            return

        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        checkpoint_file = Path(checkpoint_path) / f"step_{step}_checkpoint.json"

        try:
            with checkpoint_file.open("w") as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"Step {step}: Saved checkpoint to {checkpoint_file}")
        except OSError as e:
            print(f"Error saving checkpoint at step {step}: {e}")

    def load_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Load simulation state from a checkpoint.

        Args:
            checkpoint: Checkpoint data dictionary.
        """
        # Load entities
        for entity_name, state in checkpoint.get("entities", {}).items():
            self._load_entity_from_state(entity_name, state, Role.ENTITY)

        # Load game masters
        for gm_name, state in checkpoint.get("game_masters", {}).items():
            role_name = state.get("role", "GAME_MASTER")
            role = getattr(Role, role_name, Role.GAME_MASTER)
            self._load_entity_from_state(gm_name, state, role)

        # Update game masters with entity list
        for game_master in self.game_masters:
            if hasattr(game_master, "entities"):
                game_master.entities = self.entities

        self._checkpoint_counter = checkpoint.get("checkpoint_counter", 0)
        self._raw_log = checkpoint.get("raw_log", [])

    def _load_entity_from_state(
        self,
        entity_name: str,
        state: dict[str, Any],
        role: Role,
    ) -> None:
        """Load a single entity from checkpoint state."""
        prefab_type = state.get("prefab_type")
        entity_params = state.get("entity_params")
        components_state = state.get("components")

        if not prefab_type or prefab_type not in self._config.prefabs:
            print(f"Warning: Prefab type {prefab_type} not found for {entity_name}")
            return

        if entity_params is None or components_state is None:
            print(f"Warning: Missing params or state for {entity_name}")
            return

        instance_config = prefab_lib.InstanceConfig(
            prefab=prefab_type,
            role=role,
            params=entity_params,
        )

        if role == Role.ENTITY:
            existing = next((e for e in self.entities if e.name == entity_name), None)
            if existing and isinstance(existing, entity_component.EntityWithComponents):
                existing.set_state(components_state)
            else:
                self.add_entity(instance_config, state=components_state)
        else:
            existing = next((gm for gm in self.game_masters if gm.name == entity_name), None)
            if existing and isinstance(existing, entity_component.EntityWithComponents):
                existing.set_state(components_state)
            else:
                self.add_game_master(instance_config, state=components_state)
