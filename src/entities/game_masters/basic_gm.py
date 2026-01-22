"""Basic Game Master prefab with standard GM components.

This module provides a generic game master prefab using Concordia's
SwitchAct pattern for managing entity turns and event resolution.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class BasicGameMaster(prefab_lib.Prefab):
    """A basic game master prefab for turn-based simulations.

    This prefab provides:
    - Shared memory across GM operations
    - Instructions and player list components
    - Observation generation for players
    - Turn order management (all entities each round)
    - Event resolution with narrative style

    Attributes:
        description: Human-readable description of this prefab.
        params: Default parameters for GM instances.
        entities: List of entities managed by this GM (set at build time).
    """

    description: str = "A basic game master for turn-based simulation management."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "narrator",
            "acting_order": "fixed",  # "fixed" or "random"
            "world_description": "",
            "rules": [],
        }
    )

    # Game masters receive entity references before build
    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the game master with all components.

        Args:
            model: Language model for narrative generation.
            memory_bank: Shared memory bank for GM operations.

        Returns:
            Configured EntityAgentWithLogging instance for the GM.
        """
        name = self.params.get("name", "narrator")
        world_description = self.params.get("world_description", "")
        rules = self.params.get("rules", [])

        player_names = [e.name for e in self.entities]

        components: dict[str, Any] = {}

        # === MEMORY COMPONENT ===
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank
        )

        # === GM INSTRUCTIONS ===
        instructions_key = "instructions"
        components[instructions_key] = gm_components.instructions.Instructions()

        # === WORLD DESCRIPTION (if provided) ===
        if world_description:
            world_key = "world"
            components[world_key] = agent_components.constant.Constant(
                state=world_description,
                pre_act_label="\nWorld description",
            )

        # === RULES (if provided) ===
        if rules:
            rules_text = "\n".join(f"- {rule}" for rule in rules)
            rules_key = "rules"
            components[rules_key] = agent_components.constant.Constant(
                state=rules_text,
                pre_act_label="\nWorld rules",
            )

        # === PLAYER LIST ===
        players_key = "players"
        components[players_key] = gm_components.instructions.PlayerCharacters(
            player_characters=player_names,
        )

        # === OBSERVATION TO MEMORY (stores observations in memory for scanning) ===
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        # === OBSERVATION HISTORY ===
        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=100,
        )

        # === MAKE OBSERVATION (generates what players see) ===
        make_obs_key = gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
        context_components = [instructions_key, players_key]
        if world_description:
            context_components.append("world")
        if rules:
            context_components.append("rules")

        components[make_obs_key] = gm_components.make_observation.MakeObservation(
            model=model,
            player_names=player_names,
            components=context_components,
        )

        # === NEXT ACTOR (determines who acts next) ===
        next_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
        components[next_key] = gm_components.next_acting.NextActingInFixedOrder(
            sequence=player_names,
        )

        # === EVENT RESOLUTION ===
        # Default thought chain: identity just passes through the event
        event_resolution_steps = [thought_chains.identity]

        resolution_key = gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
        components[resolution_key] = gm_components.event_resolution.EventResolution(
            model=model,
            event_resolution_steps=event_resolution_steps,
            components=context_components,
            notify_observers=True,
        )

        # === GM ACTING COMPONENT (SwitchAct) ===
        act_component = gm_components.switch_act.SwitchAct(
            model=model,
            entity_names=player_names,
            component_order=list(components.keys()),
        )

        # === CREATE GM AGENT ===
        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )
