"""Basic Entity prefab with standard components.

This module provides a generic, reusable entity prefab that can be
extended or used directly for simple agents.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class BasicEntity(prefab_lib.Prefab):
    """A basic entity prefab with essential components.

    This prefab provides:
    - Associative memory for storing observations
    - Instructions component for role-playing context
    - Observation components for tracking recent events
    - Situation perception for understanding current context
    - Optional goal component

    Attributes:
        description: Human-readable description of this prefab.
        params: Default parameters for entity instances.
    """

    description: str = "A basic agent with memory, observation, and reasoning capabilities."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Agent",
            "goal": "",
            "personality": "",
            "background": "",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the entity agent with all components.

        Args:
            model: Language model for reasoning.
            memory_bank: Memory bank for associative memory.

        Returns:
            Configured EntityAgentWithLogging instance.
        """
        name = self.params.get("name", "Agent")
        goal = self.params.get("goal", "")
        personality = self.params.get("personality", "")
        background = self.params.get("background", "")

        components: dict[str, Any] = {}

        # === MEMORY COMPONENT (Required) ===
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        # === INSTRUCTIONS COMPONENT ===
        instructions_key = "Instructions"
        instructions_text = f"You are {name}."
        if personality:
            instructions_text += f" {personality}"
        if background:
            instructions_text += f" {background}"

        components[instructions_key] = agent_components.instructions.Instructions(
            agent_name=name,
            pre_act_label="\nRole playing instructions",
        )

        # === OBSERVATION COMPONENTS ===
        # Store observations to memory
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        # Retrieve recent observations for context
        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=50,
            pre_act_label="\nRecent observations",
        )

        # === REASONING COMPONENTS ===
        # Situation perception
        situation_key = "SituationPerception"
        components[situation_key] = (
            agent_components.question_of_recent_memories.SituationPerception(
                model=model,
                pre_act_label=f"\nQuestion: What situation is {name} currently in?\nAnswer",
            )
        )

        # === GOAL COMPONENT (Optional) ===
        if goal:
            goal_key = "Goal"
            components[goal_key] = agent_components.constant.Constant(
                state=goal,
                pre_act_label="\nCurrent goal",
            )

        # === ACTING COMPONENT ===
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=list(components.keys()),
            prefix_entity_name=True,
        )

        # === CREATE AGENT ===
        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )
