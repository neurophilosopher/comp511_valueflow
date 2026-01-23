"""Planning Agent prefab with goal-oriented planning capabilities.

This module provides an agent prefab that includes planning and
self-reflection components for more sophisticated decision-making.
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
class PlanningAgent(prefab_lib.Prefab):
    """An agent prefab with planning and reflection capabilities.

    This prefab extends BasicEntity with:
    - Plan generation component
    - Self-reflection component
    - Available options analysis
    - Best action selection

    Attributes:
        description: Human-readable description of this prefab.
        params: Default parameters for agent instances.
    """

    description: str = "An agent with planning, reflection, and strategic action selection."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Agent",
            "goal": "",
            "personality": "",
            "background": "",
            "planning_horizon": 3,  # How many steps ahead to plan
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the planning agent with all components.

        Args:
            model: Language model for reasoning and planning.
            memory_bank: Memory bank for associative memory.

        Returns:
            Configured EntityAgentWithLogging instance.
        """
        name = self.params.get("name", "Agent")
        goal = self.params.get("goal", "")
        personality = self.params.get("personality", "")
        background = self.params.get("background", "")

        components: dict[str, Any] = {}

        # === MEMORY COMPONENT ===
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        # === INSTRUCTIONS COMPONENT ===
        instructions_key = "Instructions"
        instructions_content = f"You are {name}, a thoughtful and strategic agent."
        if personality:
            instructions_content += f" {personality}"
        if background:
            instructions_content += f" {background}"

        components[instructions_key] = agent_components.instructions.Instructions(
            agent_name=name,
            pre_act_label="\nRole playing instructions",
        )

        # === OBSERVATION COMPONENTS ===
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=50,
            pre_act_label="\nRecent observations",
        )

        # === GOAL COMPONENT ===
        if goal:
            goal_key = "Goal"
            components[goal_key] = agent_components.constant.Constant(
                state=goal,
                pre_act_label="\nPrimary goal",
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

        # Self-reflection on recent experiences
        reflection_key = "SelfReflection"
        components[reflection_key] = (
            agent_components.question_of_recent_memories.QuestionOfRecentMemories(
                model=model,
                pre_act_label=f"\nQuestion: What has {name} learned from recent experiences?\nAnswer",
                question=(
                    f"Reflect on {name}'s recent experiences. What lessons or insights "
                    f"can be drawn? How should this inform future actions?"
                ),
                answer_prefix=f"{name} reflects: ",
                add_to_memory=False,
                num_memories_to_retrieve=10,
            )
        )

        # Available options analysis
        options_key = "AvailableOptions"
        components[options_key] = (
            agent_components.question_of_recent_memories.AvailableOptionsPerception(
                model=model,
                pre_act_label=f"\nQuestion: What options are available to {name}?\nAnswer",
            )
        )

        # Best action selection
        best_action_key = "BestActionSelection"
        components[best_action_key] = (
            agent_components.question_of_recent_memories.BestOptionPerception(
                model=model,
                pre_act_label=f"\nQuestion: What is the best action for {name} to take?\nAnswer",
            )
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
