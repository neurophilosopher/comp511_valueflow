"""ValueFlow agent prefab for value perturbation propagation experiments.

Agents are simple discussion participants with neutral value orientations.
The perturbed agent receives a modified persona that strongly endorses
a target Schwartz value.
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
class ValueFlowAgent(prefab_lib.Prefab):
    """A discussion participant for ValueFlow experiments.

    Agents have:
    - A persona describing their baseline value orientation
    - Observation of peer outputs (controlled by the topology)
    - Ability to express views in each discussion round

    The perturbed agent's persona is overridden by the game master
    to strongly endorse a specific Schwartz value.
    """

    description: str = "A discussion participant for value propagation experiments."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Agent",
            "persona": "A thoughtful discussion participant with balanced values.",
            "goal": "Engage authentically in the discussion",
            "openness": "medium",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the ValueFlow agent."""
        name = str(self.params.get("name", "Agent"))
        persona = str(self.params.get("persona", "A thoughtful discussion participant."))
        goal = str(self.params.get("goal", "Engage authentically in the discussion"))
        openness = str(self.params.get("openness", "medium"))

        # Build openness instruction
        openness_descriptions = {
            "high": (
                "You are very open to being influenced by others' perspectives "
                "and readily update your views based on compelling arguments."
            ),
            "medium": (
                "You consider others' perspectives but maintain your own views "
                "unless presented with particularly strong reasoning."
            ),
            "low": (
                "You hold your views firmly and are not easily swayed by "
                "others' arguments, though you listen respectfully."
            ),
        }
        openness_desc = openness_descriptions.get(openness, openness_descriptions["medium"])

        components: dict[str, Any] = {}

        # Memory
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        # Instructions
        instructions_key = "Instructions"
        components[instructions_key] = agent_components.instructions.Instructions(
            agent_name=name,
            pre_act_label="\nIdentity",
        )

        # Persona context
        persona_key = "Persona"
        components[persona_key] = agent_components.constant.Constant(
            state=f"Persona: {persona}",
            pre_act_label="\nBackground",
        )

        # Openness to influence
        openness_key = "Openness"
        components[openness_key] = agent_components.constant.Constant(
            state=f"Social orientation: {openness_desc}",
            pre_act_label="\nDisposition",
        )

        # Goal
        goal_key = "Goal"
        components[goal_key] = agent_components.constant.Constant(
            state=f"Goal: {goal}",
            pre_act_label="\nObjective",
        )

        # Observations (peer outputs delivered by game master per topology)
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=10,
            pre_act_label="\nWhat others have said recently",
        )

        # Situation perception
        situation_key = "SituationPerception"
        components[situation_key] = (
            agent_components.question_of_recent_memories.SituationPerception(
                model=model,
                pre_act_label=(f"\nQuestion: What is {name} noticing in the discussion?\nAnswer"),
            )
        )

        # Decision — what to say next
        decision_key = "Decision"
        components[decision_key] = (
            agent_components.question_of_recent_memories.QuestionOfRecentMemories(
                model=model,
                pre_act_label=(f"\nQuestion: What should {name} say in the discussion?\nAnswer"),
                question=(
                    f"Based on {name}'s personality, values, and what they've "
                    f"heard from others, what would they say in this discussion? "
                    f"They should express their genuine perspective on values "
                    f"and life priorities."
                ),
                answer_prefix=f"{name} says: ",
                add_to_memory=False,
                num_memories_to_retrieve=10,
            )
        )

        # Acting component
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=list(components.keys()),
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )
