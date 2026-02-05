"""Social media user agent prefab for misinformation scenario."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

# Action format instructions for agents
ACTION_FORMAT_INSTRUCTIONS = """
You must respond with ONE action in this exact format:
ACTION: <type> | TARGET: <id or username or none> | CONTENT: <text or none>

Available actions:
- post: Share a new thought (TARGET: none, CONTENT: your message)
- reply: Respond to a post (TARGET: post number, CONTENT: your reply)
- like: Like a post (TARGET: post number, CONTENT: none)
- boost: Repost to your followers (TARGET: post number, CONTENT: none)
- follow: Follow a user (TARGET: username, CONTENT: none)
- unfollow: Unfollow a user (TARGET: username, CONTENT: none)
- skip: Do nothing this turn (TARGET: none, CONTENT: none)

Examples:
ACTION: post | TARGET: none | CONTENT: Just saw something interesting today!
ACTION: reply | TARGET: 42 | CONTENT: I'm not sure that's accurate
ACTION: like | TARGET: 15 | CONTENT: none
ACTION: boost | TARGET: 23 | CONTENT: none
ACTION: follow | TARGET: Alice | CONTENT: none
ACTION: skip | TARGET: none | CONTENT: none
""".strip()


@dataclasses.dataclass
class SocialMediaUserAgent(prefab_lib.Prefab):
    """A social media user agent.

    Users have:
    - A persona describing their background and behavior
    - Goals affecting what they post and engage with
    - Observation of their timeline
    - Ability to post, reply, like, boost, follow, unfollow
    """

    description: str = "A social media user agent."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "User",
            "persona": "A typical social media user",
            "goal": "Engage authentically on social media",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the social media user agent."""
        name = str(self.params.get("name", "User"))
        persona = str(self.params.get("persona", "A typical social media user"))
        goal = str(self.params.get("goal", "Engage authentically on social media"))

        components: dict[str, Any] = {}

        # Memory
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        # Instructions - who they are
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

        # Goal
        goal_key = "Goal"
        components[goal_key] = agent_components.constant.Constant(
            state=f"Goal: {goal}",
            pre_act_label="\nObjective",
        )

        # Observations (timeline from social media app)
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=5,  # Keep recent timelines in context
            pre_act_label="\nYour social media feed",
        )

        # Situation perception - what's happening on social media
        situation_key = "SituationPerception"
        components[situation_key] = (
            agent_components.question_of_recent_memories.SituationPerception(
                model=model,
                pre_act_label=f"\nQuestion: What is {name} noticing on social media?\nAnswer",
            )
        )

        # Decision making - what to do next
        decision_key = "Decision"
        components[decision_key] = (
            agent_components.question_of_recent_memories.QuestionOfRecentMemories(
                model=model,
                pre_act_label=f"\nQuestion: What should {name} do on social media?\nAnswer",
                question=(
                    f"Based on {name}'s personality and what they see on their feed, "
                    f"what action would they most likely take? Consider: posting something new, "
                    f"replying to an interesting post, liking content they agree with, "
                    f"boosting something worth sharing, or just scrolling past (skip)."
                ),
                answer_prefix=f"{name} decides to ",
                add_to_memory=False,
                num_memories_to_retrieve=10,
            )
        )

        # Action format instructions
        format_key = "ActionFormat"
        components[format_key] = agent_components.constant.Constant(
            state=ACTION_FORMAT_INSTRUCTIONS,
            pre_act_label="\nAction format (YOU MUST USE THIS FORMAT)",
        )

        # Acting component
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=list(components.keys()),
            prefix_entity_name=False,  # Don't prefix, we want raw action format
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )
