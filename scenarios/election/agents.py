"""Election agent prefabs.

This module provides specialized agent prefabs for the election scenario:
- VoterAgent: Citizens forming opinions and deciding how to vote
- CandidateAgent: Politicians campaigning for office
- NewsAgent: News outlet providing election coverage
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
class VoterAgent(prefab_lib.Prefab):
    """A voter agent that forms opinions and decides how to vote.

    Voters have:
    - A persona context describing their background and concerns
    - An initial political leaning (can shift during simulation)
    - A communication style affecting how they discuss politics
    """

    description: str = "A voter agent with political opinions and voting preferences."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Voter",
            "goal": "Make an informed voting decision",
            "persona_context": "A citizen participating in the election",
            "communication_style": "thoughtful",
            "initial_lean": "undecided",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the voter agent."""
        name = self.params.get("name", "Voter")
        goal = self.params.get("goal", "")
        persona_context = self.params.get("persona_context", "")
        communication_style = self.params.get("communication_style", "thoughtful")
        initial_lean = self.params.get("initial_lean", "undecided")

        components: dict[str, Any] = {}

        # Memory
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        # Instructions with persona
        style_descriptions = {
            "pragmatic": "focuses on practical outcomes and real-world impacts",
            "thoughtful": "carefully considers multiple perspectives before forming opinions",
            "analytical": "examines data and evidence to inform decisions",
            "direct": "expresses views plainly and values straightforward communication",
            "empathetic": "considers how policies affect people and communities",
        }
        style_desc = style_descriptions.get(
            communication_style, "engages thoughtfully with political topics"
        )

        instructions_key = "Instructions"
        components[instructions_key] = agent_components.instructions.Instructions(
            agent_name=name,
            pre_act_label="\nRole",
        )

        # Persona context
        persona_key = "Persona"
        components[persona_key] = agent_components.constant.Constant(
            state=f"Background: {persona_context}",
            pre_act_label="\nPersonal context",
        )

        # Political disposition
        lean_descriptions = {
            "conservative": "generally supports conservative policies",
            "slight_conservative": "leans somewhat toward conservative positions",
            "undecided": "is genuinely undecided and open to persuasion",
            "slight_progressive": "leans somewhat toward progressive positions",
            "progressive": "generally supports progressive policies",
        }
        lean_desc = lean_descriptions.get(initial_lean, "is considering their options")

        disposition_key = "Disposition"
        components[disposition_key] = agent_components.constant.Constant(
            state=f"Political stance: {name} {lean_desc}. Communication style: {style_desc}.",
            pre_act_label="\nPolitical disposition",
        )

        # Observations
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=30,
            pre_act_label="\nRecent events",
        )

        # Goal
        if goal:
            goal_key = "Goal"
            components[goal_key] = agent_components.constant.Constant(
                state=goal,
                pre_act_label="\nObjective",
            )

        # Situation perception
        situation_key = "SituationPerception"
        components[situation_key] = (
            agent_components.question_of_recent_memories.SituationPerception(
                model=model,
                pre_act_label=f"\nQuestion: What is {name}'s current understanding of the election?\nAnswer",
            )
        )

        # Vote intention analysis
        vote_key = "VoteIntention"
        components[vote_key] = (
            agent_components.question_of_recent_memories.QuestionOfRecentMemories(
                model=model,
                pre_act_label=f"\nQuestion: How is {name} currently leaning in the election?\nAnswer",
                question=(
                    f"Based on what {name} has heard and experienced, which candidate "
                    f"seems more aligned with their values and interests? What factors "
                    f"are most important in their decision?"
                ),
                answer_prefix=f"{name} is currently thinking that ",
                add_to_memory=False,
                num_memories_to_retrieve=15,
            )
        )

        # Best action
        best_action_key = "BestAction"
        components[best_action_key] = (
            agent_components.question_of_recent_memories.BestOptionPerception(
                model=model,
                pre_act_label=f"\nQuestion: What should {name} do or say next?\nAnswer",
            )
        )

        # Acting component
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=list(components.keys()),
            prefix_entity_name=True,
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )


@dataclasses.dataclass
class CandidateAgent(prefab_lib.Prefab):
    """A candidate agent that campaigns for office.

    Candidates have:
    - A partisan type (conservative, progressive)
    - Policy proposals they advocate for
    - A campaign style
    """

    description: str = "A candidate agent campaigning for election."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Candidate",
            "goal": "Win the election by persuading voters",
            "partisan_type": "moderate",
            "policy_proposals": [],
            "campaign_style": "traditional",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the candidate agent."""
        name = self.params.get("name", "Candidate")
        goal = self.params.get("goal", "")
        partisan_type = self.params.get("partisan_type", "moderate")
        policy_proposals = self.params.get("policy_proposals", [])
        campaign_style = self.params.get("campaign_style", "traditional")

        components: dict[str, Any] = {}

        # Memory
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        # Instructions
        style_descriptions = {
            "traditional": "campaigns through established channels and formal events",
            "grassroots": "emphasizes community organizing and personal connections",
            "populist": "appeals directly to ordinary citizens against established interests",
            "technocratic": "emphasizes expertise and evidence-based policy",
        }
        style_desc = style_descriptions.get(campaign_style, "campaigns to win voter support")

        instructions_key = "Instructions"
        components[instructions_key] = agent_components.instructions.Instructions(
            agent_name=name,
            pre_act_label="\nRole",
        )

        # Political platform
        partisan_descriptions = {
            "conservative": "conservative values including fiscal responsibility and traditional institutions",
            "progressive": "progressive values including social investment and systemic reform",
            "moderate": "moderate positions seeking compromise and pragmatic solutions",
        }
        partisan_desc = partisan_descriptions.get(partisan_type, "a political platform")

        platform_key = "Platform"
        platform_text = f"Political identity: {name} represents {partisan_desc}."
        if policy_proposals:
            policies_str = "\n  - ".join(policy_proposals)
            platform_text += f"\n\nKey policy proposals:\n  - {policies_str}"

        components[platform_key] = agent_components.constant.Constant(
            state=platform_text,
            pre_act_label="\nCampaign platform",
        )

        # Campaign approach
        campaign_key = "CampaignStyle"
        components[campaign_key] = agent_components.constant.Constant(
            state=f"Campaign approach: {name} {style_desc}.",
            pre_act_label="\nCampaign strategy",
        )

        # Observations
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=30,
            pre_act_label="\nCampaign events",
        )

        # Goal
        if goal:
            goal_key = "Goal"
            components[goal_key] = agent_components.constant.Constant(
                state=goal,
                pre_act_label="\nObjective",
            )

        # Situation perception
        situation_key = "SituationPerception"
        components[situation_key] = (
            agent_components.question_of_recent_memories.SituationPerception(
                model=model,
                pre_act_label=f"\nQuestion: What is the current state of {name}'s campaign?\nAnswer",
            )
        )

        # Voter analysis
        voter_key = "VoterAnalysis"
        components[voter_key] = (
            agent_components.question_of_recent_memories.QuestionOfRecentMemories(
                model=model,
                pre_act_label="\nQuestion: What are voters concerned about?\nAnswer",
                question=(
                    f"Based on recent interactions and observations, what issues are "
                    f"most important to voters? How can {name} best address their concerns?"
                ),
                answer_prefix=f"{name} observes that voters ",
                add_to_memory=False,
                num_memories_to_retrieve=10,
            )
        )

        # Best action
        best_action_key = "BestAction"
        components[best_action_key] = (
            agent_components.question_of_recent_memories.BestOptionPerception(
                model=model,
                pre_act_label=f"\nQuestion: What is the best campaign action for {name}?\nAnswer",
            )
        )

        # Acting component
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=list(components.keys()),
            prefix_entity_name=True,
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )


@dataclasses.dataclass
class NewsAgent(prefab_lib.Prefab):
    """A news agent that provides election coverage.

    News agents:
    - Report on election events and candidate activities
    - Provide headlines that influence voter opinions
    - Maintain journalistic standards
    """

    description: str = "A news agent providing election coverage."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "News",
            "goal": "Provide informative and fair election coverage",
            "outlet_style": "local_journalism",
            "headlines": [],
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the news agent."""
        name = self.params.get("name", "News")
        goal = self.params.get("goal", "")
        outlet_style = self.params.get("outlet_style", "local_journalism")
        headlines = self.params.get("headlines", [])

        components: dict[str, Any] = {}

        # Memory
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        # Instructions
        style_descriptions = {
            "local_journalism": "provides community-focused coverage with local relevance",
            "investigative": "digs deep into issues and holds candidates accountable",
            "balanced": "presents multiple perspectives without editorial bias",
            "breaking_news": "focuses on fast-paced coverage of latest developments",
        }
        style_desc = style_descriptions.get(outlet_style, "reports on election news")

        instructions_key = "Instructions"
        components[instructions_key] = agent_components.instructions.Instructions(
            agent_name=name,
            pre_act_label="\nRole",
        )

        # Journalistic approach
        approach_key = "Approach"
        components[approach_key] = agent_components.constant.Constant(
            state=f"Editorial approach: {name} {style_desc}.",
            pre_act_label="\nJournalistic standards",
        )

        # Headlines/stories ready to report
        if headlines:
            headlines_text = "\n".join(f"- {h}" for h in headlines)
            headlines_key = "Headlines"
            components[headlines_key] = agent_components.constant.Constant(
                state=headlines_text,
                pre_act_label="\nPrepared headlines",
            )

        # Observations
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=40,
            pre_act_label="\nRecent developments",
        )

        # Goal
        if goal:
            goal_key = "Goal"
            components[goal_key] = agent_components.constant.Constant(
                state=goal,
                pre_act_label="\nMission",
            )

        # Situation perception
        situation_key = "SituationPerception"
        components[situation_key] = (
            agent_components.question_of_recent_memories.SituationPerception(
                model=model,
                pre_act_label="\nQuestion: What is the current state of the election race?\nAnswer",
            )
        )

        # Story analysis
        story_key = "StoryAnalysis"
        components[story_key] = (
            agent_components.question_of_recent_memories.QuestionOfRecentMemories(
                model=model,
                pre_act_label="\nQuestion: What are the most newsworthy developments?\nAnswer",
                question=(
                    "Based on recent events, what stories would be most relevant to "
                    "voters? What angles would best serve the public interest?"
                ),
                answer_prefix="The most newsworthy developments are ",
                add_to_memory=False,
                num_memories_to_retrieve=15,
            )
        )

        # Best action
        best_action_key = "BestAction"
        components[best_action_key] = (
            agent_components.question_of_recent_memories.BestOptionPerception(
                model=model,
                pre_act_label=f"\nQuestion: What should {name} report on next?\nAnswer",
            )
        )

        # Acting component
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=list(components.keys()),
            prefix_entity_name=True,
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )
