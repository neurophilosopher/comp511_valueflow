"""Election Game Master prefab.

This module provides the ElectionGameMaster that manages the election
simulation, including campaign events, voter interactions, and election rules.
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
class ElectionGameMaster(prefab_lib.Prefab):
    """Game master for the election scenario.

    Manages:
    - Campaign events (debates, rallies, town halls)
    - Voter interactions and political discourse
    - Election rules and fair conduct
    - Turn order for candidates, voters, and news
    """

    description: str = "Game master for election simulation with campaign management."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "election_master",
            "candidates": [],
            "election_rules": [
                "All citizens have the right to express their political views",
                "Candidates should present their policies honestly",
                "Voters should consider issues carefully before deciding",
                "Civil discourse is expected during political discussions",
            ],
            "campaign_events": ["town_hall", "debate", "rally", "endorsement"],
        }
    )

    # Entities are set before build
    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the election game master."""
        name = self.params.get("name", "election_master")
        election_rules = self.params.get("election_rules", [])
        candidates = self.params.get("candidates", [])

        player_names = [e.name for e in self.entities]

        components: dict[str, Any] = {}

        # Memory
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        # GM Instructions
        instructions_key = "instructions"
        components[instructions_key] = gm_components.instructions.Instructions()

        # Election description
        election_desc_key = "election_description"
        candidates_str = " and ".join(candidates) if candidates else "the candidates"
        election_description = (
            f"This is an election simulation where voters are deciding between "
            f"{candidates_str}. The campaign is in full swing with candidates "
            f"sharing their platforms, news outlets covering the race, and voters "
            f"discussing the issues. The goal is to simulate realistic political "
            f"discourse and opinion formation."
        )
        components[election_desc_key] = agent_components.constant.Constant(
            state=election_description,
            pre_act_label="\nElection setting",
        )

        # Election rules
        if election_rules:
            rules_text = "\n".join(f"- {rule}" for rule in election_rules)
            rules_key = "election_rules"
            components[rules_key] = agent_components.constant.Constant(
                state=rules_text,
                pre_act_label="\nElection rules",
            )

        # Candidates list
        if candidates:
            candidates_key = "candidates"
            candidates_text = f"The candidates in this election are: {', '.join(candidates)}."
            components[candidates_key] = agent_components.constant.Constant(
                state=candidates_text,
                pre_act_label="\nCandidates",
            )

        # Player characters
        players_key = "players"
        components[players_key] = gm_components.instructions.PlayerCharacters(
            player_characters=player_names,
        )

        # Observation to memory (stores observations for EventResolution scanning)
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        # Observation history
        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=100,
        )

        # Make observation for players
        make_obs_key = gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
        context_keys = [instructions_key, election_desc_key, players_key]
        if election_rules:
            context_keys.append("election_rules")
        if candidates:
            context_keys.append("candidates")

        components[make_obs_key] = gm_components.make_observation.MakeObservation(
            model=model,
            player_names=player_names,
            components=context_keys,
        )

        # Next actor - fixed order for sequential execution
        next_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
        components[next_key] = gm_components.next_acting.NextActingInFixedOrder(
            sequence=player_names,
        )

        # Event resolution
        event_resolution_steps = [thought_chains.identity]

        resolution_key = gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
        components[resolution_key] = gm_components.event_resolution.EventResolution(
            model=model,
            event_resolution_steps=event_resolution_steps,
            components=context_keys,
            notify_observers=True,
        )

        # GM Acting component
        act_component = gm_components.switch_act.SwitchAct(
            model=model,
            entity_names=player_names,
            component_order=list(components.keys()),
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )
