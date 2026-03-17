"""Marketplace Game Master prefab.

This module provides the MarketGameMaster that manages the marketplace
simulation, including trade facilitation, event generation, and rule enforcement.
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
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class MarketGameMaster(prefab_lib.Prefab):
    """Game master for the marketplace scenario.

    Manages:
    - Trade validation and execution
    - Market events (new arrivals, price changes, etc.)
    - Rule enforcement
    - Turn order for buyers, sellers, and auctioneer
    """

    description: str = "Game master for marketplace simulation with trade management."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "market_master",
            "market_rules": [
                "All trades must be witnessed and recorded",
                "Buyers must have sufficient funds before purchasing",
                "Sellers must accurately describe their items",
                "The auctioneer's decisions on auction matters are final",
            ],
            "event_frequency": 0.3,  # Probability of random market event
        }
    )

    # Entities are set before build
    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the market game master."""
        name = self.params.get("name", "market_master")
        market_rules = self.params.get("market_rules", [])

        player_names = [e.name for e in self.entities]

        components: dict[str, Any] = {}

        # Memory
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        # GM Instructions
        instructions_key = "instructions"
        components[instructions_key] = gm_components.instructions.Instructions()

        # Market description
        market_desc_key = "market_description"
        market_description = (
            "This is the Grand Marketplace, a bustling trading hub where "
            "buyers and sellers come together to exchange goods. The marketplace "
            "operates through a combination of direct sales and auctions. "
            "An auctioneer facilitates major transactions and ensures fair dealings."
        )
        components[market_desc_key] = agent_components.constant.Constant(
            state=market_description,
            pre_act_label="\nMarketplace setting",
        )

        # Market rules
        if market_rules:
            rules_text = "\n".join(f"- {rule}" for rule in market_rules)
            rules_key = "market_rules"
            components[rules_key] = agent_components.constant.Constant(
                state=rules_text,
                pre_act_label="\nMarket rules",
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
        context_keys = [instructions_key, market_desc_key, players_key]
        if market_rules:
            context_keys.append("market_rules")

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
        event_resolution_steps = [gm_components.event_resolution.identity]

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
