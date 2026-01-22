"""Marketplace agent prefabs.

This module provides specialized agent prefabs for the marketplace scenario:
- BuyerAgent: Seeks to purchase goods within budget
- SellerAgent: Offers goods with pricing strategies
- AuctioneerAgent: Facilitates trades and auctions
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
class BuyerAgent(prefab_lib.Prefab):
    """A buyer agent that seeks to purchase goods within budget constraints.

    Buyers have:
    - A budget they cannot exceed
    - Preferred item categories
    - A purchasing strategy (value_seeker, collector, bargain_hunter)
    """

    description: str = "A buyer agent with budget constraints and purchasing strategies."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Buyer",
            "goal": "Find and purchase desired items within budget",
            "budget": 1000,
            "strategy": "value_seeker",  # value_seeker, collector, bargain_hunter
            "preferred_categories": [],
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the buyer agent."""
        name = self.params.get("name", "Buyer")
        goal = self.params.get("goal", "")
        budget = self.params.get("budget", 1000)
        strategy = self.params.get("strategy", "value_seeker")
        preferred_categories = self.params.get("preferred_categories", [])

        components: dict[str, Any] = {}

        # Memory
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank
        )

        # Instructions
        strategy_descriptions = {
            "value_seeker": "carefully evaluates quality versus price",
            "collector": "seeks rare and unique items regardless of cost",
            "bargain_hunter": "always looks for the best deals and discounts",
        }
        strategy_desc = strategy_descriptions.get(strategy, "seeks good purchases")

        instructions_text = (
            f"You are {name}, a buyer in the marketplace. "
            f"You have a budget of {budget} currency units. "
            f"Your buying style: you {strategy_desc}. "
        )
        if preferred_categories:
            categories_str = ", ".join(preferred_categories)
            instructions_text += f"You are particularly interested in: {categories_str}. "

        instructions_key = "Instructions"
        components[instructions_key] = agent_components.instructions.Instructions(
            agent_name=name,
            pre_act_label="\nRole",
        )

        # Budget tracking
        budget_key = "Budget"
        components[budget_key] = agent_components.constant.Constant(
            state=f"Current budget: {budget} currency units",
            pre_act_label="\nFinancial status",
        )

        # Observations
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=30,
            pre_act_label="\nMarket activity",
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
        components[situation_key] = agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            pre_act_label=f"\nQuestion: What is {name}'s current situation in the marketplace?\nAnswer",
        )

        # Available options
        options_key = "AvailableOptions"
        components[options_key] = agent_components.question_of_recent_memories.AvailableOptionsPerception(
            model=model,
            pre_act_label=f"\nQuestion: What purchasing options are available to {name}?\nAnswer",
        )

        # Best action
        best_action_key = "BestAction"
        components[best_action_key] = agent_components.question_of_recent_memories.BestOptionPerception(
            model=model,
            pre_act_label=f"\nQuestion: What is the best action for {name} given budget constraints?\nAnswer",
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
class SellerAgent(prefab_lib.Prefab):
    """A seller agent that offers goods with various pricing strategies.

    Sellers have:
    - An inventory of items to sell
    - A pricing strategy (premium, competitive, clearance)
    - Sales goals
    """

    description: str = "A seller agent with inventory and pricing strategies."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Seller",
            "goal": "Sell inventory at optimal prices",
            "inventory": [],  # List of {item, category, base_price, quantity}
            "pricing_strategy": "competitive",  # premium, competitive, clearance
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the seller agent."""
        name = self.params.get("name", "Seller")
        goal = self.params.get("goal", "")
        inventory = self.params.get("inventory", [])
        pricing_strategy = self.params.get("pricing_strategy", "competitive")

        components: dict[str, Any] = {}

        # Memory
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank
        )

        # Instructions
        strategy_descriptions = {
            "premium": "prices items at a premium, emphasizing quality and exclusivity",
            "competitive": "matches or slightly undercuts market prices",
            "clearance": "offers aggressive discounts to move inventory quickly",
        }
        strategy_desc = strategy_descriptions.get(pricing_strategy, "uses market-based pricing")

        instructions_key = "Instructions"
        components[instructions_key] = agent_components.instructions.Instructions(
            agent_name=name,
            pre_act_label="\nRole",
        )

        # Inventory status
        inventory_lines = []
        for item in inventory:
            item_name = item.get("item", "Unknown")
            qty = item.get("quantity", 0)
            price = item.get("base_price", 0)
            inventory_lines.append(f"- {item_name}: {qty} units @ {price} base price")

        inventory_text = "\n".join(inventory_lines) if inventory_lines else "No items in stock"

        inventory_key = "Inventory"
        components[inventory_key] = agent_components.constant.Constant(
            state=inventory_text,
            pre_act_label="\nCurrent inventory",
        )

        # Pricing strategy
        pricing_key = "PricingStrategy"
        components[pricing_key] = agent_components.constant.Constant(
            state=f"Pricing approach: {strategy_desc}",
            pre_act_label="\nSales strategy",
        )

        # Observations
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=30,
            pre_act_label="\nMarket activity",
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
        components[situation_key] = agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            pre_act_label=f"\nQuestion: What is {name}'s current situation in the marketplace?\nAnswer",
        )

        # Market analysis
        market_key = "MarketAnalysis"
        components[market_key] = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f"\nQuestion: What is the current market demand for {name}'s products?\nAnswer",
            question=(
                f"Based on recent market activity, what items are buyers interested in? "
                f"How should {name} adjust pricing or approach?"
            ),
            answer_prefix=f"{name} observes that ",
            add_to_memory=False,
            num_memories_to_retrieve=10,
        )

        # Best action
        best_action_key = "BestAction"
        components[best_action_key] = agent_components.question_of_recent_memories.BestOptionPerception(
            model=model,
            pre_act_label=f"\nQuestion: What is the best sales action for {name}?\nAnswer",
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
class AuctioneerAgent(prefab_lib.Prefab):
    """An auctioneer agent that facilitates trades and manages auctions.

    Auctioneers:
    - Announce items for auction
    - Track bids
    - Declare winners
    - Collect commissions
    """

    description: str = "An auctioneer agent that facilitates marketplace trades."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Auctioneer",
            "goal": "Facilitate fair and efficient trades",
            "auction_style": "english",  # english, dutch, sealed
            "commission_rate": 0.05,
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the auctioneer agent."""
        name = self.params.get("name", "Auctioneer")
        goal = self.params.get("goal", "")
        auction_style = self.params.get("auction_style", "english")
        commission_rate = self.params.get("commission_rate", 0.05)

        components: dict[str, Any] = {}

        # Memory
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank
        )

        # Instructions
        style_descriptions = {
            "english": "conducts ascending price auctions where bidders openly compete",
            "dutch": "starts high and lowers prices until a buyer accepts",
            "sealed": "collects sealed bids and awards to highest bidder",
        }
        style_desc = style_descriptions.get(auction_style, "manages auction proceedings")

        instructions_key = "Instructions"
        components[instructions_key] = agent_components.instructions.Instructions(
            agent_name=name,
            pre_act_label="\nRole",
        )

        # Auction rules
        rules_text = (
            f"Auction style: {style_desc}. "
            f"Commission rate: {commission_rate * 100:.1f}% of sale price. "
            f"All sales are final. Buyers must have sufficient funds. "
            f"Sellers must accurately describe items."
        )

        rules_key = "AuctionRules"
        components[rules_key] = agent_components.constant.Constant(
            state=rules_text,
            pre_act_label="\nAuction rules",
        )

        # Observations
        obs_to_memory_key = "ObservationToMemory"
        components[obs_to_memory_key] = agent_components.observation.ObservationToMemory()

        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        components[observation_key] = agent_components.observation.LastNObservations(
            history_length=50,
            pre_act_label="\nAuction proceedings",
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
        components[situation_key] = agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            pre_act_label=f"\nQuestion: What is the current state of the auction?\nAnswer",
        )

        # Bid tracking
        bid_key = "BidAnalysis"
        components[bid_key] = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f"\nQuestion: What are the current bids and who is leading?\nAnswer",
            question=(
                "Review recent bids. What items are being auctioned? "
                "What are the current highest bids? Who are the leading bidders?"
            ),
            answer_prefix="The current auction status is: ",
            add_to_memory=False,
            num_memories_to_retrieve=15,
        )

        # Best action
        best_action_key = "BestAction"
        components[best_action_key] = agent_components.question_of_recent_memories.BestOptionPerception(
            model=model,
            pre_act_label=f"\nQuestion: What should the auctioneer do next?\nAnswer",
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
