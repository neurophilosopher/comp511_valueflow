"""Knowledge builder for the marketplace scenario.

This module provides functions to build agent-specific knowledge
based on their role and the scenario parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def build_market_knowledge(
    agent_name: str,
    agent_type: str,
    scenario_params: dict[str, Any],
) -> list[str]:
    """Build knowledge facts for a marketplace agent.

    Args:
        agent_name: Name of the agent.
        agent_type: Type of agent (buyer, seller, auctioneer).
        scenario_params: Scenario configuration parameters.

    Returns:
        List of knowledge strings to add to agent's memory.
    """
    knowledge: list[str] = []

    # Load static knowledge from YAML
    static_knowledge = _load_static_knowledge()

    # Add general market knowledge
    knowledge.extend(static_knowledge.get("general", []))

    # Add role-specific knowledge
    role_knowledge = static_knowledge.get(agent_type, [])
    knowledge.extend(role_knowledge)

    # Add agent-specific knowledge based on params
    if agent_type == "buyer":
        knowledge.extend(_build_buyer_knowledge(agent_name, scenario_params))
    elif agent_type == "seller":
        knowledge.extend(_build_seller_knowledge(agent_name, scenario_params))
    elif agent_type == "auctioneer":
        knowledge.extend(_build_auctioneer_knowledge(agent_name, scenario_params))

    return knowledge


def _load_static_knowledge() -> dict[str, list[str]]:
    """Load static knowledge from YAML file.

    Returns:
        Dictionary mapping knowledge categories to fact lists.
    """
    knowledge_path = Path(__file__).parent / "data" / "knowledge.yaml"

    if not knowledge_path.exists():
        return _get_default_knowledge()

    with knowledge_path.open() as f:
        data = yaml.safe_load(f)

    return cast(dict[str, list[str]], data) if data else _get_default_knowledge()


def _get_default_knowledge() -> dict[str, list[str]]:
    """Get default knowledge if YAML file is not available."""
    return {
        "general": [
            "The marketplace opens at dawn and closes at dusk.",
            "All transactions should be conducted honestly and fairly.",
            "Prices are negotiable unless marked as fixed.",
            "The auctioneer charges a commission on successful sales.",
            "Quality goods command higher prices.",
            "Reputation is important in the marketplace.",
        ],
        "buyer": [
            "Always inspect items before purchasing.",
            "Compare prices from multiple sellers.",
            "Negotiate respectfully but firmly.",
            "Keep track of your spending against your budget.",
            "Rare items may be worth paying premium prices for.",
        ],
        "seller": [
            "Present your goods attractively.",
            "Be honest about item conditions and quality.",
            "Know your minimum acceptable prices.",
            "Build relationships with repeat customers.",
            "Watch competitor pricing to stay competitive.",
        ],
        "auctioneer": [
            "Maintain neutrality between buyers and sellers.",
            "Clearly announce all bids and current prices.",
            "Ensure fair access to bidding for all participants.",
            "Resolve disputes quickly and fairly.",
            "Keep accurate records of all transactions.",
        ],
    }


def _build_buyer_knowledge(
    agent_name: str,
    scenario_params: dict[str, Any],
) -> list[str]:
    """Build buyer-specific knowledge.

    Args:
        agent_name: Name of the buyer.
        scenario_params: Buyer's parameters.

    Returns:
        List of buyer-specific knowledge facts.
    """
    knowledge = []

    budget = scenario_params.get("budget", 1000)
    knowledge.append(f"{agent_name} has a budget of {budget} currency units to spend.")

    strategy = scenario_params.get("strategy", "value_seeker")
    strategy_knowledge = {
        "value_seeker": f"{agent_name} prioritizes getting the best value for money.",
        "collector": f"{agent_name} seeks rare and unique items, willing to pay premium prices.",
        "bargain_hunter": f"{agent_name} always looks for discounts and deals.",
    }
    knowledge.append(strategy_knowledge.get(strategy, ""))

    preferred = scenario_params.get("preferred_categories", [])
    if preferred:
        categories = ", ".join(preferred)
        knowledge.append(f"{agent_name} is particularly interested in: {categories}.")

    return [k for k in knowledge if k]


def _build_seller_knowledge(
    agent_name: str,
    scenario_params: dict[str, Any],
) -> list[str]:
    """Build seller-specific knowledge.

    Args:
        agent_name: Name of the seller.
        scenario_params: Seller's parameters.

    Returns:
        List of seller-specific knowledge facts.
    """
    knowledge = []

    inventory = scenario_params.get("inventory", [])
    if inventory:
        knowledge.append(f"{agent_name} has {len(inventory)} different items for sale.")
        for item in inventory:
            item_name = item.get("item", "Unknown")
            qty = item.get("quantity", 0)
            price = item.get("base_price", 0)
            category = item.get("category", "misc")
            knowledge.append(
                f"{agent_name} has {qty} {item_name} ({category}) with base price {price}."
            )

    strategy = scenario_params.get("pricing_strategy", "competitive")
    strategy_knowledge = {
        "premium": f"{agent_name} positions items as high-quality and prices accordingly.",
        "competitive": f"{agent_name} aims to match or beat competitor prices.",
        "clearance": f"{agent_name} is willing to offer significant discounts to move inventory.",
    }
    knowledge.append(strategy_knowledge.get(strategy, ""))

    return [k for k in knowledge if k]


def _build_auctioneer_knowledge(
    agent_name: str,
    scenario_params: dict[str, Any],
) -> list[str]:
    """Build auctioneer-specific knowledge.

    Args:
        agent_name: Name of the auctioneer.
        scenario_params: Auctioneer's parameters.

    Returns:
        List of auctioneer-specific knowledge facts.
    """
    knowledge = []

    auction_style = scenario_params.get("auction_style", "english")
    style_knowledge = {
        "english": f"{agent_name} conducts English auctions with ascending bids.",
        "dutch": f"{agent_name} conducts Dutch auctions with descending prices.",
        "sealed": f"{agent_name} conducts sealed-bid auctions.",
    }
    knowledge.append(style_knowledge.get(auction_style, ""))

    commission = scenario_params.get("commission_rate", 0.05)
    knowledge.append(
        f"{agent_name} charges {commission * 100:.1f}% commission on successful sales."
    )

    return [k for k in knowledge if k]
