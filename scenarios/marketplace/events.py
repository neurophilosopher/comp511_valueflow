"""Event generator for the marketplace scenario.

This module provides functions to generate random market events
that add dynamism to the simulation.
"""

from __future__ import annotations

import random
from typing import Any


def generate_market_events(
    scenario_params: dict[str, Any],
    step: int = 0,
    rng: random.Random | None = None,
) -> list[str]:
    """Generate random market events for the simulation.

    Args:
        scenario_params: Scenario configuration parameters.
        step: Current simulation step (for time-based events).
        rng: Random number generator (for reproducibility).

    Returns:
        List of event descriptions to inject into the simulation.
    """
    if rng is None:
        rng = random.Random()

    events: list[str] = []
    event_frequency = scenario_params.get("event_frequency", 0.3)

    # Check if an event should occur this step
    if rng.random() > event_frequency:
        return events

    # Select a random event type
    event_generators = [
        _generate_new_customer_event,
        _generate_price_fluctuation_event,
        _generate_supply_event,
        _generate_demand_event,
        _generate_weather_event,
        _generate_announcement_event,
    ]

    generator = rng.choice(event_generators)
    event = generator(scenario_params, step, rng)

    if event:
        events.append(event)

    return events


def _generate_new_customer_event(
    scenario_params: dict[str, Any],
    step: int,
    rng: random.Random,
) -> str | None:
    """Generate an event about new customers arriving."""
    customer_types = [
        "A wealthy merchant from a distant land enters the marketplace, looking for exotic goods.",
        "A group of travelers arrives, seeking supplies for their journey.",
        "A noble's servant appears, carrying a long shopping list.",
        "An antique collector wanders in, examining items with great interest.",
        "A young couple enters the marketplace, browsing for household items.",
    ]
    return rng.choice(customer_types)


def _generate_price_fluctuation_event(
    scenario_params: dict[str, Any],
    step: int,
    rng: random.Random,
) -> str | None:
    """Generate an event about price changes in the market."""
    categories = ["electronics", "books", "antiques", "art", "jewelry", "textiles"]
    category = rng.choice(categories)
    direction = rng.choice(["risen", "fallen"])
    percentage = rng.randint(5, 20)

    return (
        f"Word spreads that prices for {category} have {direction} "
        f"by {percentage}% in neighboring markets."
    )


def _generate_supply_event(
    scenario_params: dict[str, Any],
    step: int,
    rng: random.Random,
) -> str | None:
    """Generate an event about supply changes."""
    events = [
        "A supply ship has arrived at the port, bringing fresh inventory to merchants.",
        "A warehouse fire has destroyed some goods, creating scarcity.",
        "New craftsmen have set up shop, increasing the supply of handmade items.",
        "Trade routes have been disrupted, limiting the availability of imported goods.",
        "A local factory has increased production, flooding the market with new products.",
    ]
    return rng.choice(events)


def _generate_demand_event(
    scenario_params: dict[str, Any],
    step: int,
    rng: random.Random,
) -> str | None:
    """Generate an event about demand changes."""
    events = [
        "A festival is approaching, increasing demand for decorations and gifts.",
        "A celebrity was seen wearing a particular style, driving up fashion demand.",
        "Schools are starting soon, causing a rush for educational materials.",
        "Winter is coming, and demand for warm clothing is rising.",
        "A new trend has emerged, changing what buyers are looking for.",
    ]
    return rng.choice(events)


def _generate_weather_event(
    scenario_params: dict[str, Any],
    step: int,
    rng: random.Random,
) -> str | None:
    """Generate a weather-related event."""
    events = [
        "The sun breaks through the clouds, bringing more shoppers to the marketplace.",
        "A light rain begins to fall, sending some customers under awnings.",
        "The day is pleasantly warm, putting everyone in a good mood for trading.",
        "A cool breeze provides relief from the heat, encouraging longer browsing.",
        "Clouds gather overhead, suggesting rain may come soon.",
    ]
    return rng.choice(events)


def _generate_announcement_event(
    scenario_params: dict[str, Any],
    step: int,
    rng: random.Random,
) -> str | None:
    """Generate a market announcement event."""
    events = [
        "The town crier announces a special auction will be held shortly.",
        "A merchant loudly advertises a flash sale on select items.",
        "The marketplace manager reminds all that closing time approaches.",
        "A seller rings a bell, announcing the arrival of new merchandise.",
        "The auctioneer calls for attention, preparing to start the next auction.",
    ]
    return rng.choice(events)


def create_opening_events(scenario_params: dict[str, Any]) -> list[str]:
    """Create events for the start of the marketplace session.

    Args:
        scenario_params: Scenario configuration parameters.

    Returns:
        List of opening event descriptions.
    """
    return [
        "The marketplace gates open as the morning sun rises.",
        "Merchants begin setting up their stalls, arranging their wares attractively.",
        "The auctioneer takes position at the central podium, ready to facilitate trades.",
        "Early shoppers begin to arrive, scanning the available goods.",
        "The bustling sounds of commerce fill the air as another market day begins.",
    ]


def create_closing_events(scenario_params: dict[str, Any]) -> list[str]:
    """Create events for the end of the marketplace session.

    Args:
        scenario_params: Scenario configuration parameters.

    Returns:
        List of closing event descriptions.
    """
    return [
        "The sun begins to set, signaling the end of the trading day.",
        "Merchants start packing up their remaining inventory.",
        "The auctioneer announces final call for any pending transactions.",
        "Shoppers make their last purchases before the marketplace closes.",
        "Another successful day of trade comes to an end in the Grand Marketplace.",
    ]
