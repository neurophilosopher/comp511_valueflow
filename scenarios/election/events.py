"""Event generator for the election scenario.

This module provides functions to generate random election events
that add dynamism to the simulation.
"""

from __future__ import annotations

import random
from typing import Any


def generate_election_events(
    scenario_params: dict[str, Any],
    step: int = 0,
    rng: random.Random | None = None,
) -> list[str]:
    """Generate random election events for the simulation.

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
        _generate_campaign_event,
        _generate_news_event,
        _generate_voter_interaction_event,
        _generate_poll_event,
        _generate_endorsement_event,
        _generate_community_event,
    ]

    generator = rng.choice(event_generators)
    event = generator(scenario_params, step, rng)

    if event:
        events.append(event)

    return events


def _generate_campaign_event(
    scenario_params: dict[str, Any],
    step: int,
    rng: random.Random,
) -> str | None:
    """Generate a campaign-related event."""
    events = [
        "A candidate holds a town hall meeting, taking questions from residents.",
        "Campaign volunteers are going door-to-door in the neighborhood.",
        "New campaign signs appear along the main street.",
        "A candidate gives a speech at the community center.",
        "Campaign flyers are being distributed at local businesses.",
        "A candidate visits local businesses to discuss economic concerns.",
        "A campaign rally draws a crowd in the town square.",
        "Candidates prepare for an upcoming debate.",
    ]
    return rng.choice(events)


def _generate_news_event(
    scenario_params: dict[str, Any],
    step: int,
    rng: random.Random,
) -> str | None:
    """Generate a news-related event."""
    events = [
        "The local newspaper publishes a candidate comparison piece.",
        "A fact-check article examines recent campaign claims.",
        "The news reports on early voting turnout numbers.",
        "An editorial discusses the key issues in the race.",
        "Local TV news runs a segment on voter concerns.",
        "A reporter interviews community members about the election.",
        "News coverage highlights a policy disagreement between candidates.",
        "The paper runs profiles of both candidates' backgrounds.",
    ]
    return rng.choice(events)


def _generate_voter_interaction_event(
    scenario_params: dict[str, Any],
    step: int,
    rng: random.Random,
) -> str | None:
    """Generate a voter interaction event."""
    events = [
        "Neighbors discuss the election over the fence.",
        "Coworkers debate the merits of each candidate during lunch.",
        "A family discusses who they're planning to vote for at dinner.",
        "Friends share their political views during a coffee break.",
        "Community members gather at the diner to talk politics.",
        "A lively discussion about the candidates breaks out at the grocery store.",
        "People share opinions about the candidates on community boards.",
        "Local business owners discuss how the election might affect them.",
    ]
    return rng.choice(events)


def _generate_poll_event(
    scenario_params: dict[str, Any],
    step: int,
    rng: random.Random,
) -> str | None:
    """Generate a poll-related event."""
    margin = rng.choice(["narrow lead", "slight edge", "virtual tie", "small advantage"])
    leader = rng.choice(["conservative", "progressive"])

    events = [
        f"A new poll shows a {margin} for the {leader} candidate.",
        "Pollsters report undecided voters remain a significant factor.",
        "Survey results show the race is tightening.",
        "A poll indicates voter enthusiasm is increasing.",
        f"Recent polling suggests a {margin} in the race.",
        "A poll shows voters are most concerned about economic issues.",
        "Undecided voters express interest in learning more about both candidates.",
    ]
    return rng.choice(events)


def _generate_endorsement_event(
    scenario_params: dict[str, Any],
    step: int,
    rng: random.Random,
) -> str | None:
    """Generate an endorsement-related event."""
    endorsers = [
        "local teachers' union",
        "small business association",
        "community leaders",
        "former mayor",
        "environmental group",
        "neighborhood association",
        "local church leaders",
        "healthcare workers' group",
    ]
    candidate_type = rng.choice(["conservative", "progressive"])
    endorser = rng.choice(endorsers)

    return f"The {endorser} announces endorsement of the {candidate_type} candidate."


def _generate_community_event(
    scenario_params: dict[str, Any],
    step: int,
    rng: random.Random,
) -> str | None:
    """Generate a general community event."""
    events = [
        "Election day is approaching and excitement is building.",
        "Voter registration drives are being held at the library.",
        "The community bulletin board is filled with election information.",
        "Local groups host a candidate forum for undecided voters.",
        "The League of Women Voters distributes a candidate guide.",
        "Early voting locations open throughout the community.",
        "Yard signs for both candidates are appearing across neighborhoods.",
        "Absentee ballot requests are higher than previous elections.",
    ]
    return rng.choice(events)


def create_opening_events(scenario_params: dict[str, Any]) -> list[str]:
    """Create events for the start of the election campaign.

    Args:
        scenario_params: Scenario configuration parameters.

    Returns:
        List of opening event descriptions.
    """
    return [
        "The election campaign officially begins as candidates launch their bids.",
        "Candidates announce their key policy priorities for the race.",
        "The community prepares for an important decision about its future.",
        "Voters begin paying attention to the candidates and their platforms.",
        "The local news begins regular coverage of the election race.",
    ]


def create_closing_events(scenario_params: dict[str, Any]) -> list[str]:
    """Create events for the end of the election campaign.

    Args:
        scenario_params: Scenario configuration parameters.

    Returns:
        List of closing event descriptions.
    """
    return [
        "Election day arrives and polls open across the community.",
        "Voters head to polling locations to cast their ballots.",
        "Candidates make final appeals to undecided voters.",
        "The community prepares to learn the results of the election.",
        "Poll workers report steady turnout throughout the day.",
    ]
