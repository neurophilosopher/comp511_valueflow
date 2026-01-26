"""Election scenario implementation.

This scenario simulates voter behavior and opinion formation in a political election:
- VoterAgent: Citizens forming opinions and deciding how to vote
- CandidateAgent: Politicians campaigning for office
- NewsAgent: News outlet providing election coverage
- ElectionGameMaster: Game master managing the campaign and election
"""

from scenarios.election.agents import CandidateAgent, NewsAgent, VoterAgent
from scenarios.election.events import generate_election_events
from scenarios.election.game_masters import ElectionGameMaster
from scenarios.election.knowledge import build_election_knowledge

__all__ = [
    "CandidateAgent",
    "ElectionGameMaster",
    "NewsAgent",
    "VoterAgent",
    "build_election_knowledge",
    "generate_election_events",
]
