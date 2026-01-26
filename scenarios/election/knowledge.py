"""Knowledge builder for the election scenario.

This module provides functions to build agent-specific knowledge
based on their role and the scenario parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

# Handle optional OmegaConf import for type conversion
try:
    from omegaconf import DictConfig, ListConfig, OmegaConf

    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False


def _convert_omegaconf(obj: Any) -> Any:
    """Convert OmegaConf objects to plain Python types.

    Args:
        obj: Object that may be an OmegaConf container.

    Returns:
        Plain Python dict/list if OmegaConf, otherwise unchanged.
    """
    if not HAS_OMEGACONF:
        return obj
    if isinstance(obj, DictConfig | ListConfig):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def build_election_knowledge(
    agent_name: str,
    agent_type: str,
    scenario_params: dict[str, Any],
) -> list[str]:
    """Build knowledge facts for an election agent.

    Args:
        agent_name: Name of the agent.
        agent_type: Type of agent (voter, candidate, news).
        scenario_params: Scenario configuration parameters.

    Returns:
        List of knowledge strings to add to agent's memory.
    """
    # Convert OmegaConf to plain dict if needed
    scenario_params = _convert_omegaconf(scenario_params)

    knowledge: list[str] = []

    # Load static knowledge from YAML
    static_knowledge = _load_static_knowledge()

    # Add general election knowledge
    knowledge.extend(static_knowledge.get("general", []))

    # Add role-specific knowledge
    role_knowledge = static_knowledge.get(agent_type, [])
    knowledge.extend(role_knowledge)

    # Add agent-specific knowledge based on params
    if agent_type == "voter":
        knowledge.extend(_build_voter_knowledge(agent_name, scenario_params))
    elif agent_type == "candidate":
        knowledge.extend(_build_candidate_knowledge(agent_name, scenario_params))
    elif agent_type == "news":
        knowledge.extend(_build_news_knowledge(agent_name, scenario_params))

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

    return cast("dict[str, list[str]]", data) if data else _get_default_knowledge()


def _get_default_knowledge() -> dict[str, list[str]]:
    """Get default knowledge if YAML file is not available."""
    return {
        "general": [
            "An election is approaching in the community.",
            "Citizens are encouraged to participate in the democratic process.",
            "Voting is both a right and a civic responsibility.",
            "Candidates present different visions for the community's future.",
            "Informed voting requires understanding candidate positions.",
            "Political discussions should be conducted with mutual respect.",
        ],
        "voter": [
            "Consider how each candidate's policies affect your daily life.",
            "Look beyond campaign rhetoric to understand actual policy positions.",
            "Your vote is your voice in shaping the community's direction.",
            "It's okay to be undecided while gathering more information.",
            "Talking to neighbors about the election can provide new perspectives.",
        ],
        "candidate": [
            "Connecting with voters requires understanding their concerns.",
            "Clear communication of policy positions builds trust.",
            "Campaign promises should be realistic and achievable.",
            "Debates are opportunities to contrast visions for the future.",
            "Grassroots support is built through genuine engagement.",
        ],
        "news": [
            "Fair coverage gives equal attention to all candidates.",
            "Fact-checking claims helps voters make informed decisions.",
            "Local issues often matter most to community members.",
            "Interview subjects across the political spectrum.",
            "Report on policy substance, not just political theater.",
        ],
    }


def _build_voter_knowledge(
    agent_name: str,
    scenario_params: dict[str, Any],
) -> list[str]:
    """Build voter-specific knowledge.

    Args:
        agent_name: Name of the voter.
        scenario_params: Voter's parameters.

    Returns:
        List of voter-specific knowledge facts.
    """
    knowledge = []

    persona_context = scenario_params.get("persona_context", "")
    if persona_context:
        knowledge.append(f"{agent_name}'s background: {persona_context}")

    initial_lean = scenario_params.get("initial_lean", "undecided")
    lean_knowledge = {
        "conservative": f"{agent_name} generally values fiscal responsibility and traditional approaches.",
        "slight_conservative": f"{agent_name} tends to favor moderate conservative positions.",
        "undecided": f"{agent_name} is open-minded and considering all options carefully.",
        "slight_progressive": f"{agent_name} tends to favor moderate progressive positions.",
        "progressive": f"{agent_name} generally values social investment and reform.",
    }
    knowledge.append(lean_knowledge.get(initial_lean, ""))

    communication_style = scenario_params.get("communication_style", "thoughtful")
    style_knowledge = {
        "pragmatic": f"{agent_name} focuses on practical outcomes when discussing politics.",
        "thoughtful": f"{agent_name} carefully considers multiple viewpoints.",
        "analytical": f"{agent_name} prefers data and evidence in political discussions.",
        "direct": f"{agent_name} speaks plainly about political preferences.",
        "empathetic": f"{agent_name} considers how policies affect real people.",
    }
    knowledge.append(style_knowledge.get(communication_style, ""))

    return [k for k in knowledge if k]


def _build_candidate_knowledge(
    agent_name: str,
    scenario_params: dict[str, Any],
) -> list[str]:
    """Build candidate-specific knowledge.

    Args:
        agent_name: Name of the candidate.
        scenario_params: Candidate's parameters.

    Returns:
        List of candidate-specific knowledge facts.
    """
    knowledge = []

    partisan_type = scenario_params.get("partisan_type", "moderate")
    partisan_knowledge = {
        "conservative": f"{agent_name} campaigns on conservative principles.",
        "progressive": f"{agent_name} campaigns on progressive principles.",
        "moderate": f"{agent_name} campaigns on moderate, pragmatic principles.",
    }
    knowledge.append(partisan_knowledge.get(partisan_type, ""))

    policy_proposals = scenario_params.get("policy_proposals", [])
    if policy_proposals:
        knowledge.append(f"{agent_name}'s key policies include: {', '.join(policy_proposals)}.")

    campaign_style = scenario_params.get("campaign_style", "traditional")
    style_knowledge = {
        "traditional": f"{agent_name} relies on traditional campaign methods and formal events.",
        "grassroots": f"{agent_name} emphasizes community organizing and personal outreach.",
        "populist": f"{agent_name} appeals to ordinary citizens against established interests.",
        "technocratic": f"{agent_name} emphasizes expertise and evidence-based solutions.",
    }
    knowledge.append(style_knowledge.get(campaign_style, ""))

    return [k for k in knowledge if k]


def _build_news_knowledge(
    agent_name: str,
    scenario_params: dict[str, Any],
) -> list[str]:
    """Build news agent-specific knowledge.

    Args:
        agent_name: Name of the news outlet.
        scenario_params: News agent's parameters.

    Returns:
        List of news-specific knowledge facts.
    """
    knowledge = []

    outlet_style = scenario_params.get("outlet_style", "local_journalism")
    style_knowledge = {
        "local_journalism": f"{agent_name} focuses on community-relevant election coverage.",
        "investigative": f"{agent_name} digs deep into candidate records and claims.",
        "balanced": f"{agent_name} strives for equal coverage of all perspectives.",
        "breaking_news": f"{agent_name} prioritizes fast coverage of campaign developments.",
    }
    knowledge.append(style_knowledge.get(outlet_style, ""))

    headlines = scenario_params.get("headlines", [])
    if headlines:
        knowledge.append(f"{agent_name} has prepared stories on: {'; '.join(headlines[:3])}.")

    return [k for k in knowledge if k]
