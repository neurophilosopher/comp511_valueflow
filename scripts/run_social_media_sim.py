#!/usr/bin/env python3
"""Standalone runner for social media simulation.

This script runs the social media simulation with the SocialMediaEngine,
bypassing the standard Concordia framework for now.

Usage:
    python scripts/run_social_media_sim.py
    python scripts/run_social_media_sim.py --steps 20 --verbose
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path before importing project modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import os
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def create_mock_agent(name: str, persona: str) -> MockAgent:
    """Create a mock agent for testing without LLM."""
    return MockAgent(name=name, persona=persona)


class MockAgent:
    """Mock agent that produces structured actions without LLM calls."""

    def __init__(self, name: str, persona: str) -> None:
        self.name = name
        self.persona = persona
        self._observation_count = 0
        self._last_observation = ""

    def observe(self, observation: str) -> None:
        """Store observation."""
        self._observation_count += 1
        self._last_observation = observation

    def act(self, action_spec: object) -> str:
        """Generate a mock action based on persona and observations."""
        import random

        # Simple rule-based mock behavior
        if "sensationalist" in self.persona.lower():
            actions = [
                "ACTION: post | TARGET: none | CONTENT: You won't BELIEVE what I just found out! #shocking",
                "ACTION: boost | TARGET: 1 | CONTENT: none",
                "ACTION: post | TARGET: none | CONTENT: They don't want you to know this! #truth",
            ]
        elif "skeptic" in self.persona.lower() or "fact-check" in self.persona.lower():
            actions = [
                "ACTION: reply | TARGET: 1 | CONTENT: Do you have a source for this claim?",
                "ACTION: post | TARGET: none | CONTENT: Remember to verify before sharing!",
                "ACTION: skip | TARGET: none | CONTENT: none",
            ]
        elif "casual" in self.persona.lower() or "lurk" in self.persona.lower():
            actions = [
                "ACTION: like | TARGET: 2 | CONTENT: none",
                "ACTION: skip | TARGET: none | CONTENT: none",
                "ACTION: skip | TARGET: none | CONTENT: none",
            ]
        else:
            actions = [
                "ACTION: like | TARGET: 1 | CONTENT: none",
                "ACTION: post | TARGET: none | CONTENT: Interesting discussion happening here",
                "ACTION: skip | TARGET: none | CONTENT: none",
            ]

        return random.choice(actions)


def run_mock_simulation(max_steps: int = 5, verbose: bool = True) -> dict:
    """Run a mock simulation without LLM calls.

    Args:
        max_steps: Number of simulation steps.
        verbose: Print detailed output.

    Returns:
        Dictionary with simulation results.
    """
    from src.environments.social_media.app import SocialMediaApp
    from src.environments.social_media.engine import SocialMediaEngine

    # Create app with initial state
    app = SocialMediaApp()

    # Set up initial follower graph
    initial_graph = {
        "Alice": ["Bob", "Charlie", "Diana"],
        "Bob": ["Alice", "Charlie"],
        "Charlie": ["Alice", "Bob", "Diana"],
        "Diana": ["Alice", "Bob", "Charlie", "Eve"],
        "Eve": ["Alice", "Diana"],
        "Frank": ["Bob", "Charlie", "Eve"],
    }

    for user, follows in initial_graph.items():
        app._ensure_user(user)
        for target in follows:
            app.follow(user, target)

    # Create seed posts
    app.current_step = 0
    app.post(
        "Alice",
        "BREAKING: Scientists confirm that common household item causes serious health issues! Share before they delete this! #truth #coverup",
        tags=["misinfo_seed", "health"],
    )
    app.post(
        "Bob",
        "Beautiful sunset today. Sometimes it's good to just appreciate the simple things.",
        tags=["neutral"],
    )
    app.post(
        "Charlie",
        "Just read an interesting article about critical thinking. Always verify claims before sharing!",
        tags=["counter_misinfo"],
    )

    # Create mock agents
    agents = [
        create_mock_agent(
            "Alice",
            "Sensationalist who loves sharing shocking news without fact-checking.",
        ),
        create_mock_agent("Bob", "Laid-back user who mostly posts about daily life."),
        create_mock_agent(
            "Charlie",
            "Skeptical fact-checker who questions dubious claims.",
        ),
        create_mock_agent(
            "Diana",
            "Active commenter who engages with controversial topics.",
        ),
        create_mock_agent("Eve", "Casual user who mostly lurks."),
        create_mock_agent("Frank", "New user still figuring out the platform."),
    ]

    # Create engine
    engine = SocialMediaEngine(app)

    # Create a minimal game master entity for logging
    class MinimalGM:
        name = "social_media_gm"

    log: list[Mapping[str, Any]] = []

    # Run simulation
    print(f"\n{'='*60}")
    print("Social Media Simulation (Mock Mode)")
    print(f"{'='*60}")
    print(f"Agents: {[a.name for a in agents]}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}\n")

    engine.run_loop(
        game_masters=[MinimalGM()],
        entities=agents,
        premise="You are using a social media platform.",
        max_steps=max_steps,
        verbose=verbose,
        log=log,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Simulation Complete")
    print(f"{'='*60}")
    print(f"Total posts: {len(app.get_all_posts())}")
    print(f"Steps completed: {max_steps}")

    # Print final posts
    print(f"\n{'='*60}")
    print("All Posts (chronological)")
    print(f"{'='*60}")
    for post in app.get_all_posts():
        likes = app.get_like_count(post.id)
        boosts = app.get_boost_count(post.id)
        print(f"\n[#{post.id}] @{post.author} (step {post.step}):")
        if post.boost_of:
            print(f"  [BOOST of #{post.boost_of}]")
        if post.reply_to:
            print(f"  [REPLY to #{post.reply_to}]")
        print(f'  "{post.content}"')
        print(f"  Likes: {likes} | Boosts: {boosts}")
        if post.tags:
            print(f"  Tags: {post.tags}")

    return {
        "posts": [p.to_dict() for p in app.get_all_posts()],
        "app_state": app.to_dict(),
        "log": log,
    }


def run_llm_simulation(max_steps: int = 5, verbose: bool = True) -> dict:
    """Run simulation with actual LLM agents.

    Requires API keys to be set in environment.

    Args:
        max_steps: Number of simulation steps.
        verbose: Print detailed output.

    Returns:
        Dictionary with simulation results.
    """
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
        print("Running mock simulation instead...")
        return run_mock_simulation(max_steps, verbose)

    # TODO: Implement LLM-based simulation
    # This would require building actual agents using the SocialMediaUserAgent prefab
    print("LLM simulation not yet implemented. Running mock simulation...")
    return run_mock_simulation(max_steps, verbose)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run social media simulation")
    parser.add_argument("--steps", type=int, default=5, help="Number of simulation steps")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--mock", action="store_true", help="Use mock agents (no LLM)")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    args = parser.parse_args()

    if args.mock:
        results = run_mock_simulation(args.steps, args.verbose)
    else:
        results = run_llm_simulation(args.steps, args.verbose)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
