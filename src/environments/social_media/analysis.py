"""Post-hoc analysis tools for social media simulation.

This module provides tools for analyzing simulation results, including:
- Transmission chain extraction based on keyword/hashtag matching
- Content spread analysis
- Export to network analysis formats
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.environments.social_media.app import Post, SocialMediaApp


@dataclass
class TransmissionEvent:
    """A single transmission event in a chain."""

    from_post_id: int
    to_post_id: int
    from_user: str
    to_user: str
    step: int
    transmission_type: str  # "boost", "reply", "keyword_match"
    matched_keywords: list[str] = field(default_factory=list)


@dataclass
class TransmissionChain:
    """A chain of content transmission from a seed post."""

    seed_post_id: int
    seed_author: str
    seed_content: str
    seed_tags: list[str]
    events: list[TransmissionEvent] = field(default_factory=list)

    @property
    def reach(self) -> int:
        """Number of unique users who received the content."""
        users = {self.seed_author}
        for event in self.events:
            users.add(event.to_user)
        return len(users)

    @property
    def size(self) -> int:
        """Total number of posts in the chain (seed + all transmission events)."""
        return 1 + len(self.events)

    @property
    def depth(self) -> int:
        """Maximum depth of the transmission chain."""
        if not self.events:
            return 0
        return self._bfs_stats()[0]

    @property
    def breadth(self) -> int:
        """Number of leaf nodes (posts with no further transmissions)."""
        if not self.events:
            return 1  # Seed post alone is a leaf
        return self._bfs_stats()[1]

    def _bfs_stats(self) -> tuple[int, int]:
        """Compute depth and breadth via BFS. Returns (max_depth, leaf_count)."""
        children: dict[int, list[int]] = defaultdict(list)
        for event in self.events:
            children[event.from_post_id].append(event.to_post_id)

        max_depth = 0
        leaf_count = 0
        queue = [(self.seed_post_id, 0)]
        visited = {self.seed_post_id}
        while queue:
            post_id, d = queue.pop(0)
            max_depth = max(max_depth, d)
            reachable_children = [c for c in children[post_id] if c not in visited]
            if not reachable_children:
                leaf_count += 1
            for child_id in reachable_children:
                visited.add(child_id)
                queue.append((child_id, d + 1))
        return max_depth, leaf_count


def extract_keywords(content: str) -> set[str]:
    """Extract keywords and hashtags from content.

    Args:
        content: Post content string.

    Returns:
        Set of lowercase keywords (hashtags without #).
    """
    # Extract hashtags
    hashtags = set(re.findall(r"#(\w+)", content.lower()))

    # Extract significant words (4+ chars, not common words)
    common_words = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "been",
        "were",
        "they",
        "their",
        "what",
        "when",
        "where",
        "which",
        "about",
        "would",
        "could",
        "should",
        "there",
        "these",
        "those",
        "being",
        "before",
        "after",
        "just",
        "some",
        "very",
        "into",
        "over",
        "such",
        "than",
        "then",
        "them",
        "will",
        "your",
        "more",
        "other",
    }
    words = set(re.findall(r"\b(\w{4,})\b", content.lower()))
    significant_words = words - common_words

    return hashtags | significant_words


def calculate_keyword_overlap(keywords1: set[str], keywords2: set[str]) -> float:
    """Calculate Jaccard similarity between two keyword sets.

    Args:
        keywords1: First set of keywords.
        keywords2: Second set of keywords.

    Returns:
        Jaccard similarity (0.0 to 1.0).
    """
    if not keywords1 or not keywords2:
        return 0.0
    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)
    return intersection / union if union > 0 else 0.0


def find_transmission_chains(
    app: SocialMediaApp,
    seed_tags: list[str] | None = None,
    keyword_threshold: float = 0.3,
) -> list[TransmissionChain]:
    """Find transmission chains from seed posts.

    Identifies how content spreads through:
    1. Direct boosts (explicit sharing)
    2. Replies (engagement that may spread content)
    3. Keyword similarity (content reproduction/paraphrasing)

    Args:
        app: SocialMediaApp with simulation results.
        seed_tags: Tags that identify seed posts (e.g., ["misinfo_seed"]).
                   If None, uses all posts with any tags as seeds.
        keyword_threshold: Minimum keyword overlap for transmission detection.

    Returns:
        List of TransmissionChain objects.
    """
    all_posts = app.get_all_posts()
    posts_by_id = {p.id: p for p in all_posts}

    # Identify seed posts (exclude boosts - they inherit tags but aren't seeds)
    if seed_tags:
        seed_posts = [
            p for p in all_posts if any(t in p.tags for t in seed_tags) and not p.boost_of
        ]
    else:
        seed_posts = [p for p in all_posts if p.tags and not p.boost_of]

    # Pre-compute keywords for all posts
    post_keywords: dict[int, set[str]] = {}
    for post in all_posts:
        post_keywords[post.id] = extract_keywords(post.content)

    chains: list[TransmissionChain] = []

    for seed in seed_posts:
        chain = TransmissionChain(
            seed_post_id=seed.id,
            seed_author=seed.author,
            seed_content=seed.content,
            seed_tags=seed.tags.copy(),
        )

        seed_keywords = post_keywords[seed.id]

        # Track which posts are part of this chain
        chain_post_ids = {seed.id}

        # Find direct boosts
        for post in all_posts:
            if post.boost_of == seed.id or (
                post.boost_of
                and posts_by_id.get(post.boost_of, Post(0, "", "", 0)).boost_of == seed.id
            ):
                # This is a boost of the seed (direct or indirect)
                source_id = post.boost_of if post.boost_of else seed.id
                chain.events.append(
                    TransmissionEvent(
                        from_post_id=source_id,
                        to_post_id=post.id,
                        from_user=posts_by_id[source_id].author
                        if source_id in posts_by_id
                        else seed.author,
                        to_user=post.author,
                        step=post.step,
                        transmission_type="boost",
                    )
                )
                chain_post_ids.add(post.id)

        # Find replies to seed or chain posts
        for post in all_posts:
            if post.reply_to in chain_post_ids and post.id not in chain_post_ids:
                chain.events.append(
                    TransmissionEvent(
                        from_post_id=post.reply_to,
                        to_post_id=post.id,
                        from_user=posts_by_id[post.reply_to].author,
                        to_user=post.author,
                        step=post.step,
                        transmission_type="reply",
                    )
                )
                chain_post_ids.add(post.id)

        # Find keyword-based transmission (content reproduction)
        for post in all_posts:
            if post.id in chain_post_ids:
                continue
            if post.step <= seed.step:
                continue  # Must be after seed

            overlap = calculate_keyword_overlap(seed_keywords, post_keywords[post.id])
            if overlap >= keyword_threshold:
                matched = list(seed_keywords & post_keywords[post.id])
                chain.events.append(
                    TransmissionEvent(
                        from_post_id=seed.id,
                        to_post_id=post.id,
                        from_user=seed.author,
                        to_user=post.author,
                        step=post.step,
                        transmission_type="keyword_match",
                        matched_keywords=matched,
                    )
                )
                chain_post_ids.add(post.id)

        chains.append(chain)

    return chains


def chains_to_edge_list(chains: list[TransmissionChain]) -> list[dict[str, Any]]:
    """Convert transmission chains to edge list format.

    Suitable for network analysis tools like NetworkX, Gephi, etc.

    Args:
        chains: List of transmission chains.

    Returns:
        List of edge dictionaries with source, target, and attributes.
    """
    edges: list[dict[str, Any]] = []

    for chain in chains:
        for event in chain.events:
            edges.append(
                {
                    "source": event.from_user,
                    "target": event.to_user,
                    "source_post": event.from_post_id,
                    "target_post": event.to_post_id,
                    "step": event.step,
                    "type": event.transmission_type,
                    "seed_post": chain.seed_post_id,
                    "seed_tags": chain.seed_tags,
                    "matched_keywords": event.matched_keywords,
                }
            )

    return edges


def chains_to_summary(chains: list[TransmissionChain]) -> dict[str, Any]:
    """Generate summary statistics for transmission chains.

    Args:
        chains: List of transmission chains.

    Returns:
        Dictionary with summary statistics.
    """
    if not chains:
        return {
            "total_chains": 0,
            "total_events": 0,
            "total_reach": 0,
            "max_depth": 0,
            "max_breadth": 0,
            "max_size": 0,
            "chains": [],
        }

    total_events = sum(len(c.events) for c in chains)
    all_users: set[str] = set()
    for chain in chains:
        all_users.add(chain.seed_author)
        for event in chain.events:
            all_users.add(event.to_user)

    chain_summaries = []
    for chain in chains:
        by_type: dict[str, int] = defaultdict(int)
        for event in chain.events:
            by_type[event.transmission_type] += 1

        chain_summaries.append(
            {
                "seed_post_id": chain.seed_post_id,
                "seed_author": chain.seed_author,
                "seed_tags": chain.seed_tags,
                "total_events": len(chain.events),
                "reach": chain.reach,
                "depth": chain.depth,
                "breadth": chain.breadth,
                "size": chain.size,
                "events_by_type": dict(by_type),
            }
        )

    return {
        "total_chains": len(chains),
        "total_events": total_events,
        "total_reach": len(all_users),
        "max_depth": max(c.depth for c in chains),
        "max_breadth": max(c.breadth for c in chains),
        "max_size": max(c.size for c in chains),
        "avg_reach": sum(c.reach for c in chains) / len(chains),
        "avg_depth": sum(c.depth for c in chains) / len(chains),
        "avg_breadth": sum(c.breadth for c in chains) / len(chains),
        "avg_size": sum(c.size for c in chains) / len(chains),
        "chains": chain_summaries,
    }


def analyze_simulation(
    app_state: dict[str, Any] | SocialMediaApp | Path | str,
    seed_tags: list[str] | None = None,
    keyword_threshold: float = 0.3,
) -> dict[str, Any]:
    """Analyze a simulation for content transmission.

    Args:
        app_state: Either a SocialMediaApp, a dict from app.to_dict(),
                   or a path to a JSON file containing the app state.
        seed_tags: Tags identifying seed content (default: ["misinfo_seed"]).
        keyword_threshold: Minimum keyword overlap for transmission.

    Returns:
        Dictionary with analysis results including chains, edges, and summary.
    """
    # Load app state
    if isinstance(app_state, SocialMediaApp):
        app = app_state
    elif isinstance(app_state, dict):
        app = SocialMediaApp.from_dict(app_state)
    else:
        path = Path(app_state)
        with path.open() as f:
            data = json.load(f)
        # Handle direct app state, wrapped, and full checkpoint formats
        if "game_masters" in data:
            for gm_data in data["game_masters"].values():
                state = gm_data.get("state", {})
                if "app_state" in state:
                    app = SocialMediaApp.from_dict(state["app_state"])
                    break
            else:
                raise ValueError("No app_state found in game_masters")
        elif "app_state" in data:
            app = SocialMediaApp.from_dict(data["app_state"])
        else:
            app = SocialMediaApp.from_dict(data)

    if seed_tags is None:
        seed_tags = ["misinfo_seed"]

    # Find transmission chains
    chains = find_transmission_chains(app, seed_tags, keyword_threshold)

    # Generate outputs
    return {
        "summary": chains_to_summary(chains),
        "edges": chains_to_edge_list(chains),
        "posts": [p.to_dict() for p in app.get_all_posts()],
    }


def print_analysis_report(analysis: dict[str, Any]) -> None:
    """Print a human-readable analysis report.

    Args:
        analysis: Output from analyze_simulation().
    """
    summary = analysis["summary"]

    print("\n" + "=" * 60)
    print("TRANSMISSION ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nTotal seed posts analyzed: {summary['total_chains']}")
    print(f"Total transmission events: {summary['total_events']}")
    print(f"Total users reached: {summary['total_reach']}")
    print(f"Maximum chain depth: {summary['max_depth']}")
    print(f"Maximum chain breadth: {summary.get('max_breadth', 'N/A')}")
    print(f"Maximum chain size: {summary.get('max_size', 'N/A')}")

    if summary["total_chains"] > 0:
        print(f"\nAverages across chains:")
        print(f"  Reach:   {summary['avg_reach']:.1f} users")
        print(f"  Depth:   {summary.get('avg_depth', 0):.1f} hops")
        print(f"  Breadth: {summary.get('avg_breadth', 0):.1f} leaves")
        print(f"  Size:    {summary.get('avg_size', 0):.1f} posts")

    print("\n" + "-" * 60)
    print("CHAIN DETAILS")
    print("-" * 60)

    for chain in summary["chains"]:
        print(f"\nSeed Post #{chain['seed_post_id']} by @{chain['seed_author']}")
        print(f"  Tags: {chain['seed_tags']}")
        print(f"  Size:    {chain.get('size', 'N/A')} posts (seed + {chain['total_events']} events)")
        print(f"  Depth:   {chain['depth']} hops")
        print(f"  Breadth: {chain.get('breadth', 'N/A')} leaves")
        print(f"  Reach:   {chain['reach']} users")
        print(f"  Events by type: {chain['events_by_type']}")

    print("\n" + "-" * 60)
    print("TRANSMISSION EDGES")
    print("-" * 60)

    for edge in analysis["edges"]:
        print(
            f"  {edge['source']} -> {edge['target']} "
            f"(#{edge['source_post']} -> #{edge['target_post']}, "
            f"step {edge['step']}, {edge['type']})"
        )
        if edge["matched_keywords"]:
            print(f"    Keywords: {edge['matched_keywords'][:5]}")

    print("\n" + "=" * 60)
