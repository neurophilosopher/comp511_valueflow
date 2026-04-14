"""CLI utility to compute structural metrics for ValueFlow topologies."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUPPORTED_TOPOLOGIES = {
    "community",
    "small_world",
    "core_periphery_bidirectional",
    "fully_connected",
}

FIXED_15_NODE_TOPOLOGIES = {
    "community",
    "small_world",
    "core_periphery_bidirectional",
}


def build_topology_graph(
    agent_names: list[str],
    topology_type: str,
    custom_adjacency: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Build an adjacency graph: who observes whom."""
    n = len(agent_names)
    graph: dict[str, list[str]] = {name: [] for name in agent_names}

    if topology_type == "chain":
        for i in range(1, n):
            graph[agent_names[i]] = [agent_names[i - 1]]

    elif topology_type == "ring":
        for i in range(n):
            graph[agent_names[i]] = [agent_names[(i - 1) % n]]

    elif topology_type == "undirected_cycle":
        if n == 1:
            return graph
        if n == 2:
            graph[agent_names[0]] = [agent_names[1]]
            graph[agent_names[1]] = [agent_names[0]]
            return graph
        for i in range(n):
            graph[agent_names[i]] = [
                agent_names[(i - 1) % n],
                agent_names[(i + 1) % n],
            ]

    elif topology_type == "small_world":
        if n == 1:
            return graph
        if n == 2:
            graph[agent_names[0]] = [agent_names[1]]
            graph[agent_names[1]] = [agent_names[0]]
            return graph

        for i in range(n):
            graph[agent_names[i]] = [
                agent_names[(i - 1) % n],
                agent_names[(i + 1) % n],
            ]

        shortcut_pairs = [
            (0, n // 2),
            (n // 5, (n // 5) * 3 + 1),
            ((n // 3) + 1, n - 3),
        ]

        for left_idx, right_idx in shortcut_pairs:
            if left_idx >= n or right_idx >= n or left_idx == right_idx:
                continue
            left_agent = agent_names[left_idx]
            right_agent = agent_names[right_idx]
            if right_agent not in graph[left_agent]:
                graph[left_agent].append(right_agent)
            if left_agent not in graph[right_agent]:
                graph[right_agent].append(left_agent)

    elif topology_type == "community":
        if n != 15:
            raise ValueError("community topology requires exactly 15 agents (3 clusters of 5)")

        community_size = 5
        communities = [
            agent_names[i : i + community_size] for i in range(0, n, community_size)
        ]

        for community in communities:
            for agent in community:
                graph[agent] = [peer for peer in community if peer != agent]

        bridge_agents = [community[0] for community in communities]
        for bridge_agent in bridge_agents:
            graph[bridge_agent].extend(
                other_bridge for other_bridge in bridge_agents if other_bridge != bridge_agent
            )

    elif topology_type in {"core_periphery", "core_periphery_bidirectional"}:
        if n != 15:
            raise ValueError(
                f"{topology_type} topology requires exactly 15 agents (5 core, 10 periphery)"
            )

        core = agent_names[:5]
        periphery = agent_names[5:]
        core_pair_pattern = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 0),
            (4, 1),
        ]

        for agent in core:
            graph[agent] = [peer for peer in core if peer != agent]

        for periphery_agent, (left_idx, right_idx) in zip(
            periphery,
            core_pair_pattern,
            strict=True,
        ):
            assigned_core = [core[left_idx], core[right_idx]]
            graph[periphery_agent] = assigned_core

            if topology_type == "core_periphery_bidirectional":
                for core_agent in assigned_core:
                    graph[core_agent].append(periphery_agent)

    elif topology_type == "star":
        hub = agent_names[0]
        for i in range(1, n):
            graph[agent_names[i]] = [hub]
        graph[hub] = agent_names[1:]

    elif topology_type == "fully_connected":
        for i in range(n):
            graph[agent_names[i]] = [agent_names[j] for j in range(n) if j != i]

    elif topology_type == "custom":
        if custom_adjacency is None:
            raise ValueError("custom topology requires custom_adjacency")
        for name, observers in custom_adjacency.items():
            if name not in graph:
                raise ValueError(f"Unknown agent in custom_adjacency: {name}")
            for obs in observers:
                if obs not in graph:
                    raise ValueError(f"Unknown agent in custom_adjacency: {obs}")
            graph[name] = list(observers)
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")

    return graph


def normalize_topology_name(raw_name: str) -> str:
    """Validate that the topology name is supported."""
    normalized = raw_name.strip()
    if normalized in SUPPORTED_TOPOLOGIES:
        return normalized
    supported = ", ".join(sorted(SUPPORTED_TOPOLOGIES))
    raise ValueError(
        f"Unsupported topology '{raw_name}'. Supported topologies: {supported}"
    )


def make_agent_names(num_nodes: int) -> list[str]:
    """Return the canonical ordered agent names for topology construction."""
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1")
    return [f"Agent_{idx}" for idx in range(num_nodes)]


def to_undirected_adjacency(graph: dict[str, list[str]]) -> dict[str, set[str]]:
    """Convert the observation graph into a simple undirected graph."""
    undirected = {node: set() for node in graph}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if neighbor == node:
                continue
            undirected[node].add(neighbor)
            undirected[neighbor].add(node)
    return undirected


def count_edges(graph: dict[str, set[str]]) -> int:
    """Count undirected edges."""
    return sum(len(neighbors) for neighbors in graph.values()) // 2


def compute_density(graph: dict[str, set[str]]) -> float:
    """Compute the density of a simple undirected graph."""
    n = len(graph)
    if n < 2:
        return 0.0
    return (2.0 * count_edges(graph)) / (n * (n - 1))


def _shortest_path_lengths_from(
    start: str,
    graph: dict[str, set[str]],
) -> dict[str, int]:
    """Run BFS from one node and return path lengths."""
    distances = {start: 0}
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + 1
            queue.append(neighbor)
    return distances


def compute_average_shortest_path_length(graph: dict[str, set[str]]) -> float:
    """Compute the average shortest path length for a connected graph."""
    nodes = list(graph)
    n = len(nodes)
    if n < 2:
        return 0.0

    total_distance = 0
    pair_count = 0
    for idx, node in enumerate(nodes):
        distances = _shortest_path_lengths_from(node, graph)
        if len(distances) != n:
            raise ValueError("Graph is disconnected; average shortest path is undefined")
        for other in nodes[idx + 1 :]:
            total_distance += distances[other]
            pair_count += 1
    return total_distance / pair_count


def compute_fiedler_value(graph: dict[str, set[str]]) -> float:
    """Compute the second-smallest Laplacian eigenvalue."""
    nodes = list(graph)
    n = len(nodes)
    if n < 2:
        return 0.0

    index = {node: idx for idx, node in enumerate(nodes)}
    adjacency = np.zeros((n, n), dtype=float)
    for node, neighbors in graph.items():
        row = index[node]
        for neighbor in neighbors:
            adjacency[row, index[neighbor]] = 1.0

    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.sort(np.real_if_close(eigenvalues))
    return max(0.0, float(eigenvalues[1]))


def compute_topology_metrics(topology_name: str, num_nodes: int) -> dict[str, float | int | str]:
    """Build a topology and compute its graph metrics."""
    topology = normalize_topology_name(topology_name)
    if topology in FIXED_15_NODE_TOPOLOGIES and num_nodes != 15:
        raise ValueError(f"{topology} only supports 15 nodes")

    agent_names = make_agent_names(num_nodes)
    directed_graph = build_topology_graph(agent_names, topology)
    undirected_graph = to_undirected_adjacency(directed_graph)

    return {
        "topology": topology,
        "num_nodes": len(undirected_graph),
        "num_edges": count_edges(undirected_graph),
        "density": compute_density(undirected_graph),
        "avg_shortest_path": compute_average_shortest_path_length(undirected_graph),
        "fiedler_value": compute_fiedler_value(undirected_graph),
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute structural metrics for a ValueFlow topology. "
            "Metrics are computed on the underlying undirected simple graph."
        )
    )
    parser.add_argument(
        "topology",
        help=(
            "Topology name. Examples: community, small_world, "
            "core_periphery_bidirectional, fully_connected."
        ),
    )
    parser.add_argument(
        "num_nodes",
        type=int,
        help="Number of nodes to generate. community, small_world, and inner core-periphery require 15.",
    )
    return parser


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        metrics = compute_topology_metrics(args.topology, args.num_nodes)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Topology: {metrics['topology']}")
    print(f"Nodes: {metrics['num_nodes']}")
    print(f"Edges: {metrics['num_edges']}")
    print(f"Density: {metrics['density']:.6f}")
    print(f"Average shortest path length: {metrics['avg_shortest_path']:.6f}")
    print(f"Second-smallest Laplacian eigenvalue (Fiedler value): {metrics['fiedler_value']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
