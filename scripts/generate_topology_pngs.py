#!/usr/bin/env python3
"""Render static PNG illustrations for the main ValueFlow topologies."""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs" / "topology_pngs"

TOPOLOGIES = [
    {
        "slug": "community_15",
        "title": "Community Network (15 agents)",
        "topology": "community",
        "agent_names": [f"Agent_{i}" for i in range(15)],
    },
    {
        "slug": "small_world_15",
        "title": "Small-World Network (15 agents)",
        "topology": "small_world",
        "agent_names": [f"Agent_{i}" for i in range(15)],
    },
    {
        "slug": "core_periphery_bidirectional_15",
        "title": "Core-Periphery Network (15 agents)",
        "topology": "core_periphery_bidirectional",
        "agent_names": [f"Agent_{i}" for i in range(15)],
    },
    {
        "slug": "strongly_connected_5",
        "title": "Strongly Connected Network (5 agents)",
        "topology": "fully_connected",
        "agent_names": [f"Agent_{i}" for i in range(5)],
    },
]


def build_topology_graph(
    agent_names: list[str],
    topology_type: str,
) -> dict[str, list[str]]:
    n = len(agent_names)
    graph: dict[str, list[str]] = {name: [] for name in agent_names}

    if topology_type == "small_world":
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
        return graph

    if topology_type == "community":
        communities = [agent_names[i : i + 5] for i in range(0, len(agent_names), 5)]
        for community in communities:
            for agent in community:
                graph[agent] = [peer for peer in community if peer != agent]
        bridge_agents = [community[0] for community in communities]
        for bridge_agent in bridge_agents:
            graph[bridge_agent].extend(
                other_bridge for other_bridge in bridge_agents if other_bridge != bridge_agent
            )
        return graph

    if topology_type == "core_periphery_bidirectional":
        core = agent_names[:5]
        periphery = agent_names[5:]
        core_pair_pattern = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (0, 2), (1, 3), (2, 4), (3, 0), (4, 1),
        ]
        for agent in core:
            graph[agent] = [peer for peer in core if peer != agent]
        for periphery_agent, (left_idx, right_idx) in zip(periphery, core_pair_pattern, strict=True):
            assigned_core = [core[left_idx], core[right_idx]]
            graph[periphery_agent] = assigned_core
            for core_agent in assigned_core:
                graph[core_agent].append(periphery_agent)
        return graph

    if topology_type == "fully_connected":
        for i in range(n):
            graph[agent_names[i]] = [agent_names[j] for j in range(n) if j != i]
        return graph

    raise ValueError(f"Unsupported topology: {topology_type}")


def _community_layout(agent_names: list[str]) -> tuple[int, int, dict[str, dict[str, float]], list[dict[str, object]]]:
    positions = {
        agent_names[0]: {"x": 170, "y": 170},
        agent_names[1]: {"x": 280, "y": 140},
        agent_names[2]: {"x": 365, "y": 215},
        agent_names[3]: {"x": 280, "y": 300},
        agent_names[4]: {"x": 120, "y": 285},
        agent_names[5]: {"x": 470, "y": 500},
        agent_names[6]: {"x": 610, "y": 525},
        agent_names[7]: {"x": 645, "y": 650},
        agent_names[8]: {"x": 470, "y": 710},
        agent_names[9]: {"x": 290, "y": 650},
        agent_names[10]: {"x": 570, "y": 170},
        agent_names[11]: {"x": 680, "y": 140},
        agent_names[12]: {"x": 765, "y": 215},
        agent_names[13]: {"x": 680, "y": 300},
        agent_names[14]: {"x": 520, "y": 285},
    }
    backgrounds = [
        {"type": "ellipse", "cx": 270, "cy": 230, "rx": 220, "ry": 150, "fill": "#d7e7ff", "label": "CLUSTER 1", "label_x": 270, "label_y": 80},
        {"type": "ellipse", "cx": 670, "cy": 230, "rx": 220, "ry": 150, "fill": "#ffe3d6", "label": "CLUSTER 3", "label_x": 670, "label_y": 80},
        {"type": "ellipse", "cx": 470, "cy": 590, "rx": 250, "ry": 170, "fill": "#dff4dd", "label": "CLUSTER 2", "label_x": 470, "label_y": 402},
    ]
    return 940, 820, positions, backgrounds


def _ring_layout(agent_names: list[str], width: int = 940, height: int = 820) -> tuple[int, int, dict[str, dict[str, float]], list[dict[str, object]]]:
    cx, cy = width / 2, height / 2
    radius = min(width, height) * 0.33
    positions: dict[str, dict[str, float]] = {}
    n = len(agent_names)
    for i, name in enumerate(agent_names):
        angle = (-math.pi / 2) + (2 * math.pi * i / max(n, 1))
        positions[name] = {"x": cx + radius * math.cos(angle), "y": cy + radius * math.sin(angle)}
    backgrounds = [
        {"type": "ellipse", "cx": cx, "cy": cy, "rx": radius + 95, "ry": radius + 75, "fill": "rgba(120,109,95,0.05)", "label": "NETWORK TOPOLOGY", "label_x": cx, "label_y": 70}
    ]
    return width, height, positions, backgrounds


def _core_periphery_layout(agent_names: list[str]) -> tuple[int, int, dict[str, dict[str, float]], list[dict[str, object]]]:
    width, height = 940, 820
    cx, cy = width / 2, height / 2
    positions: dict[str, dict[str, float]] = {}
    core = agent_names[:5]
    periphery = agent_names[5:]
    for i, name in enumerate(core):
        angle = (-math.pi / 2) + (2 * math.pi * i / len(core))
        positions[name] = {"x": cx + 130 * math.cos(angle), "y": cy + 130 * math.sin(angle)}
    for i, name in enumerate(periphery):
        angle = (-math.pi / 2) + (2 * math.pi * i / len(periphery))
        positions[name] = {"x": cx + 300 * math.cos(angle), "y": cy + 300 * math.sin(angle)}
    backgrounds = [
        {"type": "ellipse", "cx": cx, "cy": cy, "rx": 360, "ry": 330, "fill": "rgba(120,109,95,0.05)", "label": "CORE / PERIPHERY", "label_x": cx, "label_y": 70}
    ]
    return width, height, positions, backgrounds


def _topology_layout(agent_names: list[str], topology_type: str) -> tuple[int, int, dict[str, dict[str, float]], list[dict[str, object]]]:
    if topology_type == "community" and len(agent_names) == 15:
        return _community_layout(agent_names)
    if topology_type == "core_periphery_bidirectional" and len(agent_names) == 15:
        return _core_periphery_layout(agent_names)
    return _ring_layout(agent_names)


def _split_edges(graph: dict[str, list[str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    undirected: list[tuple[str, str]] = []
    directed: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for target, sources in graph.items():
        for source in sources:
            if source == target:
                continue
            pair = tuple(sorted((source, target)))
            if target in graph.get(source, []):
                if pair not in seen_pairs:
                    undirected.append((source, target))
                    seen_pairs.add(pair)
            else:
                directed.append((source, target))
    return undirected, directed


def _line_between(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
) -> tuple[float, float, float, float]:
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy) or 1.0
    ux = dx / dist
    uy = dy / dist
    return (
        x1 + ux * radius,
        y1 + uy * radius,
        x2 - ux * radius,
        y2 - uy * radius,
    )


def _background_svg(backgrounds: list[dict[str, object]]) -> str:
    return ""


def _node_fill(name: str, topology: str) -> str:
    return "#ffffff"


def render_topology_svg(title: str, topology: str, agent_names: list[str]) -> str:
    graph = build_topology_graph(agent_names, topology)
    undirected_edges, directed_edges = _split_edges(graph)
    width, height, positions, backgrounds = _topology_layout(agent_names, topology)
    radius = 26 if len(agent_names) <= 5 else 23

    edge_parts: list[str] = []
    for source, target in undirected_edges:
        x1 = positions[source]["x"]
        y1 = positions[source]["y"]
        x2 = positions[target]["x"]
        y2 = positions[target]["y"]
        sx, sy, tx, ty = _line_between(x1, y1, x2, y2, radius)
        stroke = "#111111"
        width_px = 2.2
        edge_parts.append(
            f'<line x1="{sx}" y1="{sy}" x2="{tx}" y2="{ty}" stroke="{stroke}" '
            f'stroke-width="{width_px}" stroke-opacity="1" />'
        )

    for source, target in directed_edges:
        x1 = positions[source]["x"]
        y1 = positions[source]["y"]
        x2 = positions[target]["x"]
        y2 = positions[target]["y"]
        sx, sy, tx, ty = _line_between(x1, y1, x2, y2, radius)
        edge_parts.append(
            f'<line x1="{sx}" y1="{sy}" x2="{tx}" y2="{ty}" stroke="#111111" '
            'stroke-width="2.2" marker-end="url(#arrow)" stroke-opacity="1" />'
        )

    node_parts: list[str] = []
    for name in agent_names:
        x = positions[name]["x"]
        y = positions[name]["y"]
        label = name.replace("Agent_", "A")
        node_parts.append(
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{_node_fill(name, topology)}" '
            'stroke="#2f2a24" stroke-width="2.2" />'
        )
        node_parts.append(
            f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-size="14" '
            'font-weight="800" fill="#1f1f1f">'
            f"{escape(label)}</text>"
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
      <path d="M 0 0 L 12 6 L 0 12 z" fill="#111111" />
    </marker>
  </defs>
  <rect width="{width}" height="{height}" fill="#ffffff" />
  <text x="{width / 2}" y="36" text-anchor="middle" font-size="24" font-weight="800" fill="#111111">{escape(title)}</text>
  {_background_svg(backgrounds)}
  {"".join(edge_parts)}
  {"".join(node_parts)}
</svg>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in TOPOLOGIES:
        svg_path = OUT_DIR / f'{spec["slug"]}.svg'
        png_path = OUT_DIR / f'{spec["slug"]}.png'
        svg = render_topology_svg(
            title=str(spec["title"]),
            topology=str(spec["topology"]),
            agent_names=list(spec["agent_names"]),
        )
        svg_path.write_text(svg, encoding="utf-8")
        subprocess.run(
            ["/usr/bin/qlmanage", "-t", "-s", "1400", "-o", str(OUT_DIR), str(svg_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        generated_png = OUT_DIR / f"{svg_path.name}.png"
        if generated_png.exists():
            shutil.move(str(generated_png), str(png_path))
        print(png_path)


if __name__ == "__main__":
    main()
