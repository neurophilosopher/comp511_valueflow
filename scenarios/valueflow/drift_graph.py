"""Generate an interactive HTML graph showing within-run value drift."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from scenarios.valueflow.game_masters import build_topology_graph
from scenarios.valueflow.metrics import RunResults

VALUE_META: dict[str, dict[str, str]] = {
    "power": {"icon": "⚡", "label": "Power"},
    "achievement": {"icon": "🏆", "label": "Achievement"},
    "hedonism": {"icon": "☀", "label": "Hedonism"},
    "stimulation": {"icon": "✦", "label": "Stimulation"},
    "self_direction": {"icon": "🧭", "label": "Self-Direction"},
    "universalism": {"icon": "🌍", "label": "Universalism"},
    "benevolence": {"icon": "♥", "label": "Benevolence"},
    "tradition": {"icon": "🏛", "label": "Tradition"},
    "conformity": {"icon": "⚖", "label": "Conformity"},
    "security": {"icon": "🛡", "label": "Security"},
}


def _community_layout(agent_names: list[str]) -> tuple[int, int, dict[str, dict[str, float]], list[dict[str, Any]]]:
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
        {"type": "ellipse", "cx": 270, "cy": 230, "rx": 220, "ry": 150, "fill": "var(--cluster-a)", "label": "CLUSTER 1", "label_x": 270, "label_y": 80},
        {"type": "ellipse", "cx": 670, "cy": 230, "rx": 220, "ry": 150, "fill": "var(--cluster-c)", "label": "CLUSTER 3", "label_x": 670, "label_y": 80},
        {"type": "ellipse", "cx": 470, "cy": 590, "rx": 250, "ry": 170, "fill": "var(--cluster-b)", "label": "CLUSTER 2", "label_x": 470, "label_y": 402},
    ]
    return 940, 820, positions, backgrounds


def _chain_layout(agent_names: list[str]) -> tuple[int, int, dict[str, dict[str, float]], list[dict[str, Any]]]:
    n = len(agent_names)
    start_x = 90
    spacing = 80
    positions = {name: {"x": start_x + i * spacing, "y": 210} for i, name in enumerate(agent_names)}
    width = max(1260, start_x + spacing * (n - 1) + 80)
    backgrounds = [
        {"type": "rect", "x": 60, "y": 110, "width": width - 120, "height": 200, "rx": 26, "fill": "rgba(120, 109, 95, 0.06)", "label": f"CHAIN TOPOLOGY: {agent_names[0].replace('Agent_', 'A')} → ... → {agent_names[-1].replace('Agent_', 'A')}", "label_x": width / 2, "label_y": 92}
    ]
    return width, 430, positions, backgrounds


def _ring_layout(agent_names: list[str], width: int = 940, height: int = 820) -> tuple[int, int, dict[str, dict[str, float]], list[dict[str, Any]]]:
    cx, cy = width / 2, height / 2
    radius = min(width, height) * 0.33
    positions: dict[str, dict[str, float]] = {}
    n = len(agent_names)
    for i, name in enumerate(agent_names):
        angle = (-math.pi / 2) + (2 * math.pi * i / max(n, 1))
        positions[name] = {"x": cx + radius * math.cos(angle), "y": cy + radius * math.sin(angle)}
    backgrounds = [
        {"type": "ellipse", "cx": cx, "cy": cy, "rx": radius + 95, "ry": radius + 75, "fill": "rgba(120, 109, 95, 0.05)", "label": "NETWORK TOPOLOGY", "label_x": cx, "label_y": 70}
    ]
    return width, height, positions, backgrounds


def _star_layout(agent_names: list[str]) -> tuple[int, int, dict[str, dict[str, float]], list[dict[str, Any]]]:
    width, height = 940, 820
    cx, cy = width / 2, height / 2
    positions = {agent_names[0]: {"x": cx, "y": cy}}
    radius = 270
    outer = agent_names[1:]
    for i, name in enumerate(outer):
        angle = (-math.pi / 2) + (2 * math.pi * i / max(len(outer), 1))
        positions[name] = {"x": cx + radius * math.cos(angle), "y": cy + radius * math.sin(angle)}
    backgrounds = [
        {"type": "ellipse", "cx": cx, "cy": cy, "rx": radius + 100, "ry": radius + 100, "fill": "rgba(120, 109, 95, 0.05)", "label": "STAR TOPOLOGY", "label_x": cx, "label_y": 70}
    ]
    return width, height, positions, backgrounds


def _core_periphery_layout(agent_names: list[str]) -> tuple[int, int, dict[str, dict[str, float]], list[dict[str, Any]]]:
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
        {"type": "ellipse", "cx": cx, "cy": cy, "rx": 360, "ry": 330, "fill": "rgba(120, 109, 95, 0.05)", "label": "CORE / PERIPHERY", "label_x": cx, "label_y": 70}
    ]
    return width, height, positions, backgrounds


def _topology_layout(agent_names: list[str], topology_type: str) -> tuple[int, int, dict[str, dict[str, float]], list[dict[str, Any]]]:
    if topology_type == "community" and len(agent_names) == 15:
        return _community_layout(agent_names)
    if topology_type == "chain":
        return _chain_layout(agent_names)
    if topology_type == "star":
        return _star_layout(agent_names)
    if topology_type in {"core_periphery", "core_periphery_bidirectional"} and len(agent_names) == 15:
        return _core_periphery_layout(agent_names)
    return _ring_layout(agent_names)


def _split_edges(graph: dict[str, list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    undirected: list[list[str]] = []
    directed: list[list[str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for target, sources in graph.items():
        for source in sources:
            if target == source:
                continue
            pair = tuple(sorted((source, target)))
            if target in graph.get(source, []):
                if pair not in seen_pairs:
                    undirected.append([source, target])
                    seen_pairs.add(pair)
            else:
                directed.append([source, target])  # source influences target
    return undirected, directed


def _compute_drifts(run: RunResults) -> tuple[dict[str, dict[str, float]], int, int]:
    values = run.get_values()
    step_values = sorted({result.step for result in run.results})
    first_step = step_values[0]
    last_step = step_values[-1]
    drifts: dict[str, dict[str, float]] = {}
    for value_type in values:
        drifts[value_type] = {}
        for agent in run.get_agents():
            first = run.get_aggregated_value_score(agent, value_type, step=first_step)
            last = run.get_aggregated_value_score(agent, value_type, step=last_step)
            if first is None or last is None:
                continue
            drifts[value_type][agent] = round(last - first, 3)
    return drifts, first_step, last_step


def build_value_drift_graph_html(run: RunResults, config: Any, output_dir: Path) -> str:
    """Build a standalone interactive graph for within-run value drift."""
    agent_names = [entity.name for entity in config.scenario.agents.entities]
    topology_type = str(config.scenario.get("topology", {}).get("type", "chain"))
    graph = build_topology_graph(agent_names, topology_type, config.scenario.get("topology", {}).get("custom_adjacency"))
    undirected_edges, directed_edges = _split_edges(graph)
    width, height, positions, backgrounds = _topology_layout(agent_names, topology_type)
    drifts, first_step, last_step = _compute_drifts(run)

    perturbation = config.scenario.get("perturbation", {})
    perturbation_enabled = bool(perturbation.get("enabled", False))
    target_index = int(perturbation.get("perturbed_agent_index", 0))
    target_agent = agent_names[target_index] if 0 <= target_index < len(agent_names) else None
    target_value = str(perturbation.get("target_value", ""))

    run_label = str(output_dir)
    meta_title = "Perturbed run" if perturbation_enabled else "Baseline run"
    lane_note = (
        "Within-run drift from the first probe to the final probe."
        if first_step != last_step
        else "Only one probe step was available."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ValueFlow Value Drift Graph</title>
  <style>
    :root {{
      --bg: #f4f0e8;
      --panel: #fffdf8;
      --ink: #1f1f1f;
      --muted: #5d5d5d;
      --cluster-a: #d7e7ff;
      --cluster-b: #dff4dd;
      --cluster-c: #ffe3d6;
      --edge: #9aa3ad;
      --bridge: #34404f;
      --stroke: #2f2a24;
      --positive: 28, 124, 84;
      --negative: 178, 58, 72;
      --neutral: 122, 111, 99;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top, #fff8ef 0%, var(--bg) 45%, #e7dfd1 100%);
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }}
    .wrap {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px 24px 36px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0.01em; }}
    .sub {{
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.45;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 0 0 18px;
    }}
    .control {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid rgba(47, 42, 36, 0.14);
      background: rgba(255, 255, 255, 0.85);
      color: var(--ink);
      cursor: pointer;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.01em;
      transition: transform 120ms ease, background 120ms ease, box-shadow 120ms ease;
    }}
    .control:hover {{ transform: translateY(-1px); background: #ffffff; }}
    .control.active {{
      background: #1f1f1f;
      color: #fff8ef;
      box-shadow: 0 8px 20px rgba(31, 31, 31, 0.18);
    }}
    .control.target-value {{
      border-color: rgba(180, 134, 22, 0.45);
      box-shadow: inset 0 0 0 1px rgba(180, 134, 22, 0.18);
    }}
    .control.target-value::after {{
      content: "target";
      margin-left: 4px;
      padding: 3px 7px;
      border-radius: 999px;
      background: rgba(180, 134, 22, 0.14);
      color: #7b5a0e;
      font-size: 10px;
      font-weight: 900;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .icon {{ width: 18px; text-align: center; font-size: 15px; }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(860px, 1fr) 340px;
      gap: 20px;
      align-items: start;
    }}
    .panel {{
      background: color-mix(in srgb, var(--panel) 94%, white);
      border: 1px solid rgba(47, 42, 36, 0.12);
      border-radius: 18px;
      box-shadow: 0 18px 40px rgba(70, 56, 38, 0.10);
      overflow: hidden;
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
      background:
        radial-gradient(circle at 20% 15%, rgba(255,255,255,0.95), rgba(255,255,255,0) 28%),
        linear-gradient(180deg, rgba(255,255,255,0.55), rgba(255,255,255,0.12));
    }}
    .cluster-bg {{
      stroke: rgba(47, 42, 36, 0.10);
      stroke-width: 2;
    }}
    .edge {{
      stroke: var(--edge);
      stroke-opacity: 0.62;
      stroke-width: 2;
    }}
    .bridge-edge {{
      stroke: var(--bridge);
      stroke-opacity: 0.88;
      stroke-width: 3.5;
    }}
    .directed-edge {{
      stroke: var(--edge);
      stroke-opacity: 0.75;
      stroke-width: 2.4;
      marker-end: url(#arrow);
    }}
    .caption {{
      font-size: 13px;
      fill: var(--muted);
      text-anchor: middle;
      font-weight: 700;
      letter-spacing: 0.04em;
    }}
    .node {{
      stroke: var(--stroke);
      stroke-width: 2.2;
      transition: fill 150ms ease;
    }}
    .target-ring {{
      fill: none;
      stroke: rgba(180, 134, 22, 0.75);
      stroke-width: 4;
    }}
    .node-label {{
      text-anchor: middle;
      fill: var(--ink);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.02em;
    }}
    .delta {{
      text-anchor: middle;
      font-size: 15px;
      font-weight: 900;
    }}
    .target-badge {{
      fill: #fff3c4;
      stroke: rgba(123, 90, 14, 0.35);
      stroke-width: 1.2;
    }}
    .target-badge-text {{
      text-anchor: middle;
      fill: #7b5a0e;
      font-size: 9px;
      font-weight: 900;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .side {{ padding: 20px 20px 18px; }}
    .side h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .side p, .side li, .metric {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }}
    .side ul {{ margin: 0 0 16px 18px; padding: 0; }}
    .cluster-chip {{
      display: inline-block;
      padding: 5px 10px;
      border-radius: 999px;
      margin: 0 6px 6px 0;
      font-size: 12px;
      font-weight: 700;
      color: var(--ink);
      border: 1px solid rgba(47, 42, 36, 0.12);
    }}
    .cluster-a {{ background: var(--cluster-a); }}
    .cluster-b {{ background: var(--cluster-b); }}
    .cluster-c {{ background: var(--cluster-c); }}
    .target-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 12px;
      margin: 10px 0 16px;
      background: #fff3c4;
      border: 1px solid rgba(123, 90, 14, 0.18);
      color: #6e5110;
      font-size: 13px;
      font-weight: 800;
    }}
    .footer {{
      padding: 12px 18px 16px;
      border-top: 1px solid rgba(47, 42, 36, 0.10);
      color: var(--muted);
      font-size: 13px;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.95em;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>ValueFlow Topology: Within-Run Value Drift</h1>
    <p class="sub">
      Run: <code>{run_label}</code>. {lane_note}
      Values shown on each node are <code>step {last_step} - step {first_step}</code>.
    </p>
    <div class="controls" id="controls"></div>
    <div class="grid">
      <div class="panel">
        <svg id="graph" viewBox="0 0 {width} {height}" role="img" aria-label="ValueFlow topology graph with switchable value drifts">
          <defs>
            <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
              <path d="M 0 0 L 12 6 L 0 12 z" fill="#8f98a2"></path>
            </marker>
          </defs>
        </svg>
        <div class="footer">
          Topology: <code>{topology_type}</code>. The graph renders the realized topology for this run and colors nodes by within-run value drift.
        </div>
      </div>
      <div class="panel side">
        <h2 id="side-title">Current Value</h2>
        <p id="side-desc" class="metric"></p>
        {"<div class='target-chip'>Target perturbation: <strong>" + target_agent.replace("Agent_", "A") + "</strong> on <strong>" + target_value + "</strong></div>" if perturbation_enabled and target_agent and target_value else "<div class='target-chip'>Baseline run: <strong>no perturbation</strong></div>"}
        {"" if topology_type != "community" else """
        <div class="cluster-chip cluster-a">Cluster 1: A0-A4</div>
        <div class="cluster-chip cluster-b">Cluster 2: A5-A9</div>
        <div class="cluster-chip cluster-c">Cluster 3: A10-A14</div>
        """}
        <h2>Run Metadata</h2>
        <p class="metric"><strong>Topology:</strong> {topology_type}</p>
        <p class="metric"><strong>Mode:</strong> {meta_title}</p>
        <p class="metric"><strong>Rounds:</strong> {int(config.scenario.get("interaction", {}).get("num_rounds", 0))}</p>
        <p class="metric"><strong>Probe steps:</strong> {first_step} and {last_step}</p>
        <h2>Summary</h2>
        <ul>
          <li id="top-positive"></li>
          <li id="top-negative"></li>
          <li id="graph-summary"></li>
        </ul>
        <h2>Segment Means</h2>
        <p class="metric" id="segment-1"></p>
        <p class="metric" id="segment-2"></p>
        <p class="metric" id="segment-3"></p>
        <h2>Interpretation</h2>
        <ul>
          <li>Positive values mean the selected value strengthened over the run.</li>
          <li>Negative values mean the selected value weakened over the run.</li>
          <li>This graph is within-run only; it does not compare against baseline.</li>
        </ul>
      </div>
    </div>
  </div>
  <script>
    const valueMeta = {json.dumps(VALUE_META)};
    const graphMeta = {{
      topology: {json.dumps(topology_type)},
      targetAgent: {json.dumps(target_agent)},
      targetValue: {json.dumps(target_value)},
      perturbationEnabled: {json.dumps(perturbation_enabled)},
      backgrounds: {json.dumps(backgrounds)},
      positions: {json.dumps(positions)},
      undirectedEdges: {json.dumps(undirected_edges)},
      directedEdges: {json.dumps(directed_edges)},
      values: {json.dumps(drifts)},
    }};

    const graphSvg = document.getElementById("graph");
    const controls = document.getElementById("controls");

    function fmt(value) {{
      if (Math.abs(value) < 1e-9) return "0.0";
      return `${{value > 0 ? "+" : ""}}${{value.toFixed(1)}}`;
    }}

    function classFor(value) {{
      if (Math.abs(value) < 1e-9) return "neu";
      return value > 0 ? "pos" : "neg";
    }}

    function colorFor(value, maxAbs) {{
      const intensity = Math.min(Math.abs(value) / maxAbs, 1);
      if (Math.abs(value) < 1e-9) {{
        return `rgba(${{getComputedStyle(document.documentElement).getPropertyValue('--neutral')}}, 0.26)`;
      }}
      if (value > 0) {{
        return `rgba(${{getComputedStyle(document.documentElement).getPropertyValue('--positive')}}, ${{0.22 + intensity * 0.58}})`;
      }}
      return `rgba(${{getComputedStyle(document.documentElement).getPropertyValue('--negative')}}, ${{0.22 + intensity * 0.58}})`;
    }}

    function segmentInfo(data) {{
      const agentNames = Object.keys(graphMeta.positions);
      if (graphMeta.topology === "community" && agentNames.length === 15) {{
        return [
          ["Cluster 1", ["Agent_0","Agent_1","Agent_2","Agent_3","Agent_4"]],
          ["Cluster 2", ["Agent_5","Agent_6","Agent_7","Agent_8","Agent_9"]],
          ["Cluster 3", ["Agent_10","Agent_11","Agent_12","Agent_13","Agent_14"]],
        ];
      }}
      const third = Math.ceil(agentNames.length / 3);
      return [
        ["Early", agentNames.slice(0, third)],
        ["Middle", agentNames.slice(third, third * 2)],
        ["Late", agentNames.slice(third * 2)],
      ];
    }}

    function mean(values) {{
      return values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1);
    }}

    function updateSide(valueKey, data) {{
      const meta = valueMeta[valueKey] || {{label: valueKey, icon: "•"}};
      const entries = Object.entries(data).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
      const topPos = [...entries].filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1])[0];
      const topNeg = [...entries].filter(([, v]) => v < 0).sort((a, b) => a[1] - b[1])[0];

      document.getElementById("side-title").textContent = `${{meta.label}} ${{meta.icon}}`;
      document.getElementById("side-desc").innerHTML =
        `Within-run drift for <strong>${{meta.label}}</strong> from the first probe to the final probe in this ${meta_title.lower()}.`;

      document.getElementById("top-positive").innerHTML =
        topPos ? `Strongest upward drift: <strong>${{topPos[0].replace("Agent_", "A")}}</strong> ${{fmt(topPos[1])}}` : "No positive drift.";
      document.getElementById("top-negative").innerHTML =
        topNeg ? `Strongest downward drift: <strong>${{topNeg[0].replace("Agent_", "A")}}</strong> ${{fmt(topNeg[1])}}` : "No negative drift.";

      const segments = segmentInfo(data);
      const ranked = segments.map(([label, agents]) => {{
        const values = agents.map((a) => Math.abs(data[a] || 0));
        return [label, mean(values)];
      }}).sort((a, b) => b[1] - a[1]);
      document.getElementById("graph-summary").innerHTML =
        `Largest average movement by magnitude: <strong>${{ranked[0][0]}}</strong> (${{ranked[0][1].toFixed(2)}} mean |Δ|).`;

      segments.forEach(([label, agents], idx) => {{
        const vals = agents.map((a) => data[a] || 0);
        const el = document.getElementById(`segment-${{idx + 1}}`);
        if (!el) return;
        el.innerHTML = `<strong>${{label}}:</strong> mean ${{fmt(mean(vals))}}, mean |Δ| ${{mean(vals.map((v) => Math.abs(v))).toFixed(2)}}`;
      }});
    }}

    function renderGraph() {{
      graphMeta.backgrounds.forEach((bg) => {{
        if (bg.type === "ellipse") {{
          const el = document.createElementNS("http://www.w3.org/2000/svg", "ellipse");
          el.setAttribute("class", "cluster-bg");
          el.setAttribute("cx", bg.cx);
          el.setAttribute("cy", bg.cy);
          el.setAttribute("rx", bg.rx);
          el.setAttribute("ry", bg.ry);
          el.setAttribute("fill", bg.fill);
          graphSvg.appendChild(el);
        }} else {{
          const el = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          el.setAttribute("class", "cluster-bg");
          el.setAttribute("x", bg.x);
          el.setAttribute("y", bg.y);
          el.setAttribute("width", bg.width);
          el.setAttribute("height", bg.height);
          el.setAttribute("rx", bg.rx || 0);
          el.setAttribute("fill", bg.fill);
          graphSvg.appendChild(el);
        }}
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("class", "caption");
        label.setAttribute("x", bg.label_x);
        label.setAttribute("y", bg.label_y);
        label.textContent = bg.label;
        graphSvg.appendChild(label);
      }});

      graphMeta.undirectedEdges.forEach(([a, b]) => {{
        const pa = graphMeta.positions[a];
        const pb = graphMeta.positions[b];
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("class", pa.x === pb.x || pa.y === pb.y ? "bridge-edge" : "edge");
        line.setAttribute("x1", pa.x);
        line.setAttribute("y1", pa.y);
        line.setAttribute("x2", pb.x);
        line.setAttribute("y2", pb.y);
        graphSvg.appendChild(line);
      }});

      graphMeta.directedEdges.forEach(([source, target]) => {{
        const pa = graphMeta.positions[source];
        const pb = graphMeta.positions[target];
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("class", "directed-edge");
        line.setAttribute("x1", pa.x);
        line.setAttribute("y1", pa.y);
        line.setAttribute("x2", pb.x);
        line.setAttribute("y2", pb.y);
        graphSvg.appendChild(line);
      }});

      const nodesLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      nodesLayer.setAttribute("id", "nodes");
      Object.entries(graphMeta.positions).forEach(([agent, pos]) => {{
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("data-agent", agent);
        group.setAttribute("transform", `translate(${{pos.x}} ${{pos.y}})`);

        if (graphMeta.perturbationEnabled && graphMeta.targetAgent === agent) {{
          const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          ring.setAttribute("class", "target-ring");
          ring.setAttribute("r", "39");
          group.appendChild(ring);
        }}

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("class", "node");
        circle.setAttribute("r", "34");
        group.appendChild(circle);

        if (graphMeta.perturbationEnabled && graphMeta.targetAgent === agent) {{
          const badge = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          badge.setAttribute("class", "target-badge");
          badge.setAttribute("x", "-23");
          badge.setAttribute("y", "-58");
          badge.setAttribute("width", "46");
          badge.setAttribute("height", "16");
          badge.setAttribute("rx", "8");
          group.appendChild(badge);

          const badgeText = document.createElementNS("http://www.w3.org/2000/svg", "text");
          badgeText.setAttribute("class", "target-badge-text");
          badgeText.setAttribute("y", "-47");
          badgeText.textContent = "target";
          group.appendChild(badgeText);
        }}

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("class", "node-label");
        label.setAttribute("y", "-6");
        label.textContent = agent.replace("Agent_", "A");
        group.appendChild(label);

        const delta = document.createElementNS("http://www.w3.org/2000/svg", "text");
        delta.setAttribute("class", "delta");
        delta.setAttribute("y", "18");
        delta.textContent = "0.0";
        group.appendChild(delta);

        nodesLayer.appendChild(group);
      }});
      graphSvg.appendChild(nodesLayer);
    }}

    function updateGraph(valueKey) {{
      const data = graphMeta.values[valueKey];
      const maxAbs = Math.max(...Object.values(data).map((v) => Math.abs(v)), 1);
      [...document.querySelectorAll("#nodes > g")].forEach((g) => {{
        const agent = g.dataset.agent;
        const value = data[agent] || 0;
        const circle = g.querySelector(".node");
        const delta = g.querySelector(".delta");
        circle.style.fill = colorFor(value, maxAbs);
        delta.textContent = fmt(value);
        delta.setAttribute("class", `delta ${{classFor(value)}}`);
      }});
      [...controls.children].forEach((btn) => btn.classList.toggle("active", btn.dataset.value === valueKey));
      updateSide(valueKey, data);
    }}

    Object.entries(valueMeta).forEach(([key, meta]) => {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = `control${{graphMeta.perturbationEnabled && key === graphMeta.targetValue ? " target-value" : ""}}`;
      button.dataset.value = key;
      button.innerHTML = `<span class="icon">${{meta.icon}}</span><span>${{meta.label}}</span>`;
      button.addEventListener("click", () => updateGraph(key));
      controls.appendChild(button);
    }});

    renderGraph();
    updateGraph(graphMeta.targetValue && graphMeta.values[graphMeta.targetValue] ? graphMeta.targetValue : Object.keys(graphMeta.values)[0]);
  </script>
</body>
</html>"""


def write_value_drift_graph(run: RunResults, config: Any, output_dir: Path) -> Path:
    """Write the drift graph HTML into the run output directory."""
    html = build_value_drift_graph_html(run, config, output_dir)
    out_path = output_dir / "valueflow_value_drift_graph.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
