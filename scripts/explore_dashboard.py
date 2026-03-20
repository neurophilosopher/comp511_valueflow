#!/usr/bin/env python3
"""Interactive dashboard for exploring social media simulation results.

Usage:
    python scripts/explore_dashboard.py path/to/checkpoint.json
    python scripts/explore_dashboard.py path/to/checkpoint.json --port 8051 --debug
    python scripts/explore_dashboard.py path/to/checkpoint.json --seed-tags misinfo_seed health
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import dash_cytoscape as cyto
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update

from src.environments.social_media.analysis import TransmissionChain, find_transmission_chains
from src.environments.social_media.app import Post, SocialMediaApp

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class SimData:
    """Precomputed simulation data for the dashboard."""

    app: SocialMediaApp
    agents: list[str]
    max_step: int
    posts: list[Post]
    raw_log: list[dict[str, Any]]
    chains: list[TransmissionChain]

    # Precomputed edges: list of dicts with keys:
    #   type, source_agent, target_agent, source_post, target_post, step, label
    edges: list[dict[str, Any]] = field(default_factory=list)

    # Per-step action index: step -> list of action dicts
    step_actions: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    # Chain post ID sets for highlight
    chain_post_ids: dict[int, set[int]] = field(default_factory=dict)


def load_checkpoint(path: str | Path) -> tuple[SocialMediaApp, list[dict[str, Any]]]:
    """Load a checkpoint JSON and return (app, raw_log).

    Handles three formats:
    1. Full checkpoint: {game_masters: {gm_name: {state: {app_state: ...}}}, raw_log: [...]}
    2. App state wrapper: {app_state: {...}}
    3. Direct app state: {posts: [...], following: {...}, ...}
    """
    with Path(path).open() as f:
        data = json.load(f)

    raw_log: list[dict[str, Any]] = []

    # Format 1: full checkpoint
    if "game_masters" in data:
        raw_log = data.get("raw_log", [])
        for gm_data in data["game_masters"].values():
            state = gm_data.get("state", {})
            if "app_state" in state:
                return SocialMediaApp.from_dict(state["app_state"]), raw_log
        raise ValueError("No app_state found in game_masters")

    # Format 2: wrapped app state
    if "app_state" in data:
        raw_log = data.get("log", data.get("raw_log", []))
        return SocialMediaApp.from_dict(data["app_state"]), raw_log

    # Format 3: direct
    return SocialMediaApp.from_dict(data), raw_log


def precompute(
    app: SocialMediaApp,
    raw_log: list[dict[str, Any]],
    seed_tags: list[str],
) -> SimData:
    """Precompute all derived data needed by the dashboard."""
    posts = app.get_all_posts()
    max_step = max((p.step for p in posts), default=0)

    # Discover agents from posts + following/followers
    agents_set: set[str] = set()
    for p in posts:
        agents_set.add(p.author)
    for user in app._following:
        agents_set.add(user)
    agents = sorted(agents_set)

    # Chains
    chains = find_transmission_chains(app, seed_tags)

    # Build edges from posts
    edges: list[dict[str, Any]] = []
    posts_by_id = {p.id: p for p in posts}

    for p in posts:
        if p.reply_to and p.reply_to in posts_by_id:
            parent = posts_by_id[p.reply_to]
            edges.append(
                {
                    "type": "reply",
                    "source_agent": p.author,
                    "target_agent": parent.author,
                    "source_post": p.id,
                    "target_post": p.reply_to,
                    "step": p.step,
                    "label": f"reply #{p.id}->#{p.reply_to}",
                }
            )
        if p.boost_of and p.boost_of in posts_by_id:
            original = posts_by_id[p.boost_of]
            edges.append(
                {
                    "type": "boost",
                    "source_agent": p.author,
                    "target_agent": original.author,
                    "source_post": p.id,
                    "target_post": p.boost_of,
                    "step": p.step,
                    "label": f"boost #{p.boost_of}",
                }
            )

    # Like edges from raw_log
    for log_entry in raw_log:
        step = log_entry.get("Step", 0)
        for key, val in log_entry.items():
            if not key.startswith("Entity ["):
                continue
            if not isinstance(val, dict):
                continue
            result = val.get("result", {})
            if result.get("action_type") == "like" and result.get("success"):
                entity_name = val.get("entity", "")
                parsed = val.get("parsed", {})
                try:
                    target_id = int(parsed.get("target", 0))
                except (ValueError, TypeError):
                    continue
                if target_id in posts_by_id:
                    target_post = posts_by_id[target_id]
                    edges.append(
                        {
                            "type": "like",
                            "source_agent": entity_name,
                            "target_agent": target_post.author,
                            "source_post": None,
                            "target_post": target_id,
                            "step": step,
                            "label": f"like #{target_id}",
                        }
                    )

    # Follow edges from raw_log
    for log_entry in raw_log:
        step = log_entry.get("Step", 0)
        for key, val in log_entry.items():
            if not key.startswith("Entity ["):
                continue
            if not isinstance(val, dict):
                continue
            result = val.get("result", {})
            if result.get("action_type") == "follow" and result.get("success"):
                entity_name = val.get("entity", "")
                parsed = val.get("parsed", {})
                target_user = parsed.get("target", "")
                if target_user and target_user != "none":
                    edges.append(
                        {
                            "type": "follow",
                            "source_agent": entity_name,
                            "target_agent": target_user,
                            "source_post": None,
                            "target_post": None,
                            "step": step,
                            "label": f"follow @{target_user}",
                        }
                    )

    # Step actions index from raw_log
    step_actions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for log_entry in raw_log:
        step = log_entry.get("Step", 0)
        for key, val in log_entry.items():
            if not key.startswith("Entity ["):
                continue
            if not isinstance(val, dict):
                continue
            step_actions[step].append(val)

    # If no raw_log, infer actions from posts
    if not raw_log:
        for p in posts:
            action_type = "post"
            target: str | int = "none"
            if p.reply_to:
                action_type = "reply"
                target = p.reply_to
            elif p.boost_of:
                action_type = "boost"
                target = p.boost_of
            step_actions[p.step].append(
                {
                    "entity": p.author,
                    "parsed": {
                        "action_type": action_type,
                        "target": str(target),
                        "content": p.content,
                    },
                    "result": {
                        "success": True,
                        "action_type": action_type,
                        "message": f"#{p.id}",
                        "post_id": p.id,
                    },
                }
            )

    # Chain post ID sets
    chain_post_ids: dict[int, set[int]] = {}
    for chain in chains:
        ids = {chain.seed_post_id}
        for ev in chain.events:
            ids.add(ev.from_post_id)
            ids.add(ev.to_post_id)
        chain_post_ids[chain.seed_post_id] = ids

    return SimData(
        app=app,
        agents=agents,
        max_step=max_step,
        posts=posts,
        raw_log=raw_log,
        chains=chains,
        edges=edges,
        step_actions=dict(step_actions),
        chain_post_ids=chain_post_ids,
    )


# ---------------------------------------------------------------------------
# Module-level state (single-user local server)
# ---------------------------------------------------------------------------

SIM: SimData | None = None

# ---------------------------------------------------------------------------
# Cytoscape stylesheet
# ---------------------------------------------------------------------------

CYTO_STYLESHEET = [
    # Default node style
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "text-valign": "top",
            "text-halign": "center",
            "text-margin-y": -5,
            "text-wrap": "none",
            "background-color": "#4a90d9",
            "color": "#333",
            "font-size": "11px",
            "width": "data(size)",
            "height": "data(size)",
            "border-width": 2,
            "border-color": "#2c5f8a",
        },
    },
    # Seed author node
    {
        "selector": "node.seed-author",
        "style": {
            "background-color": "#e74c3c",
            "border-color": "#c0392b",
        },
    },
    # Edge defaults
    {
        "selector": "edge",
        "style": {
            "width": 2,
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#999",
            "line-color": "#999",
            "opacity": 0.7,
            "font-size": "9px",
        },
    },
    # Reply edges
    {
        "selector": "edge.reply",
        "style": {
            "line-color": "#27ae60",
            "target-arrow-color": "#27ae60",
            "target-arrow-shape": "triangle",
        },
    },
    # Boost edges
    {
        "selector": "edge.boost",
        "style": {
            "line-color": "#3498db",
            "target-arrow-color": "#3498db",
            "line-style": "dashed",
        },
    },
    # Like edges
    {
        "selector": "edge.like",
        "style": {
            "line-color": "#e91e8b",
            "target-arrow-color": "#e91e8b",
            "target-arrow-shape": "diamond",
            "width": 1.5,
        },
    },
    # Follow edges
    {
        "selector": "edge.follow",
        "style": {
            "line-color": "#bdc3c7",
            "target-arrow-color": "#bdc3c7",
            "line-style": "dotted",
            "width": 1,
        },
    },
    # Chain highlight
    {
        "selector": "edge.chain-highlight",
        "style": {
            "line-color": "#f39c12",
            "target-arrow-color": "#f39c12",
            "width": 4,
            "opacity": 1.0,
            "z-index": 10,
        },
    },
    {
        "selector": "node.chain-highlight",
        "style": {
            "border-color": "#f39c12",
            "border-width": 4,
        },
    },
    # Dimmed (non-chain when chain is selected)
    {
        "selector": ".dimmed",
        "style": {
            "opacity": 0.2,
        },
    },
]

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _edge_toggle_label(name: str, color: str, style: str) -> html.Span:
    """Build a checklist label with a colored line swatch next to the name."""
    return html.Span(
        style={"display": "inline-flex", "alignItems": "center", "gap": "4px"},
        children=[
            html.Span(
                style={
                    "display": "inline-block",
                    "width": "18px",
                    "height": "0",
                    "borderTop": f"3px {style} {color}",
                    "verticalAlign": "middle",
                },
            ),
            html.Span(name),
        ],
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def build_layout(sim: SimData) -> html.Div:
    """Build the Dash layout."""
    chain_options = [{"label": "None (show all)", "value": "none"}]
    for chain in sim.chains:
        label = (
            f"Chain from #{chain.seed_post_id} "
            f"[{','.join(chain.seed_tags)}] "
            f"(size:{chain.size} depth:{chain.depth} breadth:{chain.breadth})"
        )
        chain_options.append({"label": label, "value": str(chain.seed_post_id)})

    return html.Div(
        style={"fontFamily": "system-ui, sans-serif", "margin": "0", "padding": "0"},
        children=[
            # Top bar
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "padding": "10px 20px",
                    "backgroundColor": "#2c3e50",
                    "color": "#ecf0f1",
                    "gap": "20px",
                },
                children=[
                    html.H3(
                        "Social Media Simulation Explorer",
                        style={"margin": "0", "flexShrink": "0"},
                    ),
                    html.Div(
                        style={
                            "flex": "1",
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "10px",
                        },
                        children=[
                            html.Label("Step:", style={"flexShrink": "0"}),
                            dcc.Slider(
                                id="step-slider",
                                min=0,
                                max=sim.max_step,
                                step=1,
                                value=sim.max_step,
                                marks={i: str(i) for i in range(0, sim.max_step + 1)},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "5px", "flexShrink": "0"},
                        children=[
                            html.Button(
                                "Play",
                                id="play-button",
                                n_clicks=0,
                                style={
                                    "padding": "5px 15px",
                                    "backgroundColor": "#27ae60",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                },
                            ),
                            dcc.Interval(
                                id="play-interval",
                                interval=1500,
                                n_intervals=0,
                                disabled=True,
                            ),
                        ],
                    ),
                ],
            ),
            # Main content: two columns
            html.Div(
                style={
                    "display": "flex",
                    "height": "calc(100vh - 70px)",
                },
                children=[
                    # Left column
                    html.Div(
                        style={
                            "width": "55%",
                            "display": "flex",
                            "flexDirection": "column",
                            "borderRight": "1px solid #ddd",
                            "overflow": "hidden",
                        },
                        children=[
                            # Network graph
                            cyto.Cytoscape(
                                id="network-graph",
                                layout={"name": "cose", "animate": False, "randomize": False},
                                style={"flex": "1", "minHeight": "300px"},
                                stylesheet=CYTO_STYLESHEET,
                                elements=[],
                            ),
                            # Edge toggles with inline legend swatches
                            html.Div(
                                style={"padding": "8px 12px", "borderTop": "1px solid #ddd"},
                                children=[
                                    html.Label(
                                        "Edge types: ",
                                        style={"fontWeight": "bold", "marginRight": "10px"},
                                    ),
                                    dcc.Checklist(
                                        id="edge-toggles",
                                        options=[
                                            {
                                                "label": _edge_toggle_label(
                                                    "Reply", "#27ae60", "solid"
                                                ),
                                                "value": "reply",
                                            },
                                            {
                                                "label": _edge_toggle_label(
                                                    "Boost", "#3498db", "dashed"
                                                ),
                                                "value": "boost",
                                            },
                                            {
                                                "label": _edge_toggle_label(
                                                    "Like", "#e91e8b", "solid"
                                                ),
                                                "value": "like",
                                            },
                                            {
                                                "label": _edge_toggle_label(
                                                    "Follow", "#bdc3c7", "dotted"
                                                ),
                                                "value": "follow",
                                            },
                                        ],
                                        value=["reply", "boost", "like"],
                                        inline=True,
                                        style={"display": "inline-flex", "gap": "15px"},
                                    ),
                                ],
                            ),
                            # Charts
                            html.Div(
                                style={
                                    "display": "flex",
                                    "height": "200px",
                                    "borderTop": "1px solid #ddd",
                                },
                                children=[
                                    dcc.Graph(
                                        id="posts-per-step-chart",
                                        style={"flex": "1"},
                                        config={"displayModeBar": False},
                                    ),
                                    dcc.Graph(
                                        id="action-breakdown-chart",
                                        style={"flex": "1"},
                                        config={"displayModeBar": False},
                                    ),
                                ],
                            ),
                            # Chain selector
                            html.Div(
                                style={
                                    "padding": "8px 12px",
                                    "borderTop": "1px solid #ddd",
                                    "backgroundColor": "#f8f9fa",
                                },
                                children=[
                                    html.Label(
                                        "Transmission Chain: ",
                                        style={"fontWeight": "bold"},
                                    ),
                                    dcc.Dropdown(
                                        id="chain-dropdown",
                                        options=chain_options,
                                        value="none",
                                        clearable=False,
                                        style={"marginTop": "4px"},
                                    ),
                                    html.Div(
                                        id="chain-stats",
                                        style={"marginTop": "4px", "fontSize": "13px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        style={
                            "width": "45%",
                            "display": "flex",
                            "flexDirection": "column",
                            "overflow": "hidden",
                        },
                        children=[
                            # Step activity panel
                            html.Div(
                                style={
                                    "flex": "1",
                                    "overflow": "auto",
                                    "padding": "12px",
                                    "borderBottom": "1px solid #ddd",
                                },
                                children=[
                                    html.H4(id="activity-header", style={"margin": "0 0 8px 0"}),
                                    html.Div(id="activity-list"),
                                ],
                            ),
                            # Post detail panel
                            html.Div(
                                style={
                                    "flex": "1",
                                    "overflow": "auto",
                                    "padding": "12px",
                                    "backgroundColor": "#f8f9fa",
                                },
                                children=[
                                    html.H4("Detail", style={"margin": "0 0 8px 0"}),
                                    html.Div(
                                        id="detail-panel",
                                        children=[
                                            html.P(
                                                "Click a node or edge in the graph to see details.",
                                                style={"color": "#888"},
                                            )
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Helper: build cytoscape elements
# ---------------------------------------------------------------------------


def build_cytoscape_elements(
    step: int,
    edge_types: list[str],
    chain_seed_id: int | None,
) -> list[dict[str, Any]]:
    """Build cytoscape node/edge elements filtered to the given step."""
    assert SIM is not None  # nosec B101
    sim = SIM

    # Posts up to this step
    posts_up_to = [p for p in sim.posts if p.step <= step]
    post_count: dict[str, int] = defaultdict(int)
    for p in posts_up_to:
        post_count[p.author] += 1

    # Chain highlight sets
    highlight_posts: set[int] | None = None
    highlight_agents: set[str] | None = None
    if chain_seed_id is not None and chain_seed_id in sim.chain_post_ids:
        highlight_posts = sim.chain_post_ids[chain_seed_id]
        highlight_agents = set()
        for p in sim.posts:
            if p.id in highlight_posts:
                highlight_agents.add(p.author)
        # Also add seed author
        for chain in sim.chains:
            if chain.seed_post_id == chain_seed_id:
                highlight_agents.add(chain.seed_author)

    # Seed authors
    seed_authors = {c.seed_author for c in sim.chains}

    # Nodes
    elements: list[dict[str, Any]] = []
    for agent in sim.agents:
        count = post_count.get(agent, 0)
        size = max(25, min(60, 25 + count * 5))
        classes = []
        if agent in seed_authors:
            classes.append("seed-author")
        if highlight_agents is not None:
            if agent in highlight_agents:
                classes.append("chain-highlight")
            else:
                classes.append("dimmed")
        elements.append(
            {
                "data": {"id": agent, "label": f"{agent} ({count})", "size": size},
                "classes": " ".join(classes),
            }
        )

    # Edges
    edge_id = 0
    for edge in sim.edges:
        if edge["step"] > step:
            continue
        if edge["type"] not in edge_types:
            continue

        classes = [edge["type"]]

        # Chain highlight logic
        if highlight_posts is not None:
            is_in_chain = False
            if edge["source_post"] and edge["source_post"] in highlight_posts:
                is_in_chain = True
            if edge["target_post"] and edge["target_post"] in highlight_posts:
                is_in_chain = True
            if is_in_chain:
                classes.append("chain-highlight")
            else:
                classes.append("dimmed")

        elements.append(
            {
                "data": {
                    "id": f"e{edge_id}",
                    "source": edge["source_agent"],
                    "target": edge["target_agent"],
                    "label": edge["label"],
                    "edge_type": edge["type"],
                    "source_post": edge.get("source_post"),
                    "target_post": edge.get("target_post"),
                    "step": edge["step"],
                },
                "classes": " ".join(classes),
            }
        )
        edge_id += 1

    return elements


# ---------------------------------------------------------------------------
# App + Callbacks
# ---------------------------------------------------------------------------


def create_app(sim: SimData) -> Dash:
    """Create and configure the Dash app with all callbacks."""
    global SIM
    SIM = sim

    app = Dash(__name__, title="Social Media Sim Explorer")
    app.layout = build_layout(sim)

    # --- Callback 1: Update network graph ---
    @app.callback(
        Output("network-graph", "elements"),
        [
            Input("step-slider", "value"),
            Input("edge-toggles", "value"),
            Input("chain-dropdown", "value"),
        ],
    )
    def update_network(step: int, edge_types: list[str], chain_value: str) -> list[dict[str, Any]]:
        chain_seed_id = None if chain_value == "none" else int(chain_value)
        return build_cytoscape_elements(step, edge_types, chain_seed_id)

    # --- Callback 2: Update step activity panel ---
    @app.callback(
        [Output("activity-header", "children"), Output("activity-list", "children")],
        Input("step-slider", "value"),
    )
    def update_step_activity(step: int) -> tuple[str, list[Any]]:
        assert SIM is not None  # nosec B101
        actions = SIM.step_actions.get(step, [])
        header = f"Step {step} Activity ({len(actions)} actions)"

        if not actions:
            return header, [html.P("No actions at this step.", style={"color": "#888"})]

        items = []
        for act in actions:
            entity = act.get("entity", "?")
            result = act.get("result", {})
            action_type = result.get("action_type", "?")
            message = result.get("message", "")
            success = result.get("success", False)

            color = "#27ae60" if success else "#e74c3c"
            icon = _action_icon(action_type)

            items.append(
                html.Div(
                    style={
                        "padding": "4px 8px",
                        "marginBottom": "4px",
                        "borderLeft": f"3px solid {color}",
                        "backgroundColor": "#f8f9fa",
                        "fontSize": "13px",
                    },
                    children=[
                        html.Strong(f"{icon} {entity}: "),
                        html.Span(f"{action_type} — {message}"),
                    ],
                )
            )
        return header, items

    # --- Callback 3: Update charts ---
    @app.callback(
        [Output("posts-per-step-chart", "figure"), Output("action-breakdown-chart", "figure")],
        Input("step-slider", "value"),
    )
    def update_charts(step: int) -> tuple[go.Figure, go.Figure]:
        assert SIM is not None  # nosec B101

        # Posts per step bar chart
        steps_range = list(range(0, SIM.max_step + 1))
        post_counts = []
        for s in steps_range:
            post_counts.append(sum(1 for p in SIM.posts if p.step == s))

        colors = ["#3498db" if s <= step else "#bdc3c7" for s in steps_range]

        fig_posts = go.Figure(
            data=[go.Bar(x=steps_range, y=post_counts, marker_color=colors)],
            layout=go.Layout(
                title={"text": "Posts per Step", "font": {"size": 13}},
                xaxis={"title": "Step", "dtick": 1},
                yaxis={"title": "Posts"},
                margin={"l": 40, "r": 10, "t": 30, "b": 30},
                height=190,
            ),
        )

        # Action breakdown stacked bar
        action_types = ["post", "reply", "like", "boost", "follow", "skip"]
        action_colors = {
            "post": "#3498db",
            "reply": "#27ae60",
            "like": "#e91e8b",
            "boost": "#2980b9",
            "follow": "#95a5a6",
            "skip": "#bdc3c7",
        }

        traces = []
        for atype in action_types:
            counts = []
            for s in steps_range:
                actions = SIM.step_actions.get(s, [])
                c = sum(1 for a in actions if a.get("result", {}).get("action_type") == atype)
                counts.append(c)
            if any(c > 0 for c in counts):
                traces.append(
                    go.Bar(
                        name=atype,
                        x=steps_range,
                        y=counts,
                        marker_color=action_colors.get(atype, "#999"),
                    )
                )

        fig_actions = go.Figure(
            data=traces,
            layout=go.Layout(
                title={"text": "Action Breakdown", "font": {"size": 13}},
                barmode="stack",
                xaxis={"title": "Step", "dtick": 1},
                yaxis={"title": "Count"},
                margin={"l": 40, "r": 10, "t": 30, "b": 30},
                height=190,
                legend={"orientation": "h", "yanchor": "top", "y": -0.25, "font": {"size": 10}},
                showlegend=True,
            ),
        )

        return fig_posts, fig_actions

    # --- Callback 4: Chain stats ---
    @app.callback(
        Output("chain-stats", "children"),
        Input("chain-dropdown", "value"),
    )
    def update_chain_stats(chain_value: str) -> str | html.Div:
        if chain_value == "none":
            return ""
        assert SIM is not None  # nosec B101
        seed_id = int(chain_value)
        for chain in SIM.chains:
            if chain.seed_post_id == seed_id:
                return html.Div(
                    [
                        html.Span(f"Size: {chain.size}", style={"marginRight": "15px"}),
                        html.Span(f"Depth: {chain.depth}", style={"marginRight": "15px"}),
                        html.Span(f"Breadth: {chain.breadth}", style={"marginRight": "15px"}),
                        html.Span(f"Reach: {chain.reach}"),
                    ]
                )
        return "Chain not found."

    # --- Callback 5: Detail panel (node/edge tap) ---
    @app.callback(
        Output("detail-panel", "children"),
        [
            Input("network-graph", "tapNodeData"),
            Input("network-graph", "tapEdgeData"),
        ],
    )
    def show_detail(
        node_data: dict[str, Any] | None,
        edge_data: dict[str, Any] | None,
    ) -> Any:
        assert SIM is not None  # nosec B101
        triggered = callback_context.triggered
        if not triggered:
            return no_update

        trigger_id = triggered[0]["prop_id"]

        if "tapNodeData" in trigger_id and node_data:
            return _render_agent_profile(node_data["id"])
        elif "tapEdgeData" in trigger_id and edge_data:
            return _render_edge_detail(edge_data)

        return no_update

    # --- Callback 6: Play/pause ---
    @app.callback(
        [Output("play-interval", "disabled"), Output("play-button", "children")],
        Input("play-button", "n_clicks"),
        State("play-interval", "disabled"),
    )
    def toggle_play(n_clicks: int, currently_disabled: bool) -> tuple[bool, str]:
        if n_clicks == 0:
            return True, "Play"
        new_disabled = not currently_disabled
        label = "Play" if new_disabled else "Pause"
        return new_disabled, label

    @app.callback(
        Output("step-slider", "value"),
        Input("play-interval", "n_intervals"),
        State("step-slider", "value"),
    )
    def advance_step(n_intervals: int, current_step: int) -> int:
        assert SIM is not None  # nosec B101
        if current_step >= SIM.max_step:
            return 0  # wrap around
        return current_step + 1

    return app


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _action_icon(action_type: str) -> str:
    icons = {
        "post": "[POST]",
        "reply": "[REPLY]",
        "like": "[LIKE]",
        "boost": "[BOOST]",
        "follow": "[FOLLOW]",
        "unfollow": "[UNFOLLOW]",
        "skip": "[SKIP]",
    }
    return icons.get(action_type, "[?]")


def _render_agent_profile(agent_name: str) -> html.Div:
    assert SIM is not None  # nosec B101
    sim = SIM

    posts = [p for p in sim.posts if p.author == agent_name]
    following = sim.app.get_following(agent_name)
    followers = sim.app.get_followers(agent_name)

    recent_posts = sorted(posts, key=lambda p: (-p.step, -p.id))[:5]

    return html.Div(
        [
            html.H4(f"@{agent_name}", style={"margin": "0 0 8px 0", "color": "#2c3e50"}),
            html.Div(
                style={"display": "flex", "gap": "20px", "marginBottom": "10px"},
                children=[
                    html.Span(f"Posts: {len(posts)}"),
                    html.Span(f"Following: {len(following)}"),
                    html.Span(f"Followers: {len(followers)}"),
                ],
            ),
            html.Div(
                style={"marginBottom": "8px", "fontSize": "13px"},
                children=[
                    html.Strong("Following: "),
                    html.Span(", ".join(sorted(following)) if following else "nobody"),
                ],
            ),
            html.Div(
                style={"marginBottom": "8px", "fontSize": "13px"},
                children=[
                    html.Strong("Followers: "),
                    html.Span(", ".join(sorted(followers)) if followers else "none"),
                ],
            ),
            html.Hr(),
            html.H5("Recent Posts", style={"margin": "8px 0"}),
            *[_render_post_card(p) for p in recent_posts],
        ]
    )


def _render_post_card(post: Post) -> html.Div:
    assert SIM is not None  # nosec B101
    likes = SIM.app.get_like_count(post.id)
    replies = SIM.app.get_reply_count(post.id)
    boosts = SIM.app.get_boost_count(post.id)

    header_parts = [f"#{post.id} by @{post.author} (step {post.step})"]
    if post.reply_to:
        header_parts.append(f"reply to #{post.reply_to}")
    if post.boost_of:
        header_parts.append(f"boost of #{post.boost_of}")

    return html.Div(
        style={
            "padding": "8px",
            "marginBottom": "6px",
            "border": "1px solid #ddd",
            "borderRadius": "4px",
            "backgroundColor": "#fff",
            "fontSize": "13px",
        },
        children=[
            html.Div(
                " | ".join(header_parts),
                style={"fontWeight": "bold", "marginBottom": "4px", "color": "#555"},
            ),
            html.Div(
                f'"{post.content}"',
                style={"marginBottom": "4px", "fontStyle": "italic"},
            ),
            html.Div(
                f"Likes: {likes} | Replies: {replies} | Boosts: {boosts}",
                style={"color": "#888", "fontSize": "12px"},
            ),
            html.Div(
                f"Tags: {', '.join(post.tags)}" if post.tags else "",
                style={"color": "#888", "fontSize": "12px"},
            ),
        ],
    )


def _render_edge_detail(edge_data: dict[str, Any]) -> html.Div:
    assert SIM is not None  # nosec B101
    sim = SIM

    edge_type = edge_data.get("edge_type", "?")
    source_agent = edge_data.get("source", "?")
    target_agent = edge_data.get("target", "?")
    source_post_id = edge_data.get("source_post")
    target_post_id = edge_data.get("target_post")
    step = edge_data.get("step", "?")

    children: list[Any] = [
        html.H4(
            f"{edge_type.upper()}: {source_agent} -> {target_agent}",
            style={"margin": "0 0 8px 0", "color": "#2c3e50"},
        ),
        html.Div(f"Step: {step}", style={"marginBottom": "8px"}),
    ]

    # Show target post content
    if target_post_id:
        post = sim.app.get_post(int(target_post_id))
        if post:
            children.append(html.Hr())
            children.append(html.H5("Target Post"))
            children.append(_render_post_card(post))

    # Show source post content if different
    if source_post_id and source_post_id != target_post_id:
        post = sim.app.get_post(int(source_post_id))
        if post:
            children.append(html.Hr())
            children.append(html.H5("Source Post"))
            children.append(_render_post_card(post))

    return html.Div(children)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive dashboard for exploring social media simulation results.",
    )
    parser.add_argument("checkpoint", help="Path to checkpoint JSON file")
    parser.add_argument("--port", type=int, default=8050, help="Server port (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    parser.add_argument(
        "--seed-tags",
        nargs="+",
        default=["misinfo_seed"],
        help="Tags identifying seed posts (default: misinfo_seed)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    app_state, raw_log = load_checkpoint(checkpoint_path)

    print("Precomputing simulation data...")
    sim = precompute(app_state, raw_log, args.seed_tags)
    print(
        f"  {len(sim.agents)} agents, {len(sim.posts)} posts, "
        f"{sim.max_step + 1} steps, {len(sim.edges)} edges, {len(sim.chains)} chains"
    )

    dash_app = create_app(sim)
    print(f"\nStarting dashboard on http://localhost:{args.port}")
    dash_app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
