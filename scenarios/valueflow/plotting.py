"""Plotting functions for ValueFlow experiment results.

Each function takes the structured output of compute_all_metrics() or
compute_cross_topology_comparison() and returns (fig, ax) for use in
notebooks or scripts.

Figures map to the main paper results (arxiv 2602.08567):
  plot_ss_by_topology()    -> H1: topology bar chart
  plot_beta_heatmap()      -> main figure: β-susceptibility across agents x values
  plot_beta_timeseries()   -> round-by-round propagation dynamics
  plot_ss_by_value_type()  -> H3: SS grouped by Schwartz value type
  plot_location_effect()   -> H4: SS vs. perturbation position in chain
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ── Schwartz value-type ordering and colors ───────────────────────────────────

VALUE_TYPE_ORDER = [
    "power",
    "achievement",
    "hedonism",
    "stimulation",
    "self_direction",
    "universalism",
    "benevolence",
    "tradition",
    "conformity",
    "security",
]

VALUE_TYPE_COLORS = {
    "power": "#e63946",
    "achievement": "#f4a261",
    "hedonism": "#e9c46a",
    "stimulation": "#2a9d8f",
    "self_direction": "#457b9d",
    "universalism": "#1d3557",
    "benevolence": "#6a994e",
    "tradition": "#a7c957",
    "conformity": "#bc6c25",
    "security": "#8d99ae",
}

TOPOLOGY_COLORS = {
    "chain": "#457b9d",
    "ring": "#2a9d8f",
    "star": "#e9c46a",
    "fully_connected": "#e63946",
}

AGENT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ── H1: System Susceptibility by topology ─────────────────────────────────────


def plot_ss_by_topology(
    topology_comparison: dict[str, dict[str, Any]],
    target_value: str = "",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of System Susceptibility across topologies (H1).

    Args:
        topology_comparison: Output of compute_cross_topology_comparison().
            Keys are topology names; values have 'target_value_ss'.
        target_value: Label for the perturbed value (used in title).
        ax: Existing axes to draw on; creates a new figure if None.

    Returns:
        (fig, ax) tuple.
    """
    topologies = list(topology_comparison.keys())
    ss_values = [topology_comparison[t]["target_value_ss"] for t in topologies]

    # Sort by SS descending
    order = np.argsort(ss_values)[::-1]
    topologies = [topologies[i] for i in order]
    ss_values = [ss_values[i] for i in order]
    colors = [TOPOLOGY_COLORS.get(t, "#888888") for t in topologies]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    bars = ax.bar(topologies, ss_values, color=colors, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, ss_values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    title = "System Susceptibility by Topology"
    if target_value:
        title += f"\n(perturbed value: {target_value.replace('_', ' ')})"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Topology", fontsize=11)
    ax.set_ylabel("System Susceptibility (SS)", fontsize=11)
    ax.set_ylim(0, max(ss_values) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, ax


# ── Main figure: β-susceptibility heatmap ────────────────────────────────────


def plot_beta_heatmap(
    beta_susceptibility: dict[str, dict[str, float]],
    value_type_map: dict[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of β-susceptibility: agents (columns) x values (rows).

    Diverging colormap: red = positive shift toward perturbed value,
    blue = negative shift.  Rows are grouped by Schwartz value type.

    Args:
        beta_susceptibility: metrics["beta_susceptibility"] —
            dict[value_name -> dict[agent_name -> float]].
        value_type_map: Optional dict[value_name -> value_type] for row grouping.
        ax: Existing axes; creates new figure if None.

    Returns:
        (fig, ax) tuple.
    """
    # Collect agents and values
    all_values = list(beta_susceptibility.keys())
    all_agents: list[str] = []
    for agents_dict in beta_susceptibility.values():
        for a in agents_dict:
            if a not in all_agents:
                all_agents.append(a)
    all_agents = sorted(all_agents)

    # Sort values by type if map provided
    if value_type_map:

        def _sort_key(v: str) -> tuple[int, str]:
            vtype = value_type_map.get(v, "zzz")
            type_rank = VALUE_TYPE_ORDER.index(vtype) if vtype in VALUE_TYPE_ORDER else 99
            return (type_rank, v)

        all_values = sorted(all_values, key=_sort_key)

    # Build matrix
    matrix = np.zeros((len(all_values), len(all_agents)))
    for r, value in enumerate(all_values):
        for c, agent in enumerate(all_agents):
            matrix[r, c] = beta_susceptibility[value].get(agent, 0.0)

    vmax = max(abs(matrix.max()), abs(matrix.min()), 1.0)

    fig_h = max(6, len(all_values) * 0.28)
    fig_w = max(5, len(all_agents) * 1.1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.get_figure()

    im = ax.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(range(len(all_agents)))
    ax.set_xticklabels(all_agents, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(all_values)))
    ax.set_yticklabels([v.replace("_", " ") for v in all_values], fontsize=8)

    # Annotate cells with values
    for r in range(len(all_values)):
        for c in range(len(all_agents)):
            val = matrix[r, c]
            if abs(val) > 0.05:
                ax.text(
                    c,
                    r,
                    f"{val:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(val) > vmax * 0.6 else "black",
                )

    # Draw value-type group separators
    if value_type_map:
        current_type = None
        for r, v in enumerate(all_values):
            vtype = value_type_map.get(v)
            if vtype != current_type and r > 0:
                ax.axhline(r - 0.5, color="white", linewidth=1.5)
            current_type = vtype

    plt.colorbar(im, ax=ax, label="β-susceptibility (perturbed - baseline)", shrink=0.6)
    ax.set_title("β-Susceptibility Heatmap\n(agent x Schwartz value)", fontsize=11)

    fig.tight_layout()
    return fig, ax


# ── Round-by-round propagation dynamics ──────────────────────────────────────


def plot_beta_timeseries(
    beta_timeseries: dict[str, dict[str, list[float]]],
    value_name: str,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot of β-susceptibility over rounds for a specific value.

    Args:
        beta_timeseries: metrics["beta_timeseries"] —
            dict[value_name -> dict[agent_name -> list[float]]].
        value_name: Which value to plot (e.g. "social_power").
        ax: Existing axes; creates new figure if None.

    Returns:
        (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    agents_data = beta_timeseries.get(value_name, {})
    if not agents_data:
        ax.text(0.5, 0.5, f"No data for '{value_name}'", transform=ax.transAxes, ha="center")
        return fig, ax

    for i, (agent, betas) in enumerate(sorted(agents_data.items())):
        rounds = range(1, len(betas) + 1)
        ax.plot(
            rounds,
            betas,
            marker="o",
            linewidth=2,
            markersize=5,
            color=AGENT_COLORS[i % len(AGENT_COLORS)],
            label=agent,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("β-susceptibility", fontsize=11)
    ax.set_title(
        f"β-Susceptibility Over Rounds\n(value: {value_name.replace('_', ' ')})",
        fontsize=11,
    )
    ax.legend(title="Agent", fontsize=8, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, ax


# ── H3: SS by Schwartz value type ────────────────────────────────────────────


def plot_ss_by_value_type(
    ss_by_value_type: dict[str, float],
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of mean SS grouped by Schwartz value type (H3).

    Args:
        ss_by_value_type: dict mapping value_type name to mean SS.
            Typically computed by averaging system_susceptibility over
            all values within each type.
        ax: Existing axes; creates new figure if None.

    Returns:
        (fig, ax) tuple.
    """
    # Order by VALUE_TYPE_ORDER, falling back to alphabetical
    ordered = sorted(
        ss_by_value_type.items(),
        key=lambda kv: (
            VALUE_TYPE_ORDER.index(kv[0]) if kv[0] in VALUE_TYPE_ORDER else 99,
            kv[0],
        ),
    )
    types = [k for k, _ in ordered]
    values = [v for _, v in ordered]
    colors = [VALUE_TYPE_COLORS.get(t, "#888888") for t in types]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()

    bars = ax.bar(types, values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Mean System Susceptibility by Schwartz Value Type (H3)", fontsize=11)
    ax.set_xlabel("Value Type", fontsize=11)
    ax.set_ylabel("Mean SS", fontsize=11)
    ax.set_xticklabels(types, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, max(values) * 1.25 if values else 1)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, ax


# ── H4: Perturbation location effect in chain ─────────────────────────────────


def plot_location_effect(
    ss_by_location: dict[str | int, float],
    n_agents: int = 5,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of SS by perturbation location in a chain topology (H4).

    Args:
        ss_by_location: dict mapping agent index (int or str) to SS value.
        n_agents: Total number of agents in the chain (for x-axis labelling).
        ax: Existing axes; creates new figure if None.

    Returns:
        (fig, ax) tuple.
    """
    # Normalise keys to int
    data = {int(k): v for k, v in ss_by_location.items()}
    indices = sorted(data.keys())
    ss_values = [data[i] for i in indices]
    labels = [
        f"Agent_{i}\n({'head' if i == 0 else 'tail' if i == n_agents - 1 else 'middle'})"
        for i in indices
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    bars = ax.bar(labels, ss_values, color="#457b9d", edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, ss_values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Annotate direction arrow
    ax.annotate(
        "← upstream       downstream →",
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        color="grey",
    )

    ax.set_title("SS by Perturbation Location in Chain (H4)", fontsize=11)
    ax.set_ylabel("System Susceptibility (SS)", fontsize=11)
    ax.set_ylim(0, max(ss_values) * 1.25 if ss_values else 1)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, ax


# ── Convenience: all four main figures in one call ────────────────────────────


def plot_summary_grid(
    topology_comparison: dict[str, dict[str, Any]],
    beta_susceptibility: dict[str, dict[str, float]],
    beta_timeseries: dict[str, dict[str, list[float]]],
    ss_by_value_type: dict[str, float],
    target_value: str = "social_power",
    value_type_map: dict[str, str] | None = None,
) -> plt.Figure:
    """Produce all four main paper figures as a 2x2 grid.

    Args:
        topology_comparison: Output of compute_cross_topology_comparison().
        beta_susceptibility: metrics["beta_susceptibility"].
        beta_timeseries: metrics["beta_timeseries"].
        ss_by_value_type: dict[value_type -> mean SS].
        target_value: Perturbed value name for timeseries panel.
        value_type_map: Optional dict[value_name -> value_type] for heatmap sorting.

    Returns:
        matplotlib Figure.
    """
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    plot_ss_by_topology(topology_comparison, target_value=target_value, ax=ax1)
    plot_beta_timeseries(beta_timeseries, value_name=target_value, ax=ax2)
    plot_ss_by_value_type(ss_by_value_type, ax=ax3)

    # Heatmap in its own figure (too tall for the grid); use a compact version
    # showing only the top-N values by absolute β
    top_n = 15
    flat_beta = {
        v: agents
        for v, agents in beta_susceptibility.items()
        if any(abs(b) > 0 for b in agents.values())
    }
    if len(flat_beta) > top_n:
        # Keep top N by mean abs β
        ranked = sorted(
            flat_beta.items(),
            key=lambda kv: np.mean([abs(b) for b in kv[1].values()]),
            reverse=True,
        )
        flat_beta = dict(ranked[:top_n])
    plot_beta_heatmap(flat_beta, value_type_map=value_type_map, ax=ax4)

    fig.suptitle("ValueFlow: Value Perturbation Propagation", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig
