#!/usr/bin/env python3
"""Analyze value shifts for nodes furthest from the perturbed agent.

This script supports two input styles:
1. An ``analysis.json`` produced by ``scripts/run_valueflow.py``.
2. A directory containing raw run outputs with ``.hydra/config.yaml`` files.

For each baseline+perturbed pair, it:
- rebuilds the topology graph
- finds the node(s) at maximum graph distance from the perturbed agent
- computes a per-run furthest-node statistic for a target value

The default statistic is mean absolute beta on the final recorded step:
    mean_i in furthest nodes |score_perturbed(i) - score_baseline(i)|

Alternative metrics are available through ``--metric``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenarios.valueflow.metrics import RunResults, compute_beta_susceptibility
from scenarios.valueflow.topology_metrics import build_topology_graph, to_undirected_adjacency


@dataclass
class RunPair:
    """A matched baseline and perturbed run pair."""

    topology: str
    num_nodes: int
    target_value: str
    perturbed_agent: str
    baseline_dir: Path
    perturbed_dir: Path
    baseline_label: str
    perturbed_label: str


@dataclass
class RunResult:
    """Computed furthest-node metric for one run pair."""

    topology: str
    num_nodes: int
    perturbed_agent: str
    target_value: str
    furthest_distance: int
    furthest_nodes: list[str]
    value: float
    baseline_label: str
    perturbed_label: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute value-shift statistics for nodes furthest from the perturbed agent "
            "using either analysis.json or raw output folders."
        )
    )
    parser.add_argument(
        "input_path",
        help="Path to analysis.json or to a directory containing raw run outputs.",
    )
    parser.add_argument(
        "--value-type",
        default="power",
        help="Value type to analyze (default: power).",
    )
    parser.add_argument(
        "--metric",
        choices=("abs_beta", "signed_beta", "step_delta", "abs_step_delta"),
        default="abs_beta",
        help=(
            "Per-run metric. "
            "abs_beta = mean |final perturbed - final baseline| at furthest nodes; "
            "signed_beta = mean(final perturbed - final baseline); "
            "step_delta = mean(step10 - step0 in perturbed run); "
            "abs_step_delta = mean |step10 - step0| in perturbed run."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the per-configuration summary table.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to save the per-run table as CSV.",
    )
    return parser.parse_args()


def load_pairs_from_analysis(analysis_path: Path) -> list[RunPair]:
    """Load run pairs from analysis.json."""
    records = json.loads(analysis_path.read_text())
    pairs: list[RunPair] = []
    for record in records:
        target_agent = str(record["target_agent"])
        baseline_dir = resolve_recorded_run_dir(str(record["baseline_dir"]))
        perturbed_dir = resolve_recorded_run_dir(str(record["perturbed_dir"]))
        num_nodes = len(record.get("value_scores_baseline", {}).get(record["target_value"], {}))
        if num_nodes == 0:
            num_nodes = len(record.get("beta_susceptibility", {}).get(record["target_value"], {})) + 1
        pairs.append(
            RunPair(
                topology=str(record["topology"]),
                num_nodes=num_nodes,
                target_value=str(record["target_value"]),
                perturbed_agent=target_agent,
                baseline_dir=baseline_dir,
                perturbed_dir=perturbed_dir,
                baseline_label=baseline_dir.name,
                perturbed_label=perturbed_dir.name,
            )
        )
    return pairs


def resolve_recorded_run_dir(recorded_path: str) -> Path:
    """Resolve a run directory recorded in analysis.json.

    Some older analysis.json files store only the trailing timestamp directory.
    When that happens, search under outputs/ for a unique matching directory.
    """
    direct = (ROOT / recorded_path).resolve()
    if direct.exists():
        return direct

    run_name = Path(recorded_path).name
    matches = [path for path in (ROOT / "outputs").rglob(run_name) if path.is_dir()]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        recorded_parts = Path(recorded_path).parts
        parent_hint = recorded_parts[-2] if len(recorded_parts) >= 2 else None
        if parent_hint is not None:
            hinted = [match for match in matches if parent_hint in match.parts]
            if len(hinted) == 1:
                return hinted[0].resolve()
    if not matches:
        raise FileNotFoundError(f"Could not resolve recorded run directory: {recorded_path}")
    raise FileNotFoundError(
        f"Recorded run directory '{recorded_path}' is ambiguous; matches: {matches}"
    )


def infer_num_nodes_from_scenario_name(scenario_name: str) -> int:
    """Infer node count from a scenario name."""
    if scenario_name == "valueflow_5_agents":
        return 5
    if scenario_name == "valueflow_15_agents":
        return 15
    match = re.search(r"_(\d+)_agents$", scenario_name)
    if not match:
        raise ValueError(f"Could not infer node count from scenario name '{scenario_name}'")
    return int(match.group(1))


def parse_run_config(config_path: Path) -> dict[str, str | int | bool]:
    """Extract topology and perturbation metadata from a Hydra config."""
    text = config_path.read_text()
    topology = re.search(r"\n  topology:\n    type: (.+)\n", text)
    perturbation = re.search(
        r"\n  perturbation:\n    enabled: (.+)\n    perturbed_agent_index: (\d+)\n",
        text,
    )
    target_value = re.search(r"\n    target_value: (.+)\n", text)
    scenario_name = re.search(r"\n  name: (valueflow_[^\n]+)\n", text)
    if not topology or not perturbation or not target_value or not scenario_name:
        raise ValueError(f"Could not parse required metadata from {config_path}")
    return {
        "topology": topology.group(1).strip(),
        "enabled": perturbation.group(1).strip() == "true",
        "perturbed_agent_index": int(perturbation.group(2)),
        "target_value": target_value.group(1).strip(),
        "scenario_name": scenario_name.group(1).strip(),
    }


def load_pairs_from_output_root(output_root: Path) -> list[RunPair]:
    """Load run pairs from raw output folders by pairing consecutive runs."""
    group_runs: dict[Path, list[tuple[Path, dict[str, str | int | bool]]]] = {}
    for config_path in output_root.rglob(".hydra/config.yaml"):
        run_dir = config_path.parent.parent
        group_dir = run_dir.parent
        metadata = parse_run_config(config_path)
        group_runs.setdefault(group_dir, []).append((run_dir, metadata))

    pairs: list[RunPair] = []
    for group_dir, runs in group_runs.items():
        ordered = sorted(runs, key=lambda item: item[0].name)
        for index in range(0, len(ordered), 2):
            if index + 1 >= len(ordered):
                continue
            baseline_dir, baseline_meta = ordered[index]
            perturbed_dir, perturbed_meta = ordered[index + 1]
            if baseline_meta["enabled"] or not perturbed_meta["enabled"]:
                continue
            if baseline_meta["topology"] != perturbed_meta["topology"]:
                continue
            if baseline_meta["scenario_name"] != perturbed_meta["scenario_name"]:
                continue

            pairs.append(
                RunPair(
                    topology=str(perturbed_meta["topology"]),
                    num_nodes=infer_num_nodes_from_scenario_name(str(perturbed_meta["scenario_name"])),
                    target_value=str(perturbed_meta["target_value"]),
                    perturbed_agent=f"Agent_{perturbed_meta['perturbed_agent_index']}",
                    baseline_dir=baseline_dir,
                    perturbed_dir=perturbed_dir,
                    baseline_label=baseline_dir.name,
                    perturbed_label=perturbed_dir.name,
                )
            )
    return pairs


def load_pairs(input_path: Path) -> list[RunPair]:
    """Load run pairs from supported inputs."""
    if input_path.is_file():
        if input_path.name != "analysis.json":
            raise ValueError("If input_path is a file, it must be an analysis.json")
        return load_pairs_from_analysis(input_path)
    if input_path.is_dir():
        analysis_path = input_path / "analysis.json"
        if analysis_path.exists():
            return load_pairs_from_analysis(analysis_path)
        return load_pairs_from_output_root(input_path)
    raise ValueError(f"Input path does not exist: {input_path}")


def furthest_nodes(topology: str, num_nodes: int, perturbed_agent: str) -> tuple[int, list[str]]:
    """Return graph distance and nodes furthest from the perturbed agent."""
    agent_names = [f"Agent_{idx}" for idx in range(num_nodes)]
    graph = to_undirected_adjacency(build_topology_graph(agent_names, topology))
    distances = {perturbed_agent: 0}
    queue: deque[str] = deque([perturbed_agent])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + 1
            queue.append(neighbor)
    max_distance = max(distance for node, distance in distances.items() if node != perturbed_agent)
    nodes = sorted(
        [node for node, distance in distances.items() if distance == max_distance and node != perturbed_agent],
        key=lambda node: int(node.split("_")[1]),
    )
    return max_distance, nodes


def compute_step_delta(run: RunResults, agent: str, value_type: str) -> float | None:
    """Compute step-10 minus step-0 for one agent and value type."""
    step0 = run.get_aggregated_value_score(agent, value_type, step=0)
    step10 = run.get_aggregated_value_score(agent, value_type, step=10)
    if step0 is None or step10 is None:
        return None
    return step10 - step0


def compute_run_result(pair: RunPair, metric: str, value_type: str) -> RunResult:
    """Compute the requested furthest-node metric for one run pair."""
    baseline = RunResults.from_jsonl(pair.baseline_dir / "probe_results.jsonl", condition="baseline")
    perturbed = RunResults.from_jsonl(pair.perturbed_dir / "probe_results.jsonl", condition="perturbed")

    max_distance, nodes = furthest_nodes(pair.topology, pair.num_nodes, pair.perturbed_agent)

    if metric in {"abs_beta", "signed_beta"}:
        beta = compute_beta_susceptibility(
            baseline,
            perturbed,
            pair.perturbed_agent,
            value_type,
        )
        values = [beta[node] for node in nodes if node in beta]
        if metric == "abs_beta":
            values = [abs(value) for value in values]
    else:
        values = []
        for node in nodes:
            delta = compute_step_delta(perturbed, node, value_type)
            if delta is None:
                continue
            values.append(abs(delta) if metric == "abs_step_delta" else delta)

    if not values:
        raise ValueError(
            f"Could not compute metric '{metric}' for {pair.perturbed_dir} and value '{value_type}'"
        )

    return RunResult(
        topology=pair.topology,
        num_nodes=pair.num_nodes,
        perturbed_agent=pair.perturbed_agent.replace("Agent_", "A"),
        target_value=value_type,
        furthest_distance=max_distance,
        furthest_nodes=[node.replace("Agent_", "A") for node in nodes],
        value=float(mean(values)),
        baseline_label=pair.baseline_label,
        perturbed_label=pair.perturbed_label,
    )


def summarize(results: list[RunResult]) -> list[dict[str, str | int | float]]:
    """Compute mean/std summaries by configuration."""
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for result in results:
        key = (result.topology, result.num_nodes, result.perturbed_agent)
        grouped.setdefault(key, []).append(result.value)

    summary_rows: list[dict[str, str | int | float]] = []
    for (topology, num_nodes, agent), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "topology": topology,
                "num_nodes": num_nodes,
                "perturbed_agent": agent,
                "runs": len(values),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return summary_rows


def print_per_run(results: list[RunResult], metric: str) -> None:
    """Print per-run results."""
    header = (
        "Run\tTopology\tNodes\tPerturbed Agent\tDistance\tFurthest Nodes\t"
        f"{metric}\tBaseline\tPerturbed"
    )
    print(header)
    for index, result in enumerate(results, start=1):
        print(
            f"{index}\t{result.topology}\t{result.num_nodes}\t{result.perturbed_agent}\t"
            f"{result.furthest_distance}\t{','.join(result.furthest_nodes)}\t"
            f"{result.value:.6f}\t{result.baseline_label}\t{result.perturbed_label}"
        )


def print_summary(summary_rows: list[dict[str, str | int | float]]) -> None:
    """Print summary rows."""
    print("\nSummary")
    print("Topology\tNodes\tPerturbed Agent\tRuns\tMean\tStd")
    for row in summary_rows:
        print(
            f"{row['topology']}\t{row['num_nodes']}\t{row['perturbed_agent']}\t"
            f"{row['runs']}\t{float(row['mean']):.6f}\t{float(row['std']):.6f}"
        )


def write_csv(results: list[RunResult], csv_path: Path) -> None:
    """Write per-run rows to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "topology",
                "num_nodes",
                "perturbed_agent",
                "target_value",
                "furthest_distance",
                "furthest_nodes",
                "value",
                "baseline_label",
                "perturbed_label",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.topology,
                    result.num_nodes,
                    result.perturbed_agent,
                    result.target_value,
                    result.furthest_distance,
                    ",".join(result.furthest_nodes),
                    f"{result.value:.6f}",
                    result.baseline_label,
                    result.perturbed_label,
                ]
            )


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    input_path = Path(args.input_path)
    pairs = load_pairs(input_path)
    results = [compute_run_result(pair, args.metric, args.value_type) for pair in pairs]
    if not args.summary_only:
        print_per_run(results, args.metric)
    summary_rows = summarize(results)
    print_summary(summary_rows)
    if args.csv:
        write_csv(results, Path(args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
