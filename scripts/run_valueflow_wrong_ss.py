#!/usr/bin/env python3
"""ValueFlow experiment runner.

Orchestrates the full ValueFlow experimental pipeline:
1. Run baseline simulation (no perturbation)
2. Run perturbed simulation(s)
3. Compute β-susceptibility and System Susceptibility (SS)
4. Print per-agent value scores and SS to terminal
5. Save metrics and generate summary

Usage:
    # Run a single perturbation experiment (chain topology, power)
    uv run python scripts/run_valueflow.py

    # Sweep topologies
    uv run python scripts/run_valueflow.py --topologies chain ring star fully_connected

    # Sweep values
    uv run python scripts/run_valueflow.py --values power ambitious helpful

    # Sweep perturbation locations in chain
    uv run python scripts/run_valueflow.py --locations 0 2 4

    # Dry run (print commands without executing)
    uv run python scripts/run_valueflow.py --dry-run

    # Use the 21-question Schwartz evaluation
    uv run python scripts/run_valueflow.py --evaluation valueflow_schwartz21
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Default experiment parameters
DEFAULT_TOPOLOGIES = ["chain"]
DEFAULT_VALUES = ["power"]
DEFAULT_LOCATIONS = [0]
DEFAULT_MODEL = "gpt4o"
DEFAULT_ROUNDS = 3
DEFAULT_MAX_STEPS = 20
DEFAULT_EVALUATION = "valueflow_schwartz21"
DEFAULT_SCENARIO = "valueflow"


def build_hydra_command(
    scenario: str = DEFAULT_SCENARIO,
    evaluation: str = DEFAULT_EVALUATION,
    model: str = DEFAULT_MODEL,
    topology: str = "chain",
    target_value: str = "power",
    target_value_type: str = "power",
    perturbed_index: int = 0,
    perturbation_enabled: bool = True,
    num_rounds: int = DEFAULT_ROUNDS,
    max_steps: int = DEFAULT_MAX_STEPS,
    extra_overrides: list[str] | None = None,
) -> list[str]:
    """Build the Hydra CLI command for a single run."""
    cmd = [
        os.path.expanduser(sys.executable),
        "run_experiment.py",
        f"scenario={scenario}",
        f"evaluation={evaluation}",
        f"environment=valueflow",
        f"model={model}",
        f"scenario.topology.type={topology}",
        f"scenario.perturbation.enabled={str(perturbation_enabled).lower()}",
        f"scenario.perturbation.target_value={target_value}",
        f"scenario.perturbation.target_value_type={target_value_type}",
        f"scenario.perturbation.perturbed_agent_index={perturbed_index}",
        f"scenario.interaction.num_rounds={num_rounds}",
        f"simulation.execution.max_steps={max_steps}",
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)
    return cmd


# Map value names to their Schwartz type
# VALUE_TYPE_MAP = {
#     "social_power": "power",
#     "authority": "power",
#     "wealth": "power",
#     "successful": "achievement",
#     "ambitious": "achievement",
#     "influential": "achievement",
#     "pleasure": "hedonism",
#     "enjoying_life": "hedonism",
#     "daring": "stimulation",
#     "creativity": "self_direction",
#     "freedom": "self_direction",
#     "broadminded": "universalism",
#     "equality": "universalism",
#     "social_justice": "universalism",
#     "helpful": "benevolence",
#     "honest": "benevolence",
#     "loyal": "benevolence",
#     "devout": "tradition",
#     "respect_for_tradition": "tradition",
#     "humble": "tradition",
#     "politeness": "conformity",
#     "obedient": "conformity",
#     "self_discipline": "conformity",
#     "family_security": "security",
#     "social_order": "security",
#     "national_security": "security",
# }


def run_experiment(cmd: list[str], dry_run: bool = False) -> str | None:
    """Run a single experiment and return the output directory."""
    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"[DRY RUN] {cmd_str}")
        return None

    print(f"\n{'='*60}")
    print(f"Running: {cmd_str}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.split("\n"):
            if "Output Dir:" in line:
                return line.split("Output Dir:")[-1].strip()
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        return None


def compute_metrics_from_runs(
    baseline_dir: str,
    perturbed_dir: str,
    target_agent: str,
    target_value: str,
    output_dir: str,
) -> dict | None:
    """Compute ValueFlow metrics and print results to terminal."""
    from scenarios.valueflow.metrics import (
        RunResults,
        build_html_results_block,
        compute_all_metrics,
        print_ss_results,
        print_value_scores,
        save_metrics,
    )

    baseline_path = Path(baseline_dir) / "probe_results.jsonl"
    perturbed_path = Path(perturbed_dir) / "probe_results.jsonl"

    if not baseline_path.exists():
        logger.error(f"Baseline probe results not found: {baseline_path}")
        return None
    if not perturbed_path.exists():
        logger.error(f"Perturbed probe results not found: {perturbed_path}")
        return None

    baseline = RunResults.from_jsonl(baseline_path, condition="baseline")
    perturbed = RunResults.from_jsonl(perturbed_path, condition="perturbed")

    metrics = compute_all_metrics(
        baseline=baseline,
        perturbed=perturbed,
        target_agent=target_agent,
        target_value=target_value,
    )

    save_metrics(metrics, Path(output_dir))

    # ── Print value scores for both runs ──────────────────────────────────
    print_value_scores(baseline, title="Value Scores — Baseline Run")
    print_value_scores(perturbed, title="Value Scores — Perturbed Run")

    # ── Print SS table ─────────────────────────────────────────────────────
    print_ss_results(metrics)

    # ── Append SS results to perturbed run's HTML ──────────────────────────
    perturbed_html = Path(perturbed_dir) / "simulation_log.html"
    if perturbed_html.exists():
        try:
            ss_html = build_html_results_block(
                perturbed,
                title="Value Scores — Perturbed Run",
                metrics=metrics,
            )
            with perturbed_html.open("a", encoding="utf-8") as f:
                f.write("\n<!-- ValueFlow SS Results -->\n")
                f.write(ss_html)
        except Exception as e:
            logger.warning(f"Failed to append SS HTML: {e}")

    return metrics


def main() -> None:
    """Run the ValueFlow experiment pipeline."""
    parser = argparse.ArgumentParser(description="ValueFlow experiment runner")
    parser.add_argument(
        "--scenario", default=DEFAULT_SCENARIO,
        help="Scenario config name",
    )
    parser.add_argument(
        "--topologies", nargs="+", default=DEFAULT_TOPOLOGIES,
        help="Topologies to sweep",
    )
    parser.add_argument(
        "--values", nargs="+", default=DEFAULT_VALUES,
        help="Schwartz values to perturb",
    )
    parser.add_argument(
        "--locations", nargs="+", type=int, default=DEFAULT_LOCATIONS,
        help="Agent indices to perturb",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Model config name",
    )
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS,
        help="Number of interaction rounds",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help="Maximum simulation steps",
    )
    parser.add_argument(
        "--evaluation", default=DEFAULT_EVALUATION,
        help="Evaluation config name (default: valueflow_schwartz21)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--output-dir", default="experiments/valueflow/results",
        help="Base output directory for metrics",
    )
    parser.add_argument(
        "--baseline-dir", default=None,
        help="Reuse an existing baseline run output directory.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("ValueFlow Experiment Pipeline")
    print("=" * 60)
    print(f"Scenario:    {args.scenario}")
    print(f"Topologies:  {args.topologies}")
    print(f"Values:      {args.values}")
    print(f"Locations:   {args.locations}")
    print(f"Model:       {args.model}")
    print(f"Evaluation:  {args.evaluation}")
    print(f"Rounds:      {args.rounds}")
    print(f"Max steps:   {args.max_steps}")
    print(f"Output:      {args.output_dir}")
    if args.baseline_dir:
        print(f"Baseline:    {args.baseline_dir} (reusing existing)")
    print("=" * 60)

    n_perturbed = len(args.topologies) * len(args.values) * len(args.locations)
    n_baseline = 0 if args.baseline_dir else len(args.topologies)
    print(f"\nTotal runs: {n_baseline} baseline + {n_perturbed} perturbed = {n_baseline + n_perturbed}")

    # ── Step 1: Baselines ────────────────────────────────────────────────────
    baseline_dirs: dict[str, str | None] = {}
    if args.baseline_dir:
        print(f"\n▶ Step 1: Reusing existing baseline at {args.baseline_dir}")
        for topology in args.topologies:
            baseline_dirs[topology] = args.baseline_dir
    else:
        print("\n▶ Step 1: Running baseline (no perturbation) per topology...")
        for topology in args.topologies:
            print(f"\n  • Baseline topology: {topology}")
            baseline_cmd = build_hydra_command(
                scenario=args.scenario,
                evaluation=args.evaluation,
                model=args.model,
                topology=topology,
                perturbation_enabled=False,
                num_rounds=args.rounds,
                max_steps=args.max_steps,
            )
            baseline_dirs[topology] = run_experiment(baseline_cmd, dry_run=args.dry_run)

    # ── Step 2: Perturbed runs ────────────────────────────────────────────────
    print("\n▶ Step 2: Running perturbed conditions...")
    perturbed_runs: list[dict] = []

    for topology in args.topologies:
        for value in args.values:
            # value_type = VALUE_TYPE_MAP.get(value, "power")
            for location in args.locations:
                agent_name = f"Agent_{location}"
                label = f"{topology}__{value}__agent{location}"

                cmd = build_hydra_command(
                    scenario=args.scenario,
                    evaluation=args.evaluation,
                    model=args.model,
                    topology=topology,
                    target_value=value,
                    target_value_type=value,
                    perturbed_index=location,
                    perturbation_enabled=True,
                    num_rounds=args.rounds,
                    max_steps=args.max_steps,
                )
                output = run_experiment(cmd, dry_run=args.dry_run)

                perturbed_runs.append({
                    "label": label,
                    "topology": topology,
                    "target_value": value,
                    "target_value_type": value,
                    "perturbed_index": location,
                    "target_agent": agent_name,
                    "output_dir": output,
                })

    if args.dry_run:
        print("\n[DRY RUN] Skipping metrics computation.")
        return

    # ── Step 3: Compute metrics and print results ─────────────────────────────
    print("\n▶ Step 3: Computing ValueFlow metrics and printing results...")

    if any(baseline_dirs.get(topology) is None for topology in args.topologies):
        logger.error("One or more baseline runs failed — cannot compute metrics for those topologies.")

    all_metrics: list[dict] = []
    for run in perturbed_runs:
        baseline_dir = baseline_dirs.get(run["topology"])
        if baseline_dir is None:
            logger.warning(f"Skipping {run['label']}: baseline for topology {run['topology']} failed")
            continue
        if run["output_dir"] is None:
            logger.warning(f"Skipping {run['label']}: run failed")
            continue

        metrics_dir = Path(args.output_dir) / run["label"]
        metrics = compute_metrics_from_runs(
            baseline_dir=baseline_dir,
            perturbed_dir=run["output_dir"],
            target_agent=run["target_agent"],
            target_value=run["target_value"],
            output_dir=str(metrics_dir),
        )
        if metrics:
            metrics["label"] = run["label"]
            metrics["topology"] = run["topology"]
            all_metrics.append(metrics)

    # ── Step 4: Save summary ──────────────────────────────────────────────────
    print("\n▶ Step 4: Saving experiment summary...")
    summary_path = Path(args.output_dir) / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "study": "valueflow",
        "model": args.model,
        "evaluation": args.evaluation,
        "num_runs": len(baseline_dirs) + len(perturbed_runs),
        "baseline_dirs": baseline_dirs,
        "conditions": [
            {
                "label": m.get("label", ""),
                "topology": m.get("topology", ""),
                "target_value": m["target_value"],
                "target_agent": m["target_agent"],
                "target_value_ss": m["target_value_ss"],
            }
            for m in all_metrics
        ],
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment complete! Summary: {summary_path}")
    print(f"{'='*60}")

    # Final headline numbers
    if all_metrics:
        print("\n📊 Key Results (SS on target value):")
        for m in all_metrics:
            print(
                f"  {m.get('label', '?')}: "
                f"SS({m['target_value']}) = {m['target_value_ss']:.3f}"
            )


if __name__ == "__main__":
    main()
