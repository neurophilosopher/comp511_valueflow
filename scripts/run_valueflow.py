#!/usr/bin/env python3
"""ValueFlow experiment runner.

Orchestrates the full ValueFlow experimental pipeline:
1. Run baseline simulation (no perturbation)
2. Run perturbed simulation(s)
3. Compute β-susceptibility and System Susceptibility
4. Save metrics and generate summary

Usage:
    # Run a single perturbation experiment (chain topology, social_power)
    uv run python scripts/run_valueflow.py

    # Sweep topologies
    uv run python scripts/run_valueflow.py --topologies chain ring star fully_connected

    # Sweep values
    uv run python scripts/run_valueflow.py --values social_power ambitious helpful

    # Sweep perturbation locations in chain
    uv run python scripts/run_valueflow.py --locations 0 2 4

    # Dry run (print commands without executing)
    uv run python scripts/run_valueflow.py --dry-run
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
DEFAULT_VALUES = ["social_power"]
DEFAULT_LOCATIONS = [0]
DEFAULT_MODEL = "gpt4o"
DEFAULT_ROUNDS = 3
DEFAULT_MAX_STEPS = 20


def build_hydra_command(
    scenario: str = "valueflow",
    evaluation: str = "valueflow",
    model: str = DEFAULT_MODEL,
    topology: str = "chain",
    target_value: str = "social_power",
    target_value_type: str = "power",
    perturbed_index: int = 0,
    perturbation_enabled: bool = True,
    num_rounds: int = DEFAULT_ROUNDS,
    max_steps: int = DEFAULT_MAX_STEPS,
    extra_overrides: list[str] | None = None,
) -> list[str]:
    """Build the Hydra CLI command for a single run.

    Args:
        scenario: Scenario name.
        evaluation: Evaluation config name.
        model: Model config name.
        topology: Topology type.
        target_value: Schwartz value to perturb.
        target_value_type: Higher-order Schwartz category.
        perturbed_index: Agent index to perturb.
        perturbation_enabled: Whether perturbation is active.
        num_rounds: Number of interaction rounds.
        max_steps: Maximum simulation steps.
        extra_overrides: Additional Hydra overrides.

    Returns:
        List of command-line arguments.
    """
    cmd = [
        os.path.expanduser(sys.executable),
        "run_experiment.py",
        f"scenario={scenario}",
        f"evaluation={evaluation}",
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
VALUE_TYPE_MAP = {
    "social_power": "power",
    "authority": "power",
    "wealth": "power",
    "successful": "achievement",
    "ambitious": "achievement",
    "influential": "achievement",
    "pleasure": "hedonism",
    "enjoying_life": "hedonism",
    "daring": "stimulation",
    "creativity": "self_direction",
    "freedom": "self_direction",
    "broadminded": "universalism",
    "equality": "universalism",
    "social_justice": "universalism",
    "helpful": "benevolence",
    "honest": "benevolence",
    "loyal": "benevolence",
    "devout": "tradition",
    "respect_for_tradition": "tradition",
    "humble": "tradition",
    "politeness": "conformity",
    "obedient": "conformity",
    "self_discipline": "conformity",
    "family_security": "security",
    "social_order": "security",
    "national_security": "security",
}


def run_experiment(cmd: list[str], dry_run: bool = False) -> str | None:
    """Run a single experiment and return the output directory.

    Args:
        cmd: Command-line arguments.
        dry_run: If True, print command without executing.

    Returns:
        Output directory path, or None on failure/dry-run.
    """
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
        # Try to extract output directory from stdout
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
    """Compute ValueFlow metrics from baseline and perturbed run outputs.

    Args:
        baseline_dir: Path to baseline run output directory.
        perturbed_dir: Path to perturbed run output directory.
        target_agent: Name of the perturbed agent.
        target_value: Schwartz value that was perturbed.
        output_dir: Where to save metrics.

    Returns:
        Metrics dict, or None on failure.
    """
    from scenarios.valueflow.metrics import (
        RunResults,
        compute_all_metrics,
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
    return metrics


def main() -> None:
    """Run the ValueFlow experiment pipeline."""
    parser = argparse.ArgumentParser(description="ValueFlow experiment runner")
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=DEFAULT_TOPOLOGIES,
        help="Topologies to sweep",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        default=DEFAULT_VALUES,
        help="Schwartz values to perturb",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        type=int,
        default=DEFAULT_LOCATIONS,
        help="Agent indices to perturb",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model config name",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="Number of interaction rounds",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum simulation steps",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/valueflow/results",
        help="Base output directory for metrics",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help=(
            "Path to an existing baseline run output directory. "
            "If provided, the baseline simulation is skipped and this directory "
            "is used directly. Useful when running multiple hypothesis sweeps "
            "that share the same baseline."
        ),
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("ValueFlow Experiment Pipeline")
    print("=" * 60)
    print(f"Topologies: {args.topologies}")
    print(f"Values: {args.values}")
    print(f"Locations: {args.locations}")
    print(f"Model: {args.model}")
    print(f"Rounds: {args.rounds}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output: {args.output_dir}")
    if args.baseline_dir:
        print(f"Baseline: {args.baseline_dir} (reusing existing)")
    print("=" * 60)

    # Count total experiments
    # baseline (1 unless --baseline-dir supplied) + perturbed (topologies x values x locations)
    n_perturbed = len(args.topologies) * len(args.values) * len(args.locations)
    n_baseline = 0 if args.baseline_dir else 1
    print(
        f"\nTotal runs: {n_baseline} baseline + {n_perturbed} perturbed = {n_baseline + n_perturbed}"
    )

    # Step 1: Run baseline (no perturbation), or reuse supplied dir
    if args.baseline_dir:
        print(f"\n▶ Step 1: Reusing existing baseline at {args.baseline_dir}")
        baseline_dir: str | None = args.baseline_dir
    else:
        print("\n▶ Step 1: Running baseline (no perturbation)...")
        baseline_cmd = build_hydra_command(
            model=args.model,
            perturbation_enabled=False,
            num_rounds=args.rounds,
            max_steps=args.max_steps,
        )
        baseline_dir = run_experiment(baseline_cmd, dry_run=args.dry_run)

    # Step 2: Run perturbed conditions
    print("\n▶ Step 2: Running perturbed conditions...")
    perturbed_runs: list[dict] = []

    for topology in args.topologies:
        for value in args.values:
            value_type = VALUE_TYPE_MAP.get(value, "power")
            for location in args.locations:
                agent_name = f"Agent_{location}"
                label = f"{topology}__{value}__agent{location}"

                cmd = build_hydra_command(
                    model=args.model,
                    topology=topology,
                    target_value=value,
                    target_value_type=value_type,
                    perturbed_index=location,
                    perturbation_enabled=True,
                    num_rounds=args.rounds,
                    max_steps=args.max_steps,
                )
                output = run_experiment(cmd, dry_run=args.dry_run)

                perturbed_runs.append(
                    {
                        "label": label,
                        "topology": topology,
                        "target_value": value,
                        "target_value_type": value_type,
                        "perturbed_index": location,
                        "target_agent": agent_name,
                        "output_dir": output,
                    }
                )

    if args.dry_run:
        print("\n[DRY RUN] Skipping metrics computation.")
        return

    # Step 3: Compute metrics
    print("\n▶ Step 3: Computing ValueFlow metrics...")

    if baseline_dir is None:
        logger.error("Baseline run failed — cannot compute metrics.")
        return

    all_metrics: list[dict] = []
    for run in perturbed_runs:
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

    # Step 4: Save summary
    print("\n▶ Step 4: Saving experiment summary...")
    summary_path = Path(args.output_dir) / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "study": "valueflow",
        "model": args.model,
        "num_runs": 1 + len(perturbed_runs),
        "baseline_dir": baseline_dir,
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

    # Print key results
    if all_metrics:
        print("\n📊 Key Results:")
        for m in all_metrics:
            print(
                f"  {m.get('label', '?')}: " f"SS({m['target_value']}) = {m['target_value_ss']:.3f}"
            )


if __name__ == "__main__":
    main()
