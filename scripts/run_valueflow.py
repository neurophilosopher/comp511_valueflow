#!/usr/bin/env python3
"""ValueFlow experiment runner.

Orchestrates the full ValueFlow experimental pipeline:
1. Run baseline simulation (no perturbation)
2. Run perturbed simulation
3. Compute β-susceptibility and System Susceptibility (SS)
4. Print per-agent value scores and SS to terminal
5. Save metrics JSON + append a run record to the cumulative analysis HTML

The analysis HTML (analysis.html in --output-dir) accumulates one record
per run pair. Open it at any time to see all completed pairs with their
raw SS values. Once you have enough pairs, it will also show mean ± std
across pairs for each value type.

Usage:
    # Single pair, small-world topology (the typical use case right now)
    uv run python scripts/run_valueflow.py \\
        --scenario valueflow_15_agents \\
        --topologies small_world \\
        --rounds 10

    # Sweep topologies
    uv run python scripts/run_valueflow.py \\
        --scenario valueflow_15_agents \\
        --topologies community small_world \\
        --rounds 10

    # Dry run (print commands without executing)
    uv run python scripts/run_valueflow.py --dry-run

    # Reuse an existing baseline
    uv run python scripts/run_valueflow.py \\
        --baseline-dir outputs/<existing-baseline-timestamp>
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
DEFAULT_ROUNDS = 3          # overridden per scenario in the YAML; this is just the CLI default
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
    extra_overrides: list[str] | None = None,
) -> list[str]:
    """Build the Hydra CLI command for a single run.

    NOTE: max_steps is intentionally omitted here. ValueFlowEngine
    ignores max_steps entirely — it runs for exactly num_rounds rounds
    as set by scenario.interaction.num_rounds. Passing a max_steps
    override would have no effect and would be misleading.
    """
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
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)
    return cmd


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
        build_analysis_record,
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

    # Attach raw dirs to metrics for analysis record
    metrics["_baseline_dir"] = baseline_dir
    metrics["_perturbed_dir"] = perturbed_dir
    return metrics


# ── Analysis HTML ─────────────────────────────────────────────────────────────

def load_analysis_records(analysis_path: Path) -> list[dict]:
    """Load existing run records from the JSON sidecar next to analysis.html."""
    sidecar = analysis_path.with_suffix(".json")
    if not sidecar.exists():
        return []
    try:
        with sidecar.open() as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load analysis records from %s: %s", sidecar, e)
        return []


def save_analysis_records(analysis_path: Path, records: list[dict]) -> None:
    """Save run records to the JSON sidecar."""
    sidecar = analysis_path.with_suffix(".json")
    with sidecar.open("w") as f:
        json.dump(records, f, indent=2)


def build_analysis_html(records: list[dict]) -> str:
    """Build a self-contained HTML page summarising all completed run pairs.

    Each record = one baseline + perturbed pair. The page shows:
    - A table of per-run SS values for each Schwartz value type
    - Mean ± std across all recorded pairs (grows as you add more)
    - Δpert for each pair
    - Links to the baseline and perturbed run output directories
    """
    from scenarios.valueflow.metrics import SCHWARTZ_VALUE_TYPES
    import math

    n = len(records)

    # ── Aggregate SS per value type ────────────────────────────────────────
    ss_by_vt: dict[str, list[float]] = {vt: [] for vt in SCHWARTZ_VALUE_TYPES}
    for rec in records:
        ss = rec.get("system_susceptibility", {})
        for vt in SCHWARTZ_VALUE_TYPES:
            if vt in ss:
                ss_by_vt[vt].append(ss[vt])

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    def _std(vals: list[float]) -> float:
        if len(vals) < 2:
            return float("nan")
        m = _mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

    # ── Build topology/agent info string ──────────────────────────────────
    topologies = sorted({r.get("topology", "?") for r in records})
    agents = sorted({r.get("target_agent", "?") for r in records})

    # ── Per-run rows ───────────────────────────────────────────────────────
    run_rows = ""
    for i, rec in enumerate(records):
        ss = rec.get("system_susceptibility", {})
        dp = rec.get("delta_pert", float("nan"))
        label = rec.get("label", f"run_{i+1}")
        topo = rec.get("topology", "?")
        agent = rec.get("target_agent", "?")
        bl_dir = rec.get("baseline_dir", "")
        pt_dir = rec.get("perturbed_dir", "")

        cells = ""
        for vt in SCHWARTZ_VALUE_TYPES:
            val = ss.get(vt, float("nan"))
            target_vt = rec.get("target_value", "")
            highlight = ' class="target-vt"' if vt == target_vt else ""
            val_str = f"{val:.3f}" if not math.isnan(val) else "—"
            cells += f"<td{highlight}>{val_str}</td>"

        run_rows += f"""
        <tr>
          <td class="run-label">{i+1}</td>
          <td>{topo}</td>
          <td>{agent.replace('Agent_', 'A')}</td>
          <td class="dp">{"{:.3f}".format(dp) if not math.isnan(dp) else "—"}</td>
          {cells}
          <td class="dir-cell"><a href="{bl_dir}" title="{bl_dir}">baseline</a></td>
          <td class="dir-cell"><a href="{pt_dir}" title="{pt_dir}">perturbed</a></td>
        </tr>"""

    # ── Summary rows ────────────────────────────────────────────────────────
    mean_row = "<tr class='mean-row'><td colspan='4'>Mean</td>"
    std_row = "<tr class='std-row'><td colspan='4'>Std</td>"
    for vt in SCHWARTZ_VALUE_TYPES:
        vals = ss_by_vt[vt]
        m = _mean(vals)
        s = _std(vals)
        mean_row += f"<td>{m:.3f}</td>" if not math.isnan(m) else "<td>—</td>"
        std_row += f"<td>±{s:.3f}</td>" if not math.isnan(s) else "<td>—</td>"
    mean_row += "<td colspan='2'></td></tr>"
    std_row += "<td colspan='2'></td></tr>"

    # ── Value type header cells ─────────────────────────────────────────────
    vt_headers = "".join(f"<th>{vt}</th>" for vt in SCHWARTZ_VALUE_TYPES)

    status_badge = (
        f'<span class="badge badge-single">1 pair recorded — add more to get mean ± std</span>'
        if n == 1 else
        f'<span class="badge badge-multi">{n} pairs — mean ± std available</span>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ValueFlow Analysis — {", ".join(topologies)}</title>
  <style>
    :root {{
      --bg: #f7f5f0;
      --panel: #ffffff;
      --ink: #1a1a1a;
      --muted: #6b6b6b;
      --accent: #c0392b;
      --accent2: #2471a3;
      --target-bg: #fef9e7;
      --target-border: rgba(180,134,22,0.3);
      --mean-bg: #eaf4fb;
      --std-bg: #f4f9f4;
      --border: rgba(0,0,0,0.08);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--ink);
      font-family: "Georgia", "Times New Roman", serif;
      padding: 32px 24px 60px;
    }}
    .wrap {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ font-size: 28px; font-weight: 700; letter-spacing: -0.01em; margin-bottom: 6px; }}
    .subtitle {{ color: var(--muted); font-size: 14px; margin-bottom: 20px; font-style: italic; }}
    .meta {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; align-items: center; }}
    .badge {{
      display: inline-block;
      padding: 5px 12px;
      border-radius: 4px;
      font-size: 12px;
      font-family: ui-monospace, monospace;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .badge-single {{ background: #fef3cd; color: #856404; border: 1px solid #ffc107; }}
    .badge-multi  {{ background: #d4edda; color: #155724; border: 1px solid #28a745; }}
    .chip {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 12px;
      font-family: ui-monospace, monospace;
      background: rgba(0,0,0,0.06);
      border: 1px solid var(--border);
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.06);
      overflow: auto;
      margin-bottom: 24px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    th {{
      background: #2c3e50;
      color: #fff;
      padding: 9px 10px;
      text-align: right;
      white-space: nowrap;
      font-family: ui-monospace, monospace;
      font-size: 11px;
      letter-spacing: 0.03em;
    }}
    th.left {{ text-align: left; }}
    td {{
      padding: 7px 10px;
      text-align: right;
      border-bottom: 1px solid rgba(0,0,0,0.05);
      font-family: ui-monospace, monospace;
      font-size: 12px;
    }}
    td.run-label {{
      font-weight: 700;
      color: var(--accent2);
      text-align: center;
    }}
    td.dp {{ color: var(--muted); }}
    td.dir-cell {{ font-size: 11px; }}
    td.dir-cell a {{ color: var(--accent2); text-decoration: none; }}
    td.dir-cell a:hover {{ text-decoration: underline; }}
    tr:hover td {{ background: rgba(0,0,0,0.02); }}
    td.target-vt, th.target-vt {{
      background: var(--target-bg) !important;
      border-left: 2px solid var(--target-border);
      border-right: 2px solid var(--target-border);
    }}
    tr.mean-row td {{
      background: var(--mean-bg);
      font-weight: 700;
      border-top: 2px solid #2471a3;
      color: #1a5276;
    }}
    tr.std-row td {{
      background: var(--std-bg);
      color: var(--muted);
      font-size: 11px;
      border-bottom: 2px solid var(--border);
    }}
    .note {{
      font-size: 13px;
      color: var(--muted);
      font-style: italic;
      padding: 12px 16px;
      border-top: 1px solid var(--border);
      background: rgba(0,0,0,0.02);
    }}
    .section-title {{
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
      padding: 14px 16px 10px;
      border-bottom: 1px solid var(--border);
      font-family: ui-monospace, monospace;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>ValueFlow Cumulative Analysis</h1>
    <p class="subtitle">
      System Susceptibility (SS) per value type, normalized by Δpert. Each row = one baseline+perturbed pair.
    </p>
    <div class="meta">
      {status_badge}
      {"".join(f'<span class="chip">topology: {t}</span>' for t in topologies)}
      {"".join(f'<span class="chip">perturbed: {a.replace("Agent_", "A")}</span>' for a in agents)}
    </div>

    <div class="panel">
      <div class="section-title">Per-Run SS Results</div>
      <table>
        <thead>
          <tr>
            <th class="left">#</th>
            <th class="left">Topology</th>
            <th class="left">Agent</th>
            <th>Δpert</th>
            {vt_headers}
            <th>Baseline dir</th>
            <th>Perturbed dir</th>
          </tr>
        </thead>
        <tbody>
          {run_rows}
          {mean_row if n >= 2 else ""}
          {std_row if n >= 2 else ""}
        </tbody>
      </table>
      <p class="note">
        SS = mean |β_i| / Δpert across non-perturbed agents (Eq. 1, §3.4).
        Highlighted columns = perturbed value type.
        Mean ± std rows appear once you have ≥ 2 pairs.
      </p>
    </div>
  </div>
</body>
</html>"""


def append_run_to_analysis(
    analysis_path: Path,
    metrics: dict,
    topology: str,
    perturbed_agent: str,
    run_label: str,
    baseline_dir: str,
    perturbed_dir: str,
) -> None:
    """Append one run pair's result to the cumulative analysis HTML."""
    from scenarios.valueflow.metrics import build_analysis_record

    records = load_analysis_records(analysis_path)
    record = build_analysis_record(
        metrics=metrics,
        topology=topology,
        perturbed_agent=perturbed_agent,
        run_label=run_label,
        baseline_dir=baseline_dir,
        perturbed_dir=perturbed_dir,
    )
    records.append(record)
    save_analysis_records(analysis_path, records)

    # Rebuild and overwrite the HTML
    html = build_analysis_html(records)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(html, encoding="utf-8")
    print(f"\n📊 Analysis HTML updated: {analysis_path}  ({len(records)} pair(s) recorded)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the ValueFlow experiment pipeline."""
    parser = argparse.ArgumentParser(description="ValueFlow experiment runner")
    parser.add_argument(
        "--scenario", default=DEFAULT_SCENARIO,
        help="Scenario config name (e.g. valueflow_15_agents)",
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
        help=(
            "Number of interaction rounds. For valueflow_15_agents this is "
            "already set to 10 in the YAML; only pass this to override it."
        ),
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
        help="Base output directory for metrics and analysis HTML",
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
    print(f"Output:      {args.output_dir}")
    if args.baseline_dir:
        print(f"Baseline:    {args.baseline_dir} (reusing existing)")
    print("=" * 60)

    n_perturbed = len(args.topologies) * len(args.values) * len(args.locations)
    n_baseline = 0 if args.baseline_dir else len(args.topologies)
    print(f"\nTotal runs: {n_baseline} baseline + {n_perturbed} perturbed = {n_baseline + n_perturbed}")

    analysis_path = Path(args.output_dir) / "analysis.html"

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
            )
            baseline_dirs[topology] = run_experiment(baseline_cmd, dry_run=args.dry_run)

    # ── Step 2: Perturbed runs ────────────────────────────────────────────────
    print("\n▶ Step 2: Running perturbed conditions...")
    perturbed_runs: list[dict] = []

    for topology in args.topologies:
        for value in args.values:
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

            # ── Append to cumulative analysis HTML ────────────────────────
            append_run_to_analysis(
                analysis_path=analysis_path,
                metrics=metrics,
                topology=run["topology"],
                perturbed_agent=run["target_agent"],
                run_label=run["label"],
                baseline_dir=baseline_dir,
                perturbed_dir=run["output_dir"],
            )

    # ── Step 4: Save summary JSON ─────────────────────────────────────────────
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
                "delta_pert": m["delta_pert"],
                "target_value_ss": m["target_value_ss"],
            }
            for m in all_metrics
        ],
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"  Summary JSON:  {summary_path}")
    print(f"  Analysis HTML: {analysis_path}  ← open this in a browser")
    print(f"{'='*60}")

    # Final headline numbers
    if all_metrics:
        print("\n📊 Key Results (SS on target value):")
        for m in all_metrics:
            print(
                f"  {m.get('label', '?')}: "
                f"SS({m['target_value']}) = {m['target_value_ss']:.3f}  "
                f"(Δpert = {m['delta_pert']:.3f})"
            )


if __name__ == "__main__":
    main()
