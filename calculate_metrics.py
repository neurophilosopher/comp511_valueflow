import json
from pathlib import Path
from scenarios.valueflow.metrics import RunResults, compute_all_metrics, save_metrics

# Paths to your specific data
BASELINE_DIR = Path("outputs/valueflow_15_agents_experiment/2026-04-07_11-34-51")
PERTURBED_DIR = Path("outputs/valueflow_15_agents_experiment/2026-04-07_12-12-48")

def main():
    print("Loading baseline results...")
    baseline = RunResults.from_jsonl(BASELINE_DIR / "probe_results.jsonl")
    
    print("Loading perturbed results...")
    perturbed = RunResults.from_jsonl(PERTURBED_DIR / "probe_results.jsonl")

    # The target agent and value for your experiment
    # Based on your previous commands: agent_0 and power
    target_agent = "Agent_0"
    target_value = "power"

    print(f"Computing metrics for {target_agent} with target value '{target_value}'...")
    metrics = compute_all_metrics(
        baseline=baseline,
        perturbed=perturbed,
        target_agent=target_agent,
        target_value=target_value
    )

    print(f"Saving metrics to {PERTURBED_DIR}...")
    save_metrics(metrics, PERTURBED_DIR)
    print("Done! valueflow_metrics.json has been created.")

if __name__ == "__main__":
    main()