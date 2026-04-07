import json
import math
from pathlib import Path
import argparse

# Import the specific functions from your existing script
try:
    from scripts.run_valueflow import append_run_to_analysis
except ImportError:
    print("Error: Could not find scripts/run_valueflow.py. Ensure you are in the project root.")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Rebuild analysis.html from existing experiment data.")
    parser.add_name = "rebuild_analysis"
    
    # Required paths for your specific runs
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline folder")
    parser.add_argument("--perturbed", type=str, required=True, help="Path to perturbed folder")
    
    # Metadata for the table
    parser.add_argument("--scenario", type=str, default="valueflow_15_agents")
    parser.add_argument("--topology", type=str, default="small_world")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=str, default="analysis.html")

    args = parser.parse_args()

    # 1. Locate the metrics file
    perturbed_path = Path(args.perturbed)
    metrics_file = perturbed_path / "valueflow_metrics.json"

    if not metrics_file.exists():
        print(f"Error: Could not find {metrics_file}. Did the metrics calculation finish?")
        return

    # 2. Load the calculated data
    print(f"Loading metrics from {args.perturbed}...")
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # 3. Call the fixed append function to build/update the HTML
    print(f"Updating {args.output}...")
    try:
        
        
        # Convert the string path to a Path object to avoid the 'with_suffix' error
        analysis_path_obj = Path(args.output)

        append_run_to_analysis(
            analysis_path=analysis_path_obj,  # Use the Path object here
            baseline_dir=args.baseline,
            perturbed_dir=args.perturbed,
            metrics=metrics,
            topology=args.topology,
            perturbed_agent="Agent_0",
            run_label="Manual Rebuild"
        )
        print("Successfully rebuilt analysis.html!")
    except Exception as e:
        print(f"Failed to build HTML. Did you fix the f-string bug at line 275? Error: {e}")

if __name__ == "__main__":
    main()