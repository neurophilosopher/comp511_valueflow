# ValueFlow Bug Fixes — Change Summary
# =====================================
# Two files changed:  scenarios/valueflow/metrics.py  and  scripts/run_valueflow.py
# One conceptual question answered inline.

# ─────────────────────────────────────────────────────────────────────────────
# Q: What is max_steps? Do I need it?
# ─────────────────────────────────────────────────────────────────────────────
# max_steps is passed from the Hydra config (simulation.execution.max_steps)
# through BaseSimulator.run() → Simulation.play() → engine.run_loop().
#
# ValueFlowEngine.run_loop() documents explicitly:
#   "max_steps: Ignored — num_rounds from interaction_config is used."
#
# So for *any* ValueFlow run, max_steps does nothing. The engine runs for
# exactly `scenario.interaction.num_rounds` rounds, period.
#
# The old run_valueflow.py was passing --max-steps 20 as a Hydra override,
# which had zero effect but was confusing. It has been removed from the
# build_hydra_command() call in the fixed version.
#
# You can leave max_steps in simulation.yaml / experiment.yaml as a safeguard
# for other scenarios that use the standard Sequential engine. For ValueFlow
# it is simply irrelevant.

# ─────────────────────────────────────────────────────────────────────────────
# Q: Where is probe_results.jsonl?
# ─────────────────────────────────────────────────────────────────────────────
# It is written by ProbeRunner into the experiment output directory, which
# Hydra sets to:
#   outputs/{scenario.name}_experiment/{YYYY-MM-DD_HH-MM-SS}/probe_results.jsonl
#
# The exact path is always printed to the terminal by run_experiment.py:
#   "Output Dir: outputs/valueflow_15_agents_experiment/2025-04-07_14-22-11"
# so probe_results.jsonl lives at that path + /probe_results.jsonl.
#
# run_valueflow.py reads "Output Dir:" from stdout to get this path.

# ─────────────────────────────────────────────────────────────────────────────
# Q: Single pair vs. run-5-pairs-together?
# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION: Keep single-pair runs (current approach) rather than a
# --pairs N flag that loops internally. Reasons:
#
#   1. A 15-agent, 10-round run with 21 probes × 2 steps is expensive.
#      If one out of 5 runs fails, you lose everything if they're coupled.
#
#   2. You want to inspect each pair's drift graph and decide whether a run
#      looks valid before adding it to the average.
#
#   3. The analysis.html accumulates records across separate invocations:
#      run the command once → get 1 pair appended; run again → 2 pairs, etc.
#      Mean ± std rows appear automatically once n ≥ 2.
#
#   4. You can reuse --baseline-dir to avoid re-running the baseline each time
#      (saves ~half the API cost), e.g.:
#        uv run python scripts/run_valueflow.py \
#          --scenario valueflow_15_agents \
#          --topologies small_world \
#          --rounds 10 \
#          --baseline-dir outputs/valueflow_15_agents_experiment/2025-04-07_14-22-11
#
# If you later want to batch 5 pairs automatically, it is trivial to add a
# --pairs N loop around the current main(), but it's not needed yet.
