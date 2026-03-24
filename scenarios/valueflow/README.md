# ValueFlow Scenario

Replication of [*ValueFlow: Measuring the Propagation of Value Perturbations in Multi-Agent LLM Systems*](https://arxiv.org/abs/2602.08567) (arxiv 2602.08567).

## Quick Start

```bash
# Run a single perturbed experiment (chain topology, social_power, agent 0)
uv run python run_experiment.py scenario=valueflow evaluation=valueflow environment=valueflow

# Run the full sweep (H1/H3/H4) and compute metrics automatically
uv run python scripts/run_valueflow.py --topologies chain ring star fully_connected

# Analyze results
# Open notebooks/study_valueflow.ipynb in VS Code, select kernel /home/mptouzel/venvs/simulator/bin/python
```

## How It Works

1. **Baseline run** — all 5 agents are neutral; Schwartz Value Survey probes fire after each of 3 rounds
2. **Perturbed run** — one agent's persona is overridden to strongly endorse a target value
3. **β-susceptibility** for non-perturbed agent *i*: `β_i(v) = score_perturbed - score_baseline`
4. **System Susceptibility (SS)** = mean |β| across all non-perturbed agents for the target value

## Common Modifications

### Change the number of agents
Edit `config/scenario/valueflow.yaml`:
```yaml
agents:
  entities:
    - ... # add/remove agent entries
```
Also update the chain length in H4 location sweep (`--locations 0 2 4` assumes 5 agents).

### Add a new value to test (H3 extension)
The value must exist in `scenarios/valueflow/data/schwartz_values.yaml`. Then:
```bash
uv run python scripts/run_valueflow.py \
  --values <new_value> \
  --baseline-dir outputs/valueflow_experiment/<existing-baseline-timestamp>
```
Add the value→type mapping to `VALUE_TYPE_MAP` in `scripts/run_valueflow.py` if not already there.

### Add a new topology (H1 extension)
Implement it in `scenarios/valueflow/game_masters.py::build_topology_graph()`, then:
```bash
uv run python scripts/run_valueflow.py --topologies <new_topology>
```

### Change perturbation strength
In `config/scenario/valueflow.yaml`:
```yaml
perturbation:
  strength: 9  # 1-10; higher = stronger persona override
```

### Add a new model (H2 extension)
1. Create `config/model/<name>.yaml` pointing to the new model
2. Run with `--model <name>` and a new `--output-dir` to keep results separate:
```bash
uv run python scripts/run_valueflow.py \
  --model <name> \
  --output-dir experiments/valueflow/results_<name>
```

### Re-run everything from scratch
```bash
# Full gpt-4o sweep (baseline + H1 + H3 + H4), ~4-6 hours, ~$15
uv run python scripts/run_valueflow.py \
  --topologies chain ring star fully_connected \
  --values social_power ambitious helpful equality respect_for_tradition social_order \
  --locations 0 2 4 \
  --output-dir experiments/valueflow/results_new
```

### Reuse an existing baseline
```bash
uv run python scripts/run_valueflow.py \
  --topologies ring star \
  --baseline-dir outputs/valueflow_experiment/<timestamp>
```

## File Map

| File | Purpose |
|------|---------|
| `config/scenario/valueflow.yaml` | Topology, perturbation, interaction config |
| `config/evaluation/valueflow.yaml` | 56 Schwartz value probes (JudgedNumericProbe) |
| `config/environment/valueflow.yaml` | ValueFlowEngine settings |
| `scenarios/valueflow/agents.py` | `ValueFlowAgent` prefab — neutral persona |
| `scenarios/valueflow/game_masters.py` | `ValueFlowGameMaster` — topology graph builder |
| `scenarios/valueflow/engine.py` | `ValueFlowEngine` — DAG-filtered observations, 3 rounds |
| `scenarios/valueflow/simulator.py` | `ValueFlowSimulator` — injects perturbation, wires judge |
| `scenarios/valueflow/metrics.py` | β-susceptibility, SS computation, JSON export |
| `scenarios/valueflow/plotting.py` | All visualization functions |
| `scenarios/valueflow/data/schwartz_values.yaml` | 56-value Schwartz dataset |
| `scripts/run_valueflow.py` | Sweep runner: baseline + perturbed + metrics |
| `scripts/judge_probe_results.py` | Backfill null probe values via gpt-4o-mini judge |
| `experiments/valueflow/study.yaml` | Study definition, hypothesis specs, result paths |
| `experiments/valueflow/results/` | gpt-4o metrics (one subdir per condition) |
| `experiments/valueflow/results_gpt4mini/` | gpt-4o-mini metrics |
| `notebooks/study_valueflow.ipynb` | Analysis notebook (H1-H5 + H2 model comparison) |

## Results Structure

```
experiments/valueflow/results/
  baseline/probe_results.jsonl          # raw probe scores, no perturbation
  chain__social_power__agent0/
    valueflow_metrics.json              # beta_susceptibility, system_susceptibility, target_value_ss
  ring__social_power__agent0/...
  ...
experiments/valueflow/results_gpt4mini/ # same structure for gpt-4o-mini
```

`valueflow_metrics.json` schema:
```json
{
  "target_agent": "Agent_0",
  "target_value": "social_power",
  "target_value_ss": 2.0,
  "beta_susceptibility": {"value_name": {"Agent_1": 3, "Agent_2": 1, ...}},
  "system_susceptibility": {"value_name": 1.5, ...},
  "beta_timeseries": {"value_name": {"Agent_1": [2, 3, 3], ...}}
}
```

## Experiment Status

| Hypothesis | Status | Key finding |
|-----------|--------|-------------|
| H1 topology | done (gpt-4o + mini) | fully_connected > chain > star > ring |
| H2 model | done (gpt-4o vs mini) | gpt-4o is 2.6x more susceptible |
| H3 value type | done (gpt-4o + mini) | helpful/equality propagate most |
| H4 location | done (gpt-4o + mini) | location has little effect |
| H5 baseline | done | high drift (52% > 0.5) — caveat |
