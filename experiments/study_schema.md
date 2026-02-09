# Study Schema

A study is a self-contained investigation of a research question using simulation experiments. This document defines the directory layout, file formats, analysis pipeline, and notebook structure that all studies should follow.

## Directory Layout

```
experiments/
  {study_name}/
    study.yaml                          # Study-level metadata
    summary.json                        # Pre-aggregated results across all hypotheses
    {hypothesis_id}/
      hypothesis.yaml                   # Hypothesis definition
      analysis.json                     # Cross-condition comparison for this hypothesis
      {iv}={condition_a}/{scenario}/
        run_{timestamp}/
          config.yaml                   # Run configuration (frozen at launch)
          eval.json                     # Evaluation metrics for this run
      {iv}={condition_b}/{scenario}/
        run_{timestamp}/
          config.yaml
          eval.json

notebooks/
  study_{study_name}.ipynb              # Results notebook
```

### Naming conventions

| Element | Format | Example |
|---------|--------|---------|
| Study name | `snake_case` | `style_diversity` |
| Hypothesis ID | `h{N}_{short_name}` | `h1_model_capacity` |
| Condition directory | `{iv}={value}` | `model=gpt4o-mini` |
| Run directory | `run_{ISO timestamp}` | `run_2026-02-06T23-50-55` |
| Notebook file | `study_{study_name}.ipynb` | `study_style_diversity.ipynb` |

The `{iv}={value}` convention (inspired by Hive-style partitioning) makes the independent variable and its level readable from the path alone.

## File Formats

### study.yaml

Top-level metadata for the study. Lists which scenarios and hypotheses are included.

```yaml
name: style_diversity
question: >
  Does increasing LLM capacity reduce repetitive/groupthink behavior
  in multi-agent social media simulations?
scenarios:
  - ai_conference
  - misinformation
hypotheses:
  - h1_model_capacity
```

**Required fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique study identifier (matches directory name) |
| `question` | string | The research question in plain language |
| `scenarios` | list[string] | Scenario names used across all hypotheses |
| `hypotheses` | list[string] | Hypothesis IDs tested in this study |

### hypothesis.yaml

Defines a single hypothesis within the study, including the independent variable and its conditions.

```yaml
id: h1_model_capacity
statement: >
  Larger language models produce more diverse agent behavior
  (higher lexical diversity, lower self-BLEU, more varied actions).
independent_variable: model
prediction: gpt4o outperforms gpt4o-mini on diversity metrics across scenarios.
status: testing          # testing | supported | refuted | inconclusive
conditions:
  - gpt4o-mini
  - gpt4o
```

**Required fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Matches the directory name |
| `statement` | string | Falsifiable hypothesis in one sentence |
| `independent_variable` | string | The variable being manipulated (becomes the `{iv}` in directory names) |
| `prediction` | string | Expected outcome if hypothesis is true |
| `status` | enum | One of: `testing`, `supported`, `refuted`, `inconclusive` |
| `conditions` | list[string] | Values of the IV (become `{condition}` in directory names) |

### config.yaml

Frozen snapshot of the run configuration. Captures everything needed to reproduce the run.

```yaml
source: outputs/ai_conference_experiment/2026-02-07_09-43-11
model_name: gpt-4o
model_config: gpt4o
scenario: ai_conference
scenario_description: Simulates groupthink dynamics at an AI conference
max_steps: 10
seed: 42
condition: gpt4o
hypothesis: h1_model_capacity
```

**Required fields:**

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Path to the original simulation output |
| `model_name` | string | Actual model identifier used by the API |
| `model_config` | string | Config key from `config/model/` |
| `scenario` | string | Scenario name |
| `max_steps` | int | Number of simulation steps |
| `seed` | int | Random seed |
| `condition` | string | IV condition value |
| `hypothesis` | string | Hypothesis this run belongs to |

### eval.json

Per-run evaluation output. Contains three sections: per-agent metrics, aggregated metrics, and summary counts.

```json
{
  "checkpoint": "path/to/source/checkpoint.json",
  "agents": {
    "Agent Name": {
      "self_bleu": 0.05,
      "lexical_diversity": 0.45,
      ...
    }
  },
  "aggregated": {
    "self_bleu": 0.04,
    "lexical_diversity": 0.45,
    ...
    "inter_agent_distinctiveness": 0.36
  },
  "summary": {
    "total_posts": 96,
    "seed_posts": 15,
    "model_posts": 81,
    "replies": 60,
    "boosts": 2,
    "original_posts": 19,
    "total_actions": 90,
    "agents": 9,
    "steps": 9
  }
}
```

**Sections:**

| Section | Description |
|---------|-------------|
| `agents` | Per-agent metric dict. Keys are agent names; values are metric dicts. |
| `aggregated` | Mean across agents for each metric, plus any population-level metrics (e.g. `inter_agent_distinctiveness`). |
| `summary` | Integer counts: posts, actions, agents, steps. Used for sanity checks and action-type breakdowns. |

### summary.json

Pre-aggregated cross-condition comparison at the study level. Computed by the analysis script after all runs are evaluated.

```json
{
  "conditions": [
    {
      "hypothesis": "h1_model_capacity",
      "condition": "gpt4o-mini",
      "scenario": "ai_conference",
      "aggregated": { ... },
      "summary": { ... }
    }
  ],
  "metrics_by_condition": {
    "gpt4o-mini": { "self_bleu": 0.33, ... },
    "gpt4o": { "self_bleu": 0.04, ... }
  }
}
```

`metrics_by_condition` averages each metric across all scenarios for each condition. This is the primary data source for cross-condition comparison plots.

## Analysis Pipeline

The pipeline runs in three stages:

```
1. Simulate     uv run python -m src.main scenario=X model=Y
                 -> outputs/{scenario}_experiment/{timestamp}/

2. Evaluate      uv run python scripts/eval_style_diversity.py checkpoint.json -o eval.json
                 -> per-run eval.json with agents/aggregated/summary

3. Organize      uv run python scripts/experiment_organizer.py ...
                 -> experiments/{study}/ tree with config.yaml, eval.json, summary.json
```

Each stage is idempotent. Re-running the organizer regenerates `summary.json` and `analysis.json` from the collected `eval.json` files.

## Notebook Structure

The results notebook (`notebooks/study_{name}.ipynb`) follows a fixed 8-section structure. Each section serves a specific role in the analysis narrative.

### Section 1: Title + Setup
- **Type:** markdown + code
- **Content:** Study title, load `study.yaml` and `hypothesis.yaml`, load all `eval.json` files into a structured dict, load `summary.json`, set matplotlib defaults.
- **Output:** Print study name, question, hypothesis, number of files loaded.

### Section 2: Study Overview
- **Type:** markdown + code
- **Content:** Hypothesis statement, IV, prediction. Table of conditions showing: model, scenario, agents, steps, total posts, replies, originals, boosts.

### Section 3: Key Metrics Explained
- **Type:** markdown
- **Content:** For each key metric (typically 3-5): plain-language definition, display equation (labeled as *exact* or *intuitive* form), and a "why it matters" paragraph connecting the metric to the research question.

### Section 4: Headline Comparison
- **Type:** code + markdown
- **Plot:** Grouped bar chart of key metrics, values averaged across scenarios. Annotate direction (lower/higher = better). Add value labels on bars.
- **Narrative:** One-paragraph takeaway beneath the plot.

### Section 5: Full Metric Profile
- **Type:** code + markdown
- **Plot:** Radar/spider chart with all metrics, both conditions overlaid. Normalize so outward = better (flip repetition metrics via `1 - x`).
- **Narrative:** What the overall shape tells us; call out exceptions.

### Section 6: Scenario Consistency
- **Type:** code + markdown
- **Plot:** Faceted figure (one panel per scenario), each showing all metrics as grouped horizontal bars by condition.
- **Narrative:** Is the effect consistent across scenarios or scenario-dependent?

### Section 7: Per-Agent Distributions
- **Type:** code + markdown
- **Plot:** Strip/dot plots for key metrics, each agent as a point, colored by condition, pooled across scenarios. Mean markers. Print mean and std table.
- **Narrative:** Does the IV shift the mean, tighten variance, or both?

### Section 8: Behavioral Breakdown
- **Type:** code + markdown
- **Plot:** Stacked bar chart of action type counts (e.g. replies, originals, boosts) per condition, pooled across scenarios. Label segment counts.
- **Narrative:** Qualitative behavioral differences between conditions.

### Section 9: Takeaways
- **Type:** markdown
- **Content:** Bulleted key findings (with numbers), limitations (sample size, confounds), next steps.

### Conventions

- Load data via `../experiments/` relative path from `notebooks/`.
- Use `%matplotlib inline`.
- Working/exploratory style: default matplotlib theme, clear axis labels.
- Figures approximately 8x5 to 10x6 inches, 100 dpi.
- Colors: use a consistent two-color scheme for the two conditions throughout.

## Extending the Schema

### Adding a new hypothesis to an existing study

1. Create `experiments/{study}/{hypothesis_id}/hypothesis.yaml`.
2. Run simulations for each condition x scenario combination.
3. Evaluate each run and place `eval.json` + `config.yaml` in the appropriate `{iv}={condition}/{scenario}/run_{timestamp}/` directory.
4. Re-run the organizer to regenerate `summary.json`.
5. Add hypothesis-specific sections to the notebook or create a separate notebook.

### Adding a new study

1. Create `experiments/{study_name}/study.yaml`.
2. Create one or more hypothesis directories with `hypothesis.yaml`.
3. Run the simulate -> evaluate -> organize pipeline.
4. Create `notebooks/study_{study_name}.ipynb` following the notebook structure above.

### Adding replicate runs

Multiple `run_{timestamp}/` directories under the same `{iv}={condition}/{scenario}/` path represent replicate runs (e.g. different seeds). The analysis pipeline should average across replicates when computing `summary.json`, and the notebook should show replicate variance where available.
