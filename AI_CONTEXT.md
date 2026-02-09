# AI_CONTEXT.md

This file provides guidance to AI coding assistants when working with code in this repository.

## Setup (after cloning)

```bash
uv sync                          # Install dependencies
uv run pre-commit install        # Enable git hooks for auto-validation on commit
```

## Build & Run Commands

```bash

# Run simulation with default config (marketplace scenario)
uv run python run_experiment.py

# Switch scenario/model/parameters
uv run python run_experiment.py scenario=debate model=claude
uv run python run_experiment.py simulation.execution.max_steps=50

# View resolved config without running
uv run python run_experiment.py --cfg job

# Quick test with mock models (no API calls)
uv run python run_experiment.py --quick-test
```

## Testing

```bash
# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_simulators/test_base_simulator.py

# Run specific test
uv run pytest tests/test_simulators/test_base_simulator.py::TestBaseSimulator::test_build_prefabs -v

# With coverage
uv run pytest --cov=src --cov=scenarios

# Skip slow/integration tests
uv run pytest -m "not slow and not integration"
```

## Code Quality

```bash
# Pre-commit (ruff, mypy, bandit) - auto-fixes formatting/linting
uv run pre-commit run --all-files --verbose

# Type checking only
uv run mypy src/ scenarios/

# Linting only
uv run ruff check src/ scenarios/
```

## Commit Workflow

**Always run pre-commit before committing:**

```bash
# 1. Stage changes
git add <files>

# 2. Run pre-commit (auto-fixes formatting, linting)
uv run pre-commit run --all-files --verbose

# 3. If hooks made fixes, re-stage and run again
git add <fixed-files>
uv run pre-commit run --all-files --verbose

# 4. Only manually fix errors that can't be auto-resolved (mypy, bandit)

# 5. Commit using commitizen for interactive conventional commit prompt
uv run cz c
```

**Auto-fix hooks**: ruff (linting), ruff-format, trailing-whitespace, end-of-file-fixer, mixed-line-ending

**Manual-fix hooks**: mypy (type errors), bandit (security issues)

**Commit types**: feat, fix, docs, style, refactor, perf, test, build, ci, chore

**Quick validation** (run before PRs):
```bash
uv run pre-commit run --all-files && uv run pytest
```

**Test suite**: 241 tests (run `uv run pytest` to verify)

**Coverage threshold**: 70% minimum (configured in pyproject.toml)

## Architecture Overview

**Hydra Configuration System**: All configs in `config/` compose via defaults in `experiment.yaml`. Override any value via CLI: `python run_experiment.py scenario.agents.entities[0].params.budget=2000`

**_target_ Instantiation Pattern**: Prefabs and models use Hydra's `_target_` for dynamic class loading:
```yaml
prefabs:
  buyer_agent:
    _target_: scenarios.marketplace.agents.BuyerAgent
```

**Key Extension Points**:
- New scenarios: Create `config/scenario/name.yaml` + optional `scenarios/name/` directory with agents, game_masters, knowledge, events modules
- New prefabs: Implement `concordia.typing.prefab.Prefab` interface
- Knowledge builders: Define `builders.knowledge.module` and `builders.knowledge.function` in scenario config
- New engines: Set `scenario.engine` to `sequential` (default), `simultaneous`, or `social_media`

**Data Flow**: `run_experiment.py` → `MultiModelSimulator` → `BaseSimulator.build_config()` → `Simulation.play()`

**Social Media Environment** (`src/environments/social_media/`):
- `app.py`: `Post` dataclass + `SocialMediaApp` (in-memory platform: post, reply, like, boost, follow)
- `engine.py`: `SocialMediaEngine` (parallel agent loop, action parsing)
- `game_master.py`: `SocialMediaGameMaster` prefab
- `analysis.py`: Transmission chain extraction, keyword overlap, network analysis
- Activated by `engine: social_media` in scenario config

## Key Patterns

- Scenario configs define `prefabs` mapping names to `_target_` classes (or legacy string paths)
- `BaseSimulator.build_instances()` creates entity configs from `scenario.agents.entities[]`
- Knowledge injection uses `formative_memories_initializer__GameMaster` from Concordia
- Entity-to-model mapping via `model.entity_model_mapping` or `_default_` key

## Scenario Isolation

**All scenario-specific code must be contained within:**
- `config/scenario/<name>.yaml` - scenario configuration
- `scenarios/<name>/` - implementation (agents, game_masters, knowledge, events, conftest.py)

**Keep these files scenario-agnostic:**
- `src/` - core framework code, no scenario-specific references
- `tests/conftest.py` - generic fixtures using `BasicEntity`/`BasicGameMaster`
- `config/evaluation/basic_metrics.yaml` - generic metrics only

**Scenario-specific test fixtures** go in `scenarios/<name>/conftest.py`:
```python
# scenarios/marketplace/conftest.py
@pytest.fixture
def marketplace_config() -> DictConfig:
    ...
```

**To use scenario fixtures in tests**, add pytest_plugins:
```python
# tests/test_simulators/test_something.py
pytest_plugins = ["scenarios.marketplace.conftest"]
```

**Scenario-specific evaluation metrics** go in `config/evaluation/<name>.yaml`. Use explicit override when running:
```bash
uv run python run_experiment.py scenario=election evaluation=election
```

## Evaluation Probes

The evaluation system queries agents at checkpoints without affecting their memory:

- **Probes call `agent.act()` but NOT `agent.observe()`** - responses don't enter agent memory
- **Config-driven**: Define metrics in `config/evaluation/*.yaml`
- **Role-based filtering**: Probes can target specific roles (e.g., `applies_to: [voter]`)
- **Output**: Results saved to `probe_results.jsonl` in experiment output directory

**Probe types:**
- `CategoricalProbe`: Choose from predefined categories
- `NumericProbe`: Rating within min/max range
- `BooleanProbe`: Yes/no questions

**Example metric config:**
```yaml
metrics:
  vote_preference:
    type: categorical
    categories: [conservative, progressive, undecided]
    prompt_template: |
      Based on {agent_name}'s views, which candidate do they prefer?
    applies_to: [voter]
```

## Concordia Customizations

**Do NOT modify files in the `concordia/` submodule directly.** Instead, create custom implementations in `src/` and monkey-patch at runtime.

**Custom engine utilities** in `src/simulation/engines/engine_utils.py`:
- `action_spec_parser`: Fixed version that handles LLM responses returning non-dict JSON
- `patch_concordia_parser()`: Called during `BaseSimulator.setup()` to apply the fix

**Pattern for adding Concordia fixes:**
1. Create the fix in `src/simulation/engines/` or appropriate `src/` location
2. Export from `__init__.py`
3. Apply via monkey-patch in `BaseSimulator.setup()` or similar initialization point
4. Document in this section

## Studies & Experiments

When the user asks to start, organize, or analyze a study, follow the schema defined in **[`experiments/study_schema.md`](experiments/study_schema.md)**. It specifies the directory layout (`experiments/{study}/`), required YAML/JSON file formats, the 3-stage pipeline (simulate -> evaluate -> organize), and the standard results notebook structure (`notebooks/study_{name}.ipynb`).

## Conventions

- Commits: Use conventional commits format (`feat:`, `fix:`, `refactor:`)
- Python 3.12+, type hints required, ruff enforces style
- Absolute imports only (`from src.utils...` not `from ..utils...`)

## Workflow Reminder

**IMPORTANT**: When adding a feature, update AI_CONTEXT.md if the change affects architecture, commands, or patterns.
