# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install dependencies (uv recommended)
uv sync
# Or: pip install -e ".[dev]"

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

**Data Flow**: `run_experiment.py` → `MultiModelSimulator` → `BaseSimulator.build_config()` → `Simulation.play()`

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

**Scenario-specific evaluation metrics** go in `scenarios/<name>/evaluation.yaml`

## Conventions

- Commits: Use conventional commits format (`feat:`, `fix:`, `refactor:`)
- Python 3.12+, type hints required, ruff enforces style
- Absolute imports only (`from src.utils...` not `from ..utils...`)

## Workflow Reminder

**IMPORTANT**: When the user asks to add a feature, remind them to update CLAUDE.md if the change affects architecture, commands, or patterns. Prompt: "Should I update CLAUDE.md with this new pattern/feature?"
