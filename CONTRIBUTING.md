# Contributing to Concordia Sim

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Git

### Getting Started

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/concordia-sim.git
cd concordia-sim/simulator
```

2. Install dependencies:
```bash
# Using uv (recommended)
uv sync --all-extras

# Using pip
pip install -e ".[all]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

4. Copy environment file:
```bash
cp .env.example .env
# Add your API keys
```

## Code Style

### Python Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

- Line length: 100 characters
- Use type hints for all function signatures
- Follow PEP 8 conventions
- Use descriptive variable names

### Pre-commit Hooks

Hooks run automatically on commit:
- `ruff` - Linting and formatting
- `mypy` - Type checking
- `bandit` - Security checks
- `commitizen` - Commit message validation

Run manually:
```bash
pre-commit run --all-files
```

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```bash
feat(scenario): add prisoner's dilemma scenario
fix(agent): correct memory retrieval order
docs(readme): update installation instructions
test(simulation): add multi-model integration tests
```

Use commitizen for guided commits:
```bash
cz commit
```

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov=scenarios --cov-report=html

# Specific test file
pytest tests/test_agents/test_basic_entity.py -v

# Skip slow tests
pytest -m "not slow"

# Only integration tests
pytest -m integration
```

### Writing Tests

- Place tests in `tests/` mirroring source structure
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures from `conftest.py`
- Mark slow tests with `@pytest.mark.slow`
- Mark integration tests with `@pytest.mark.integration`

Example:
```python
import pytest
from src.entities.agents.basic_entity import BasicEntity

class TestBasicEntity:
    def test_build_creates_agent(self, mock_model, mock_memory_bank):
        prefab = BasicEntity()
        prefab.params = {"name": "TestAgent", "goal": "Test goal"}

        agent = prefab.build(mock_model, mock_memory_bank)

        assert agent.name == "TestAgent"

    @pytest.mark.slow
    def test_agent_with_real_embeddings(self, real_embedder):
        # Slow test using real sentence transformer
        ...
```

## Architecture Guidelines

### Adding a New Scenario

1. Create directory: `scenarios/my_scenario/`
2. Required files:
   - `__init__.py` - Exports
   - `agents.py` - Agent prefabs
   - `game_masters.py` - GM prefabs
   - `knowledge.py` - Knowledge builder function
   - `events.py` - Event generator function
   - `data/knowledge.yaml` - Static knowledge

3. Create config: `config/scenario/my_scenario.yaml`

4. Add tests: `tests/test_scenarios/test_my_scenario.py`

### Prefab Pattern

Follow Concordia v2 prefab patterns:

```python
@dataclasses.dataclass
class MyAgent(prefab_lib.Prefab):
    description: str = "Description of this agent type"

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Agent",
            "goal": "",
            # scenario-specific params
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: AssociativeMemoryBank,
    ) -> EntityAgentWithLogging:
        # 1. Extract params
        name = self.params.get("name", "Agent")

        # 2. Create memory component
        memory = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank
        )

        # 3. Create other components
        # ...

        # 4. Create act component
        act_component = ConcatActComponent(
            model=model,
            component_order=list(components.keys()),
        )

        # 5. Return agent
        return EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )
```

### Configuration Guidelines

- Use descriptive keys
- Provide sensible defaults
- Document all parameters in comments
- Use interpolation for DRY configs

## Pull Request Process

1. Create a feature branch:
```bash
git checkout -b feat/my-feature
```

2. Make changes and commit:
```bash
git add .
cz commit
```

3. Ensure all checks pass:
```bash
pre-commit run --all-files
pytest
```

4. Push and create PR:
```bash
git push origin feat/my-feature
```

5. Fill out PR template with:
   - Description of changes
   - Related issues
   - Testing performed
   - Breaking changes (if any)

6. Request review

## Issue Reporting

### Bug Reports

Include:
- Python version
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/tracebacks
- Minimal reproducible example

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches considered
- Impact on existing functionality

## Questions?

- Open a [Discussion](https://github.com/yourusername/concordia-sim/discussions)
- Check existing [Issues](https://github.com/yourusername/concordia-sim/issues)

Thank you for contributing!
