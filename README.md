# Concordia v2 Multi-Agent Simulation Framework

A flexible multi-agent simulation framework built on [Concordia v2](https://github.com/google-deepmind/concordia) with Hydra-based configuration management and multi-model support.

## Features

- **Hydra Configuration**: Composable YAML configurations for experiments, models, scenarios
- **Multi-Model Support**: GPT-4, Claude, Ollama with per-agent model assignment
- **Dynamic Scenarios**: Switch scenarios via config (marketplace, debate, custom)
- **Modular Architecture**: Engines, simulators, and components can be mixed and matched
- **Extensible Design**: Easy to add new scenarios, agents, game masters, and components
- **Full Dev Infrastructure**: Pre-commit hooks, CI/CD, type checking, testing

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/yourusername/concordia-sim.git
cd concordia-sim/simulator

# Using uv (recommended)
uv sync

# Or using pip
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

### Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Add your API keys to `.env`:
```
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

### Running Simulations

```bash
# Run with default configuration (marketplace)
python run_experiment.py

# Switch to debate scenario
python run_experiment.py scenario=debate

# Override parameters
python run_experiment.py simulation.execution.max_steps=50 model=claude

# Multi-model simulation
python run_experiment.py model=multi_model

# View configuration without running
python run_experiment.py --cfg job
```

## Project Structure

```
simulator/
├── run_experiment.py              # Main entry point (Hydra-decorated)
├── config/                        # Hydra configuration
│   ├── experiment.yaml            # Main config with defaults
│   ├── simulation/                # Simulation mode configs
│   │   ├── sequential.yaml
│   │   └── parallel.yaml
│   ├── model/                     # Model configurations
│   │   ├── gpt4.yaml
│   │   ├── claude.yaml
│   │   ├── mock.yaml
│   │   └── multi_model.yaml
│   ├── environment/               # Environment settings
│   │   ├── generic_world.yaml
│   │   └── game_theoretic.yaml
│   ├── scenario/                  # Scenario definitions (dynamic)
│   │   ├── marketplace.yaml       # Marketplace scenario
│   │   └── debate.yaml            # Debate scenario example
│   └── evaluation/                # Evaluation metrics
│       └── basic_metrics.yaml
├── scenarios/                     # Scenario implementations
│   └── marketplace/               # Marketplace scenario
│       ├── agents.py              # BuyerAgent, SellerAgent, AuctioneerAgent
│       ├── game_masters.py        # MarketGameMaster
│       ├── knowledge.py           # Knowledge builders
│       ├── events.py              # Event generators
│       └── data/
│           └── knowledge.yaml     # Static knowledge data
├── src/                           # Core library
│   ├── simulation/                # Simulation infrastructure
│   │   ├── simulation.py          # Core Simulation class
│   │   ├── simulators/            # Simulator implementations
│   │   │   ├── base.py            # BaseSimulator (abstract)
│   │   │   └── multi_model.py     # MultiModelSimulator
│   │   └── engines/               # Execution engines
│   │       ├── base.py            # BaseEngine (abstract)
│   │       └── sequential.py      # SequentialEngine
│   ├── entities/                  # Generic entity prefabs
│   │   ├── agents/
│   │   │   ├── basic_entity.py    # BasicEntity prefab
│   │   │   └── planning_agent.py  # PlanningAgent prefab
│   │   ├── game_masters/
│   │   │   └── basic_gm.py        # BasicGameMaster prefab
│   │   └── components/            # Reusable entity components
│   │       └── base.py            # BaseComponent (abstract)
│   ├── models/                    # Model implementations
│   │   └── local_model.py         # LocalModel for Ollama/local LLMs
│   └── utils/                     # Utilities
│       ├── config_helpers.py      # Config helper functions
│       ├── validation.py          # Config validation
│       ├── logging_setup.py       # Logging configuration
│       └── testing.py             # Test utilities and mocks
└── tests/                         # Test suite
    ├── conftest.py                # Pytest fixtures
    ├── test_agents/               # Agent tests
    │   └── test_basic_entity.py
    ├── test_simulators/           # Simulator tests
    │   ├── test_base_simulator.py
    │   └── test_multi_model.py
    ├── test_scenarios/            # Scenario-specific tests
    │   ├── test_marketplace_agents.py
    │   ├── test_marketplace_events.py
    │   └── test_marketplace_knowledge.py
    └── test_integration/          # Integration tests
        └── test_simulation_run.py
```

## Configuration System

### Composable Configs

Configs are composed using Hydra's defaults list:

```yaml
# config/experiment.yaml
defaults:
  - simulation: sequential
  - model: gpt4
  - environment: generic_world
  - scenario: marketplace
  - evaluation: basic_metrics
```

### Override Examples

```bash
# Use Claude instead of GPT-4
python run_experiment.py model=claude

# Parallel simulation with multi-model
python run_experiment.py simulation=parallel model=multi_model

# Custom parameters
python run_experiment.py \
  scenario.agents.buyer.budget=1000 \
  scenario.agents.seller.pricing_strategy=competitive
```

## Multi-Model Support

Assign different models to different agents:

```yaml
# config/model/multi_model.yaml
model_registry:
  gpt4:
    provider: openai
    model_name: gpt-4-turbo
  claude:
    provider: anthropic
    model_name: claude-3-5-sonnet

entity_model_mapping:
  Alice: gpt4
  Bob: claude
  narrator: gpt4
```

## Creating Custom Scenarios

1. Create config file `config/scenario/my_scenario.yaml`:

```yaml
name: my_scenario
premise: |
  Description of your scenario...

# Optional: define valid roles for validation
roles:
  - name: player
    description: "A player in the game"
  - name: referee
    singular: true

# Generic entities list - works for any scenario
agents:
  entities:
    - name: Alice
      role: player
      prefab: basic_entity  # or custom prefab
      params:
        goal: "Win the game"

    - name: Referee Bob
      role: referee
      prefab: basic_entity
      params:
        goal: "Ensure fair play"

game_master:
  prefab: basic_game_master
  name: narrator

prefabs:
  basic_entity: src.entities.agents.basic_entity.BasicEntity
  basic_game_master: src.entities.game_masters.basic_gm.BasicGameMaster
```

2. (Optional) Create custom prefabs in `scenarios/my_scenario/agents.py`
3. Run: `python run_experiment.py scenario=my_scenario`

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov=scenarios

# Specific tests
pytest tests/test_scenarios/ -v
```

### Code Quality

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Type checking
mypy src/ scenarios/

# Linting
ruff check src/ scenarios/
```

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Using commitizen
cz commit

# Manual format
git commit -m "feat(scenario): add new auction mechanism"
git commit -m "fix(agent): resolve memory retrieval bug"
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Concordia](https://github.com/google-deepmind/concordia) by Google DeepMind
- [Hydra](https://hydra.cc/) by Facebook Research
