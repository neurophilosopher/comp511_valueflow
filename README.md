# Concordia v2 Multi-Agent Simulation Framework

A flexible multi-agent simulation framework built on [Concordia v2](https://github.com/google-deepmind/concordia) with Hydra-based configuration management and multi-model support.

## Features

- **Hydra Configuration**: Composable YAML configurations for experiments, models, scenarios
- **Multi-Model Support**: GPT-4, Claude, Ollama with per-agent model assignment
- **Dynamic Scenarios**: Switch scenarios via config (marketplace, election, misinformation, custom)
- **Social Media Environment**: In-memory social media platform with posts, replies, likes, boosts, follows
- **Evaluation Probes**: Query agents at checkpoints without affecting their memory (categorical, numeric, boolean)
- **Modular Architecture**: Engines, simulators, and components can be mixed and matched
- **Extensible Design**: Easy to add new scenarios, agents, game masters, and components
- **Full Dev Infrastructure**: Pre-commit hooks, CI/CD, type checking, 241 tests

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
uv run python run_experiment.py

# Switch to misinformation scenario (social media environment)
uv run python run_experiment.py scenario=misinformation

# Switch to election scenario
uv run python run_experiment.py scenario=election

# Override parameters
uv run python run_experiment.py simulation.execution.max_steps=50 model=claude

# Multi-model simulation
uv run python run_experiment.py model=multi_model

# Quick test with mock models (no API calls)
uv run python run_experiment.py --quick-test

# View configuration without running
uv run python run_experiment.py --cfg job
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
│   ├── scenario/                  # Scenario definitions
│   │   ├── marketplace.yaml
│   │   ├── election.yaml
│   │   └── misinformation.yaml
│   └── evaluation/                # Evaluation metrics
│       ├── basic_metrics.yaml
│       ├── election.yaml
│       └── marketplace.yaml
├── scenarios/                     # Scenario implementations
│   ├── marketplace/               # Marketplace scenario
│   │   ├── agents.py              # BuyerAgent, SellerAgent, AuctioneerAgent
│   │   ├── game_masters.py        # MarketGameMaster
│   │   ├── knowledge.py           # Knowledge builders
│   │   ├── events.py              # Event generators
│   │   └── data/knowledge.yaml
│   ├── election/                  # Election scenario
│   │   ├── agents.py, game_masters.py, knowledge.py, events.py
│   │   └── data/knowledge.yaml
│   └── misinformation/            # Misinformation scenario (social media)
│       ├── agents.py              # SocialMediaUserAgent prefab
│       └── game_masters.py        # MisinformationGameMaster
├── src/                           # Core library
│   ├── simulation/                # Simulation infrastructure
│   │   ├── simulation.py          # Core Simulation class
│   │   ├── simulators/
│   │   │   ├── base.py            # BaseSimulator (abstract)
│   │   │   └── multi_model.py     # MultiModelSimulator
│   │   └── engines/
│   │       ├── base.py            # BaseEngine (abstract)
│   │       ├── sequential.py      # SequentialEngine
│   │       └── engine_utils.py    # Action spec parser fix
│   ├── entities/                  # Generic entity prefabs
│   │   ├── agents/
│   │   │   ├── basic_entity.py    # BasicEntity prefab
│   │   │   └── planning_agent.py  # PlanningAgent prefab
│   │   ├── game_masters/
│   │   │   └── basic_gm.py        # BasicGameMaster prefab
│   │   └── components/
│   │       └── base.py            # BaseComponent (abstract)
│   ├── environments/              # Environment implementations
│   │   └── social_media/
│   │       ├── app.py             # Post dataclass + SocialMediaApp
│   │       ├── engine.py          # SocialMediaEngine
│   │       ├── game_master.py     # SocialMediaGameMaster prefab
│   │       └── analysis.py        # Transmission chain analysis
│   ├── evaluation/                # Evaluation system
│   │   ├── probes.py              # CategoricalProbe, NumericProbe, BooleanProbe
│   │   └── probe_runner.py        # ProbeRunner (orchestrates probe execution)
│   ├── models/                    # Model implementations
│   │   ├── openai_model.py        # OpenAI GPT models
│   │   ├── anthropic_model.py     # Anthropic Claude models
│   │   └── local_model.py         # LocalModel for Ollama/local LLMs
│   └── utils/                     # Utilities
│       ├── config_helpers.py      # Config helper functions
│       ├── validation.py          # Config validation
│       ├── event_logger.py        # Simulation event logging
│       ├── logging_setup.py       # Logging configuration
│       └── testing.py             # Test utilities and mocks
├── scripts/                       # Utility scripts
│   ├── run_social_media_sim.py    # Standalone social media runner
│   ├── analyze_social_media.py    # CLI analysis tool
│   └── explore_dashboard.py       # Interactive Dash explorer
└── tests/                         # Test suite (241 tests)
    ├── conftest.py                # Shared fixtures
    ├── environments/              # Social media environment tests
    │   ├── test_social_media_app.py
    │   ├── test_social_media_engine.py
    │   ├── test_social_media_game_master.py
    │   └── test_social_media_analysis.py
    ├── test_agents/
    │   └── test_basic_entity.py
    ├── test_simulators/
    │   ├── test_base_simulator.py
    │   └── test_multi_model.py
    ├── test_evaluation/
    │   ├── test_probes.py
    │   └── test_probe_runner.py
    ├── test_utils/
    │   └── test_validation.py
    ├── test_scenarios/
    │   ├── test_marketplace_agents.py
    │   ├── test_marketplace_events.py
    │   └── test_marketplace_knowledge.py
    └── test_integration/
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
uv run python run_experiment.py model=claude

# Parallel simulation with multi-model
uv run python run_experiment.py simulation=parallel model=multi_model

# Custom parameters
uv run python run_experiment.py \
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

## Social Media Environment

The social media environment provides an in-memory platform for simulating information spread:

- **Actions**: post, reply, like, unlike, boost, follow, unfollow
- **Feed**: Chronological feed from followed users
- **Engine**: `SocialMediaEngine` runs parallel agent actions each step
- **Analysis**: Transmission chain extraction, keyword overlap, network analysis

```bash
# Run misinformation scenario
uv run python run_experiment.py scenario=misinformation

# Standalone runner (mock mode)
uv run python scripts/run_social_media_sim.py

# Analyze results
uv run python scripts/analyze_social_media.py path/to/checkpoint.json

# Interactive dashboard
uv run python scripts/explore_dashboard.py path/to/checkpoint.json
```

## Evaluation Probes

Query agents at checkpoints without affecting their memory:

```yaml
# config/evaluation/election.yaml
metrics:
  vote_preference:
    type: categorical
    categories: [conservative, progressive, undecided]
    prompt_template: |
      Based on {agent_name}'s views, which candidate do they prefer?
    applies_to: [voter]
```

Probe types: `categorical`, `numeric` (min/max range), `boolean` (yes/no).

## Creating Custom Scenarios

1. Create config file `config/scenario/my_scenario.yaml`:

```yaml
name: my_scenario
premise: |
  Description of your scenario...

roles:
  - name: player
    description: "A player in the game"

agents:
  entities:
    - name: Alice
      role: player
      prefab: basic_entity
      params:
        goal: "Win the game"

game_master:
  prefab: basic_game_master
  name: narrator

prefabs:
  basic_entity: src.entities.agents.basic_entity.BasicEntity
  basic_game_master: src.entities.game_masters.basic_gm.BasicGameMaster
```

2. (Optional) Create custom prefabs in `scenarios/my_scenario/agents.py`
3. Run: `uv run python run_experiment.py scenario=my_scenario`

## Development

### Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=src --cov=scenarios

# Specific test directory
uv run pytest tests/test_evaluation/ -v

# Skip integration tests
uv run pytest -m "not integration"
```

### Code Quality

```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files

# Type checking
uv run mypy src/ scenarios/

# Linting
uv run ruff check src/ scenarios/
```

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Using commitizen
uv run cz c

# Manual format
git commit -m "feat(scenario): add new auction mechanism"
git commit -m "fix(agent): resolve memory retrieval bug"
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Concordia](https://github.com/google-deepmind/concordia) by Google DeepMind
- [Hydra](https://hydra.cc/) by Facebook Research
