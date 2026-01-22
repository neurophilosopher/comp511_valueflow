# Concordia v2 Multi-Agent Simulation Framework

A flexible multi-agent simulation framework built on [Concordia v2](https://github.com/google-deepmind/concordia) with Hydra-based configuration management and multi-model support.

## Features

- **Hydra Configuration**: Composable YAML configurations for experiments, models, scenarios
- **Multi-Model Support**: GPT-4, Claude, Ollama with per-agent model assignment
- **Marketplace Scenario**: BuyerAgent, SellerAgent, AuctioneerAgent prefabs
- **Extensible Architecture**: Easy to add new scenarios, agents, and game masters
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
# Run with default configuration
python run_experiment.py

# Run marketplace scenario
python run_experiment.py scenario=marketplace

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
├── run_experiment.py           # Main entry point (Hydra-decorated)
├── config/                     # Hydra configuration
│   ├── experiment.yaml         # Main config with defaults
│   ├── simulation/             # Simulation mode configs
│   ├── model/                  # Model configurations
│   ├── environment/            # Environment settings
│   ├── scenario/               # Scenario definitions
│   └── evaluation/             # Evaluation metrics
├── scenarios/                  # Scenario implementations
│   └── marketplace/            # Marketplace scenario
│       ├── agents.py           # Agent prefabs
│       ├── game_masters.py     # Game master prefabs
│       ├── knowledge.py        # Knowledge builders
│       ├── events.py           # Event generators
│       └── data/               # Static data
├── src/                        # Core library
│   ├── simulation/             # Simulation infrastructure
│   ├── entities/               # Generic prefabs
│   └── utils/                  # Utilities
└── tests/                      # Test suite
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

1. Create scenario directory: `scenarios/my_scenario/`
2. Define agents in `agents.py`:

```python
@dataclasses.dataclass
class MyAgent(prefab_lib.Prefab):
    description: str = "My custom agent"
    params: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def build(self, model, memory_bank):
        # Build agent with components
        ...
```

3. Create config: `config/scenario/my_scenario.yaml`
4. Run: `python run_experiment.py scenario=my_scenario`

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
