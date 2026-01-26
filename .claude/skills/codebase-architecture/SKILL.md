---
name: codebase-architecture
description: Deep dive into simulator architecture, data flow, and extension patterns. Load when working on structural changes or new scenarios.
autoload: true
---

# Concordia Simulator Architecture

## Core Data Flow

```
run_experiment.py (Hydra entry point)
    ↓
MultiModelSimulator(config)
    ↓
BaseSimulator.setup()
    ├── create_models() → dict[str, LanguageModel]
    ├── create_embedder() → Callable[[str], np.ndarray]
    ├── build_config() → prefab_lib.Config
    │   ├── build_prefabs() → loads _target_ classes
    │   └── build_instances() → entity configs + knowledge injection
    └── Simulation(config, models, embedder)
            ↓
        simulation.play()
            ↓
        [Result: HTML log or raw JSON]
```

## Configuration Composition

Hydra composes configs via `defaults` in `config/experiment.yaml`:

```
experiment.yaml
├── simulation: sequential.yaml | parallel.yaml
├── model: gpt4.yaml | claude.yaml | multi_model.yaml | mock.yaml
├── environment: generic_world.yaml | game_theoretic.yaml
├── scenario: marketplace.yaml | debate.yaml | <custom>.yaml
└── evaluation: basic_metrics.yaml
```

All values can be overridden via CLI or nested config files.

## Scenario Structure

A scenario consists of:

1. **Config file** (`config/scenario/<name>.yaml`):
   - `name`, `premise`, `setting`, `event`
   - `agents.entities[]` - list of entity definitions
   - `game_master` - GM prefab and params
   - `prefabs` - mapping of prefab names to `_target_` classes
   - `builders.knowledge` / `builders.events` - dynamic content generators
   - `shared_memories` - injected into all agents at start

2. **Implementation** (`scenarios/<name>/`):
   - `agents.py` - custom agent prefabs (extend `concordia.typing.prefab.Prefab`)
   - `game_masters.py` - custom GM prefabs
   - `knowledge.py` - `build_<scenario>_knowledge(name, role, params) -> list[str]`
   - `events.py` - `generate_<scenario>_events(config) -> list[Event]`
   - `data/` - static YAML data files

## Prefab System

Prefabs are factories that create Concordia entities. Two patterns supported:

**_target_ pattern (preferred):**
```yaml
prefabs:
  buyer_agent:
    _target_: scenarios.marketplace.agents.BuyerAgent
```

**Legacy string path:**
```yaml
prefabs:
  buyer_agent: scenarios.marketplace.agents.BuyerAgent
```

Prefabs must implement `concordia.typing.prefab.Prefab`:
- `build(config, model, memory, clock, ...)` returns the entity
- Can access `params` passed from instance config

## Entity Instance Configuration

Each entity in `agents.entities[]`:
```yaml
- name: Alice           # Required: unique entity name
  role: buyer           # Optional: scenario-specific role
  prefab: buyer_agent   # Required: key in prefabs mapping
  params:               # Passed to prefab.build()
    goal: "Find deals"
    budget: 1000
```

`BaseSimulator.build_instances()` converts these to `prefab_lib.InstanceConfig` objects.

## Knowledge Injection

1. `shared_memories` from scenario config → all agents
2. `build_agent_knowledge()` calls scenario's knowledge builder → per-agent
3. `player_specific_context` from entity `goal` param
4. All injected via `formative_memories_initializer__GameMaster` before simulation starts

## Model Configuration

**Single model** (`config/model/gpt4.yaml`):
```yaml
name: gpt4
_target_: concordia.language_model.gpt_model.GptLanguageModel
model_name: gpt-4o-mini
api_key: ${oc.env:OPENAI_API_KEY,}
```

**Multi-model** (`config/model/multi_model.yaml`):
```yaml
model_registry:
  gpt4: { _target_: ..., model_name: gpt-4o-mini }
  claude: { _target_: ..., model_name: claude-3-5-sonnet }
entity_model_mapping:
  Alice: gpt4
  Bob: claude
  _default_: gpt4
```

## Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `BaseSimulator` | `src/simulation/simulators/base.py` | Abstract base; config → Concordia objects |
| `MultiModelSimulator` | `src/simulation/simulators/multi_model.py` | Handles multi-model entity assignment |
| `Simulation` | `src/simulation/simulation.py` | Wraps Concordia simulation lifecycle |
| `BasicEntity` | `src/entities/agents/basic_entity.py` | Generic agent prefab |
| `BasicGameMaster` | `src/entities/game_masters/basic_gm.py` | Generic GM prefab |

## Adding a New Scenario

1. Create `config/scenario/<name>.yaml` with required fields
2. Create `scenarios/<name>/` with at minimum `agents.py`
3. Define prefabs in the config pointing to your agent classes
4. Optionally add `knowledge.py` and `events.py` for dynamic content
5. Run: `python run_experiment.py scenario=<name>`

## Testing Patterns

- Use `MockLanguageModel` from `src/utils/testing.py` for unit tests
- `mock_embedder()` returns zero vectors for testing
- `create_test_config()` provides minimal valid config
- Fixtures in `tests/conftest.py` for common setup
