# Closing the Evaluation Gap: Paper-to-Code Mapping

This document maps the formal abstraction introduced in **"Position: Time to Close The Validation Gap in LLM Social Simulations"** (Puelma Touzel et al., 2026) to its concrete implementation in this codebase (**ScenSim**). The goal is to make the simulator legible as a research object — not just running code — so that every symbol in the paper has a traceable counterpart in source.

ScenSim supports multiple environments (generic world, social media, game-theoretic) and multiple scenarios (election, marketplace, misinformation, debate, AI conference). This document uses the **social media environment** as a running example because it has the most explicit state representation, but every formal object exists in all environment/engine combinations. Where the mapping differs across engines, both paths are shown.

> **Paper link**: *(insert arXiv / proceedings link)*

## 1. Core Equation

The paper models a simulation as sampling trajectories from a distribution conditioned on the simulation specification and all LLM parameters:

```
p(τ | X, Θ)
```

| Symbol | Plain-English gloss |
|--------|-------------------|
| **τ** | A single trajectory — the full record of what happened during one run |
| **X** | The simulation specification — scenario, environment, component wiring, prompts |
| **Θ** | All LLM parameters — model weights, temperature, persona text, everything fed to or inside the models |
| **L(τ)** | An evaluation function — any metric computed over a trajectory |

---

## 2. Formal Abstraction → Code (Paper Sec 2.1)

These mappings are environment-independent — they hold regardless of which engine or scenario is active.

| Symbol | Paper meaning | Code |
|--------|--------------|------|
| **τ** | Trajectory | Checkpoint JSON + raw log — `Simulation.make_checkpoint_data()` at `src/simulation/simulation.py:352` |
| **X** | Simulation specification | Hydra config composition: `config/experiment.yaml` → `BaseSimulator.build_config()` at `src/simulation/simulators/base.py:345` |
| **Θ** | All LLM parameters | `config/model/*.yaml` + `scenario.agents.entities[i].params` (persona, goal, personality, background) |
| **L(τ)** | Evaluation function | `Probe.query()` at `src/evaluation/probes.py:58`; `scripts/eval_style_diversity.py` (social media-specific) |

---

## 3. System Components (Paper Sec 2.2)

Each component below exists in every environment, but the *implementation mechanism* varies. The key architectural distinction: in the **sequential engine** (Concordia default, used by `generic_world` and `game_theoretic` environments), the game master's LLM mediates observations and transitions. In **custom engines** like `SocialMediaEngine`, these are implemented as deterministic functions.

### Environment state: **s_t**

| Engine | Where s_t lives |
|--------|-----------------|
| Sequential (generic) | **Implicitly** in the game master's `AssociativeMemoryBank` (`src/simulation/simulation.py:86`). No structured state object — the GM "remembers" what has happened. |
| Social media | **Explicitly** in `SocialMediaApp` (`src/environments/social_media/app.py:47`): posts, follower graph, likes, step counter. |

### Observations: **o^i_t = O^i(s_t)**

| Engine | How O^i works |
|--------|---------------|
| Sequential (generic) | **LLM-mediated**: the engine asks the GM *"What does {entity.name} observe?"* and the GM generates a natural-language observation. O^i is stochastic — it depends on Θ. (Concordia `Sequential.make_observation()`) |
| Social media | **Deterministic**: `SocialMediaApp.format_timeline()` at `app.py:291`, called at `engine.py:282`. Returns posts from followed users, filtered by the social graph. O^i is a pure function of s_t. |

In both cases, the observation string is delivered via `entity.observe(observation)`.

### Agent internal state: **z^i_t**

This is environment-independent. The Concordia component dictionary is assembled in `BasicEntity.build()` at `src/entities/agents/basic_entity.py:47`:
- `AssociativeMemory` (line 70) — long-term memory bank
- `ObservationToMemory` (line 88) — writes observations into memory
- `LastNObservations` (line 92) — short-term buffer of recent observations
- `SituationPerception` (line 100) — LLM-based reasoning about current context
- `Constant` goal component (line 110, optional) — fixed goal string
- `Instructions` (line 80) — persona and role-play instructions

Scenario-specific prefabs (e.g., `VoterAgent`, `BuyerAgent`) extend this with domain-specific components.

### Actions: **a^i_t**

Environment-independent: `entity.act(action_spec)` → `ConcatActComponent` at `basic_entity.py:116`.

The *action spec* (what the agent is asked to do) varies by engine:
- **Sequential**: GM generates the spec — "What should {entity} do?" with output type FREE or CHOICE.
- **Social media**: Fixed structured format (`ACTION: <type> | TARGET: <id> | CONTENT: <text>`) defined at `engine.py:156`.

### Agent LLM parameters: **θ^i**

Environment-independent:
- **Model weights & temperature**: `config/model/*.yaml` (`gpt4.yaml`, `gpt4o.yaml`, `claude.yaml`, `multi_model.yaml`)
- **Persona & goals**: `scenario.agents.entities[i].params` — fields vary by scenario (e.g., `persona_context` for election voters, `budget` and `strategy` for marketplace buyers)

### Environment parameters: **θ_Env**

- `config/environment/*.yaml` — engine type, world rules, timeline limits, scoring settings
  - `generic_world.yaml`: time/location systems, GM observation/resolution style
  - `social_media.yaml`: engine selection, timeline limit
  - `game_theoretic.yaml`: scoring system, round structure, turn order
- `scenario.game_master.params` — GM persona and behavior parameters

### Action generation function: **A^i(·)**

Environment-independent. The component chain built in prefab `.build()` methods (e.g., `basic_entity.py:47`):
1. Each component in `components` dict produces a text fragment (instructions, observations, situation assessment, goal)
2. `ConcatActComponent` (line 116) concatenates all fragments in order
3. The concatenated context is sent to the LLM, which generates the action string

### Internal state update: **Z^i(·)**

Environment-independent. `entity.observe(observation)`:
- `ObservationToMemory` stores the observation text into the associative memory bank
- On the next `act()` call, reasoning components (`SituationPerception`, `LastNObservations`) re-query memories, effectively updating **z^i_t** just-in-time

### Environment transition: **T(·)**

| Engine | How T works |
|--------|-------------|
| Sequential (generic) | **LLM-mediated**: the engine calls `resolve()` — the GM observes the action, then is asked *"Because of all that came before, what happens next?"* and generates a resolution. T is stochastic. |
| Social media | **Deterministic**: `execute_action()` at `engine.py:60–134` dispatches parsed actions to `SocialMediaApp.post/reply/like/boost/follow/unfollow()`. T is a pure function. |

---

## 4. Dynamics — The Simulation Loop (Paper Sec 2.2 + Appendix A)

### Sequential Engine (Concordia default)

Used by `generic_world` and `game_theoretic` environments. The loop lives in Concordia's `Sequential.run_loop()`:

| Step | Operation | Mechanism |
|------|-----------|-----------|
| 1. **Terminate?** | GM decides whether to stop | `GM.act(OutputType.TERMINATE)` → yes/no |
| 2. **Select GM** | Choose active game master | `GM.act(OutputType.NEXT_GAME_MASTER)` (skipped if only one GM) |
| 3. **Observe** | GM generates observation for each entity (*parallel*) | `GM.act("What does X observe?")` → `entity.observe(obs)` |
| 4. **Schedule** | GM picks who acts next (*one* agent per step) | `GM.act(OutputType.NEXT_ACTING)` |
| 5. **Act** | Selected entity generates action | `entity.act(action_spec)` |
| 6. **Resolve** | GM narrates what happens | `GM.observe(action)` → `GM.act("What happens next?")` |
| 7. **Checkpoint** | Save state, run probes | `checkpoint_callback(steps)` |

Key difference from the paper's formulation: in sequential mode, **one agent acts per step** (GM-scheduled), not all agents simultaneously.

### Social Media Engine

Used by the `social_media` environment. The loop lives in `SocialMediaEngine.run_loop()` at `engine.py:209`, inner function `_entity_step()` at `engine.py:276`:

| Step | Operation | Code |
|------|-----------|------|
| 1. **Observe** | Generate per-agent timeline | `app.format_timeline(entity.name)` (line 282) |
| 2. **Think** | Store observation, update memory components | `entity.observe(observation)` (line 284) |
| 3. **Act** | Build prompt from component chain, LLM generates action | `entity.act(action_spec)` (line 287) |
| 4. **Execute** | Parse action string, apply to app state | `parse_action()` + `execute_action()` (lines 290–291) |
| 5. **Parallel** | All agents run steps 1–4 simultaneously | `concurrency.run_tasks(tasks)` (line 308) |
| 6. **Checkpoint** | Save state, run probes | `checkpoint_callback(steps)` (line 348) |

Key difference: **all agents act every step**, O^i and T are deterministic, and there is no GM-mediated scheduling.

### Trajectory Factorization (Appendix A)

The full joint trajectory distribution factorizes as:

```
p(τ) = p(s₀, z⁰₁, ..., z⁰ₙ)                    [initial state]
        × ∏_t  T(s_{t+1} | s_t, a_t)             [environment transition]
             × ∏_i O^i(o^i_t | s_t)              [observation]
                  × Z^i(z^i_t | z^i_{t-1}, o^i_t) [think / memory update]
                  × A^i(a^i_t | z^i_t)            [action generation]
```

Each factor maps to code:

| Factor | Description | Code counterpart |
|--------|------------|-----------------|
| p(s₀, z⁰₁, ..., z⁰ₙ) | Initial state | Scenario config: premise, shared_memories, agent params; `formative_memories_initializer` prefab injects memories. *Social media example*: seed_posts, initial_graph. *Election example*: candidate policies, voter personas. |
| T(s_{t+1} \| s_t, a_t) | Environment transition | **Sequential**: GM `resolve()` (stochastic). **Social media**: `execute_action()` at `engine.py:60` (deterministic). |
| O^i(o^i_t \| s_t) | Observation function | **Sequential**: GM `make_observation()` (stochastic). **Social media**: `format_timeline()` at `app.py:291` (deterministic). |
| Z^i(z^i_t \| z^i_{t-1}, o^i_t) | Internal state update | `entity.observe()` + `ObservationToMemory` + memory component re-query (all engines) |
| A^i(a^i_t \| z^i_t) | Action generation | `entity.act()` + `ConcatActComponent` at `basic_entity.py:116` (all engines) |

Note: when O^i or T is deterministic, the corresponding factor in the product is a delta distribution — it contributes no variance to the trajectory distribution. All trajectory-level stochasticity then comes from A^i (the LLM action generation).

---

## 5. Learning Problems (Paper Sec 4.1–4.3)

| Learning Problem | Formal Objective | Status | Code Entry Point |
|-----------------|-----------------|--------|-----------------|
| Next-Action Prediction | L_NAP = -log p(a_{t+1} \| history) | **Target** | Checkpoints provide trajectory data for offline evaluation |
| Model Training (SFT/RL) | θ^i = θ_base + Δθ^i | **Target** (external) | Model configs (`config/model/`) support swapping fine-tuned models |
| Model Steering | Δx^i steering vectors | **Target** (external) | — |
| Prompt Engineering | Structure of A^i | **Supported** | Prefab `.build()` methods define component chains (`basic_entity.py:47`) |
| Persona Learning | Persona from data | **Partial** | `scenario.agents.entities[i].params.persona` — currently hand-authored |
| Component Structure Learning | Learn z^i_t structure | **Target** | Concordia component architecture is modular; components are plug-and-play |
| Environment Initialization | Choose s₀ | **Supported** | `scenario.initial_graph`, `scenario.seed_posts`, shared/player memories |
| Interaction Orchestration | Design of T | **Supported** | Engine selection (`environment.engine`), GM prefab, action space in engine |
| Stability / Long-Horizon | Distributional drift over T steps | **Target** | Checkpoint series + probe time-series enable measurement |

---

## 6. Evaluation Fallacies (Paper Table 1)

### Axis I — Distributional Validity
**What we do**: Probes run at every checkpoint (not just final state). For social media scenarios, style diversity metrics (`eval_style_diversity.py`) compute population-level statistics: self-BLEU, lexical diversity, near-duplicate rate, action-type entropy.
**Gap**: No distributional comparison to real-world data. Style diversity metrics are social media-specific; no generic equivalent exists for sequential/world scenarios.

### Axis II — Agent-Human Correspondence
**What we do**: Personas are specified via config (`personality`, `background`, `goal`). Scenarios like `election` define realistic voter profiles with policy preferences and communication styles.
**Gap**: Personas are not grounded in empirical human data. No human calibration benchmarks.

### Axis III — Design Justification
**What we do**: Hydra makes **X** and **Θ** fully specified and reproducible. The component architecture supports ablations (swap/remove components in prefab `.build()`). Multi-model support (`multi_model.yaml`) enables controlled comparisons.
**Gap**: No systematic ablation studies have been run yet.

### Axis IV — Emergence & Diversity
**What we do**: Style diversity metrics directly measure homogeneity (self-BLEU, inter-agent distinctiveness). Multi-model support (`config/model/multi_model.yaml`) breaks shared-θ across agents.
**Gap**: Conditional independence p(a_t) ≠ Π p(a^i_t) is not formally tested.

---

## 7. ScenSim as Example Simulator (Paper Sec 4.4)

The paper identifies five configurable components of a simulator. Each maps to a Hydra config group:

| Paper component | Config group | Key files |
|----------------|-------------|-----------|
| **Simulator** (engine, logging, execution) | `config/simulation/` | `sequential.yaml`, `parallel.yaml` |
| **Model** (genAI models, Θ) | `config/model/` | `gpt4.yaml`, `gpt4o.yaml`, `claude.yaml`, `multi_model.yaml`, `mock.yaml` |
| **Environment** (T, θ_Env) | `config/environment/` | `generic_world.yaml`, `social_media.yaml`, `game_theoretic.yaml` |
| **Scenario** (A^i, Z^i, shared knowledge) | `config/scenario/` + `scenarios/` | Per-scenario YAML (`election`, `marketplace`, `misinformation`, `debate`, `ai_conference`) + Python prefabs |
| **Evaluation** (L) | `config/evaluation/` + `scripts/` | `basic_metrics.yaml`, `election.yaml`, `marketplace.yaml`, `eval_style_diversity.py` |

---

## 8. Alignment Analysis: Formulation ↔ Codebase Gaps

This section identifies places where the paper's formal abstraction and the codebase diverge, with suggested changes to one or both.

### 8.1 No explicit `Environment` interface

**Gap**: The paper cleanly separates s_t (state), O^i (observation function), and T (transition function) as distinct objects. In code:
- Social media: these exist as methods on `SocialMediaApp` + `execute_action()`, but not behind a shared interface.
- Sequential/generic: all three are folded into the GM's LLM calls within Concordia's `Sequential.run_loop()`. There is no `Environment` object at all.

**Suggestion (code)**: Introduce an `Environment` protocol or ABC:
```python
class Environment(Protocol):
    def get_state(self) -> dict[str, Any]: ...           # s_t
    def observe(self, agent_name: str) -> str: ...       # O^i
    def transition(self, agent: str, action: str) -> ActionResult: ...  # T
```
`SocialMediaApp` nearly implements this already. The generic path would wrap GM calls. This makes the paper's s_t / O^i / T boundaries visible in code.

### 8.2 s_t has no uniform representation

**Gap**: In social media, `SocialMediaApp.to_dict()` gives a structured state snapshot. In the generic path, the "state" is diffusely spread across the GM's memory bank — there is no `get_state()`. Checkpoints (`make_checkpoint_data()` at `simulation.py:352`) save entity/GM component states but do not include the `SocialMediaApp` state either (despite `engine.get_app_state()` existing at `engine.py:354`).

**Suggestion (code)**:
1. Have `save_checkpoint()` call `engine.get_app_state()` when the engine supports it, so τ includes s_t snapshots.
2. For the generic path, define s_t as the GM memory bank contents (which *are* already captured in the GM's component state).

### 8.3 O^i and T can be deterministic or stochastic

**Gap**: The factorization treats O^i and T as general distribution factors. This is formally correct — a deterministic function is a delta distribution — but the practical implication is important: in the social media engine, **all trajectory variance comes from A^i** (the LLM action calls). In the sequential engine, O^i and T also contribute variance through the GM's LLM calls. This affects how to interpret trajectory-level statistics.

**Suggestion (formulation)**: The paper could note this distinction and its consequences for variance decomposition. Deterministic O^i/T means ablation studies can isolate A^i's contribution cleanly. Stochastic O^i/T means the GM is a confound.

### 8.4 Turn scheduling is absent from the factorization

**Gap**: The factorization writes ∏_i at each time step, implying all agents act every step. This matches the social media engine (all agents act simultaneously). But in the sequential engine, the GM selects **one agent per step** — the other agents' ∏_i factors are trivial (no action). The paper has no scheduling factor.

**Suggestion (formulation)**: Add a scheduling indicator σ_t ∈ {1, ..., N} or a mask m^i_t ∈ {0, 1} to the factorization:
```
∏_i  m^i_t · A^i(a^i_t | z^i_t) + (1 - m^i_t) · δ(a^i_t = ∅)
```
In simultaneous engines, m^i_t = 1 for all i. In sequential engines, m^i_t is determined by the GM's scheduling decision.

**Suggestion (code)**: Alternatively, unify behavior by having the sequential engine expose all agents' "actions" each step (with skips for non-acting agents), so τ has a uniform shape across engines.

### 8.5 The GM's role is not captured by a single symbol

**Gap**: In the sequential engine, the GM implements O^i, T, *and* scheduling — it is the environment's "brain." In the social media engine, the GM is only used for initialization (injecting memories). The paper's θ_Env covers GM parameters, but the GM's triple role (observer, resolver, scheduler) in the sequential path deserves explicit treatment.

**Suggestion (formulation)**: Model the GM as a meta-agent whose LLM parameterizes O^i, T, and σ_t in the sequential case. This makes it clear that θ_Env includes a model (with its own Θ) when the environment is GM-mediated.

### 8.6 `config/environment/` conflates engine and parameters

**Gap**: The paper distinguishes the "simulator" (engine/orchestration) from the "environment" (state + transitions). In the Hydra config, `config/environment/*.yaml` combines both: `engine: social_media` (orchestration choice) sits alongside `timeline_limit: 20` (environment parameter). The engine type arguably belongs in `config/simulation/`.

**Suggestion (code)**: Either:
- Move engine selection to `config/simulation/` (where `sequential.yaml` and `parallel.yaml` already live), or
- Document that the environment config group intentionally covers both the paper's "environment" and part of its "simulator."

### 8.7 Evaluation tooling is environment-specific

**Gap**: The probe system (`src/evaluation/probes.py`) is generic — it works via `entity.act()` and applies to any scenario. But the style diversity script (`scripts/eval_style_diversity.py`) is specific to the social media environment (it requires `SocialMediaApp` posts). There is no equivalent reference-free diversity metric for sequential/generic scenarios.

**Suggestion (code)**: Factor out the generic parts of `eval_style_diversity.py` (self-BLEU, lexical diversity, n-gram overlap) into a utility that operates on any list of agent utterances extracted from `raw_log`, regardless of environment.

### Summary Table

| Gap | Affects | Suggested fix |
|-----|---------|---------------|
| 8.1 No `Environment` interface | Code legibility | Add `Environment` protocol |
| 8.2 No uniform s_t | Checkpoints, reproducibility | Include app state in checkpoints; define s_t for generic path |
| 8.3 Deterministic vs stochastic O^i/T | Variance interpretation | Note in paper; affects ablation design |
| 8.4 No scheduling in factorization | Sequential engine | Add scheduling factor to formulation or unify τ shape |
| 8.5 GM's triple role | Formal model | Model GM as meta-agent parameterizing O^i, T, σ_t |
| 8.6 Config conflates engine + env | Hydra structure | Split or document |
| 8.7 Env-specific eval | Eval coverage | Factor out generic diversity metrics |

---

## Key Source Files Referenced

| File | Role |
|------|------|
| `src/environments/social_media/app.py` | `SocialMediaApp` — environment state **s_t** (social media) |
| `src/environments/social_media/engine.py` | `SocialMediaEngine.run_loop()` — dynamics loop (social media) |
| `src/entities/agents/basic_entity.py` | `BasicEntity` prefab — **z^i**, **A^i**, **Z^i** (all environments) |
| `src/simulation/simulators/base.py` | `BaseSimulator.build_config()` — **X** assembly (all environments) |
| `src/simulation/simulation.py` | `Simulation.play()`, checkpoints — **τ** (all environments) |
| `src/evaluation/probes.py` | `Probe.query()` — **L** (all environments) |
| `scripts/eval_style_diversity.py` | Style diversity metrics — **L** (social media-specific) |
| `config/experiment.yaml` | Hydra defaults composition — **X** (all environments) |
| `config/environment/*.yaml` | Environment + engine config — **θ_Env** |
| `config/scenario/*.yaml` | Scenario definitions — agent params, premises, prefab wiring |
