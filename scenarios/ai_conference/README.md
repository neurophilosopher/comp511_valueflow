# AI Conference Groupthink Scenario

Two echo chambers — conference attendees (inside) and protesters (outside) — collide on social media. The scenario studies how in-group consensus forms, how dissent is suppressed or embraced, and what happens when echo chambers interact online.

## Research Questions

- How quickly does groupthink solidify within each echo chamber?
- Do bridge characters (journalist, conflicted engineer) successfully transmit ideas across groups?
- Does the "spiral of silence" manifest — do agents with private doubts (Lisa, Sam) self-censor?
- What happens when protest content reaches conference attendees' feeds?

## Agents (9 total)

### Conference Attendees (Inside) — 4 agents

| Agent | Role | Behavior |
|-------|------|----------|
| **Dr. Victor Huang** | Keynote speaker / thought leader | Charismatic, dismisses skeptics, shapes the optimistic narrative |
| **Priya Sharma** | Marketing VP | Adopts high-status consensus, networks aggressively, avoids appearing skeptical |
| **Derek Okafor** | Startup founder | Self-promotional, frames all concerns as solvable, business depends on AI optimism |
| **Lisa Tanaka** | Innovation manager | Privately doubtful, suppresses concerns to fit in, felt guilt seeing protesters |

### Protesters (Outside) — 3 agents

| Agent | Role | Behavior |
|-------|------|----------|
| **Marcus Rivera** | Labor union organizer | Confrontational, frames struggle in moral terms, shames conference attendees |
| **Dr. Amara Osei** | AI ethics researcher | Data-driven critiques, denied a speaker slot, builds intellectual counternarrative |
| **Zoe Chen** | Climate activist | Provocative tactics, focuses on AI energy costs, black-and-white moral framing |

### Bridge Characters — 2 agents

| Agent | Role | Behavior |
|-------|------|----------|
| **Jordan Ellis** | Tech journalist | Covers both sides, maintains neutrality, followed by both communities |
| **Sam Nakamura** | ML engineer | Conflicted between professional conformity and personal conscience |

## Network Structure

```
Conference Cluster          Bridge          Protest Cluster
┌─────────────────┐                        ┌──────────────────┐
│  Victor ←→ Priya │ ←── Jordan ──→       │ Marcus ←→ Amara  │
│    ↕        ↕    │       ↕               │    ↕        ↕    │
│  Derek ←→ Lisa  │ ←── Sam               │      Zoe         │
└─────────────────┘                        └──────────────────┘
```

Jordan follows 7/8 agents (primary bridge). Sam leans conference-side but also follows Amara. Protesters don't directly follow conference attendees — collision happens through bridges.

## Seed Posts (6 at step 0)

| Author | Tags | Content |
|--------|------|---------|
| Dr. Victor Huang | `conference_insider`, `groupthink_seed`, `keynote` | Keynote opener — "AI isn't coming, it's HERE" |
| Priya Sharma | `conference_insider`, `groupthink_seed`, `networking` | Networking hype — "every conversation confirms AI is the biggest opportunity" |
| Marcus Rivera | `protest`, `groupthink_seed`, `labor` | Protest report — "200+ workers demanding acknowledgment" |
| Dr. Amara Osei | `protest`, `groupthink_seed`, `ethics` | Critique — "zero sessions on bias, denied a speaker slot" |
| Jordan Ellis | `bridge`, `journalism` | Neutral observation — "two narratives playing out 50 feet apart" |
| Sam Nakamura | `bridge`, `conflicted`, `conference_insider` | Ambivalent — "great talks, but protesters' signs hit close to home" |

## Running

```bash
# Quick test (mock model, no API calls)
uv run python run_experiment.py scenario=ai_conference model=mock simulation.execution.max_steps=2

# Full run
uv run python run_experiment.py scenario=ai_conference model=gpt4 simulation.execution.max_steps=5

# Explore results
uv run python scripts/explore_dashboard.py outputs/ai_conference_experiment/.../checkpoints/step_N_checkpoint.json --seed-tags groupthink_seed
```

## Analysis

Use the existing analysis tools with tag-based filtering:

```python
from src.environments.social_media.analysis import find_transmission_chains

# Track how conference echo chamber content spreads
conference_chains = find_transmission_chains(app, seed_tags=["conference_insider"])

# Track how protest content spreads
protest_chains = find_transmission_chains(app, seed_tags=["protest"])

# Track bridge content that spans both perspectives
bridge_chains = find_transmission_chains(app, seed_tags=["bridge"])
```

### Tag Taxonomy

- `conference_insider` — conference echo chamber content
- `protest` — protest echo chamber content
- `groupthink_seed` — posts designed to initiate groupthink dynamics
- `bridge` — content spanning both perspectives
- `conflicted` — content expressing internal tension
- `keynote`, `networking`, `labor`, `ethics`, `journalism` — topical subtags
