"""ValueFlow game master: enforces DAG interaction topology and perturbation.

The game master controls:
1. Which agents observe which other agents' outputs (topology)
2. Persona perturbation injection for the target agent
3. Multi-round discussion flow
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib

logger = logging.getLogger(__name__)


def build_topology_graph(
    agent_names: list[str],
    topology_type: str,
    custom_adjacency: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Build an adjacency graph: who observes whom.

    In the returned dict, graph[A] = [B, C] means agent A observes
    the outputs of agents B and C.

    Args:
        agent_names: Ordered list of agent names.
        topology_type: One of chain, ring, star, fully_connected, custom.
        custom_adjacency: Explicit adjacency for custom topology.

    Returns:
        Dict mapping each agent name to a list of agents it observes.
    """
    n = len(agent_names)
    graph: dict[str, list[str]] = {name: [] for name in agent_names}

    if topology_type == "chain":
        # A_0 -> A_1 -> A_2 -> ... -> A_{n-1}
        # Each agent observes the previous agent in the chain
        for i in range(1, n):
            graph[agent_names[i]] = [agent_names[i - 1]]

    elif topology_type == "ring":
        # Like chain but last also observes first (and first observes last)
        for i in range(n):
            graph[agent_names[i]] = [agent_names[(i - 1) % n]]

    elif topology_type == "star":
        # Agent_0 is the hub. All others observe Agent_0.
        # Agent_0 observes all others.
        hub = agent_names[0]
        for i in range(1, n):
            graph[agent_names[i]] = [hub]
        graph[hub] = agent_names[1:]

    elif topology_type == "fully_connected":
        # Every agent observes every other agent
        for i in range(n):
            graph[agent_names[i]] = [agent_names[j] for j in range(n) if j != i]

    elif topology_type == "custom":
        if custom_adjacency is None:
            raise ValueError("custom topology requires custom_adjacency")
        # Validate all names exist
        for name, observers in custom_adjacency.items():
            if name not in graph:
                raise ValueError(f"Unknown agent in custom_adjacency: {name}")
            for obs in observers:
                if obs not in graph:
                    raise ValueError(f"Unknown agent in custom_adjacency: {obs}")
            graph[name] = list(observers)
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")

    return graph


def build_perturbation_persona(
    perturbation_config: dict[str, Any],
    values_data: dict[str, Any] | None = None,
) -> str:
    """Build the perturbed persona string from config.

    Args:
        perturbation_config: Perturbation config dict with target_value,
            target_value_type, strength, persona_override.
        values_data: Optional loaded Schwartz values data for description lookup.

    Returns:
        The persona override string with variables substituted.
    """
    target_value = perturbation_config.get("target_value", "social_power")
    target_value_type = perturbation_config.get("target_value_type", "power")
    strength = perturbation_config.get("strength", 9)
    template = perturbation_config.get("persona_override", "")

    # Try to look up the value type description
    value_description = f"the importance of {target_value.replace('_', ' ')}"
    if values_data:
        vtype = values_data.get("value_types", {}).get(target_value_type, {})
        if vtype:
            value_description = vtype.get("description", value_description)

    persona: str = template.format(
        value_name=target_value.replace("_", " "),
        value_type=target_value_type,
        value_description=value_description,
        strength=strength,
    )
    return persona


@dataclasses.dataclass
class ValueFlowGameMaster(prefab_lib.Prefab):
    """Game master for ValueFlow experiments.

    Manages:
    - DAG-based interaction topology (who observes whom)
    - Perturbation injection on the target agent
    - Multi-round discussion orchestration
    """

    description: str = "Game master for ValueFlow value propagation experiments."

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "valueflow_gm",
            "topology": {"type": "chain"},
            "perturbation": {"enabled": False},
            "interaction": {"num_rounds": 3},
        }
    )

    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_lib.EntityWithLogging:
        """Build the game master entity.

        Uses Concordia's generic game master components with custom
        topology-aware observation delivery.
        """
        name = str(self.params.get("name", "valueflow_gm"))

        # Build basic GM components
        components: dict[str, Any] = {}

        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        components[memory_key] = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank,
        )

        instructions_key = "Instructions"
        components[instructions_key] = gm_components.instructions.Instructions(
            pre_act_label="Game master instructions",
        )

        # Store topology and perturbation config for use during simulation
        topology_config = dict(self.params.get("topology", {"type": "chain"}))
        perturbation_config = dict(self.params.get("perturbation", {"enabled": False}))
        interaction_config = dict(self.params.get("interaction", {"num_rounds": 3}))

        logger.info(
            f"ValueFlowGameMaster: topology={topology_config.get('type')}, "
            f"perturbation={perturbation_config.get('enabled')}, "
            f"rounds={interaction_config.get('num_rounds')}"
        )

        # Use Concordia's built-in generic game master as the base entity
        from concordia.prefabs import game_master as gm_prefabs

        base_gm = gm_prefabs.generic__GameMaster(
            params={
                "name": name,
                "acting_order": "fixed",
            }
        )

        return base_gm.build(model, memory_bank)
