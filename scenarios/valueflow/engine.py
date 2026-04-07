"""ValueFlow simulation engine with topology-aware observation delivery."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import termcolor
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from concordia.utils import concurrency

from scenarios.valueflow.game_masters import build_topology_graph

logger = logging.getLogger(__name__)

_PRINT_COLOR = "yellow"


class ValueFlowEngine(engine_lib.Engine):
    """Engine for ValueFlow value propagation experiments.

    Implements topology-aware observation delivery:
    - Each round, agents observe only their topological neighbors' previous outputs
    - All agents act in parallel within a round
    - Runs for num_rounds total rounds

    Probe timing (matches paper §5.3):
    - Step 0: fired BEFORE any agent acts — true pre-interaction baseline
    - Step N (e.g. 10): fired AFTER the final round

    Set probe_steps: [0, 10] in the scenario YAML to get both checkpoints.
    """

    def __init__(
        self,
        topology_config: dict[str, Any],
        interaction_config: dict[str, Any],
    ) -> None:
        self._topology_config = topology_config
        self._interaction_config = interaction_config
        self._topology_graph: dict[str, list[str]] = {}

    # -- Engine abstract methods (unused — run_loop is overridden) --

    def make_observation(
        self,
        game_master: entity_lib.Entity,
        entity: entity_lib.Entity,
    ) -> str:
        raise NotImplementedError("ValueFlowEngine uses run_loop directly.")

    def next_acting(
        self,
        game_master: entity_lib.Entity,
        entities: Sequence[entity_lib.Entity],
    ) -> tuple[entity_lib.Entity, entity_lib.ActionSpec]:
        raise NotImplementedError("ValueFlowEngine uses run_loop directly.")

    def resolve(
        self,
        game_master: entity_lib.Entity,
        event: str,
    ) -> None:
        raise NotImplementedError("ValueFlowEngine uses run_loop directly.")

    def terminate(self, game_master: entity_lib.Entity) -> bool:
        raise NotImplementedError("ValueFlowEngine uses run_loop directly.")

    def next_game_master(
        self,
        game_master: entity_lib.Entity,
        game_masters: Sequence[entity_lib.Entity],
    ) -> entity_lib.Entity:
        raise NotImplementedError("ValueFlowEngine uses run_loop directly.")

    def run_loop(
        self,
        game_masters: Sequence[entity_lib.Entity],
        entities: Sequence[entity_lib.Entity],
        premise: str = "",
        max_steps: int = 100,
        verbose: bool = False,
        log: list[Mapping[str, Any]] | None = None,
        checkpoint_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Run the ValueFlow simulation loop.

        Probe timing:
          checkpoint_callback(0) is called BEFORE round 1 — agents have
          observed the premise but have NOT yet acted. This is the correct
          pre-interaction probe point described in the paper (§5.3).

          checkpoint_callback(round_num) is called AFTER each round completes.
          Use probe_steps: [0, 10] in the YAML to capture pre- and post-state.

        Args:
            game_masters: List of game masters (first one used for GM name).
            entities: List of agent entities.
            premise: Initial premise observed by all agents.
            max_steps: Ignored — num_rounds from interaction_config is used.
            verbose: Print Concordia-compatible event output.
            log: List to append log entries to.
            checkpoint_callback: Called at step 0 (pre-interaction) and after
                each round with the round number.
        """
        if not entities:
            raise ValueError("No entities provided.")
        if not game_masters:
            raise ValueError("No game masters provided.")

        gm = game_masters[0]
        gm_name = gm.name

        # Build topology graph from entity names
        agent_names = [e.name for e in entities]
        topology_type = self._topology_config.get("type", "chain")
        custom_adjacency = self._topology_config.get("custom_adjacency")
        self._topology_graph = build_topology_graph(agent_names, topology_type, custom_adjacency)

        num_rounds = self._interaction_config.get("num_rounds", 3)
        round_prompt_template = self._interaction_config.get(
            "round_prompt",
            "Round {round_num}: Share your perspective on the discussion topic.",
        )

        logger.info(
            "ValueFlowEngine: topology=%s, num_rounds=%d, agents=%s",
            topology_type,
            num_rounds,
            agent_names,
        )

        # Deliver premise to all agents (before any acting)
        if premise:
            for entity in entities:
                entity.observe(premise)

        # ── Step 0: pre-interaction probe (paper §5.3) ──────────────────────
        # Agents have seen the premise but have NOT acted yet.
        # This is the true baseline the paper describes.
        if checkpoint_callback is not None:
            checkpoint_callback(0)

        previous_outputs: dict[str, str] = {}

        for round_num in range(1, num_rounds + 1):
            call_to_action = round_prompt_template.format(round_num=round_num)
            action_spec = entity_lib.ActionSpec(
                call_to_action=call_to_action,
                output_type=entity_lib.OutputType.FREE,
            )

            # Snapshot previous outputs for this round's closures
            _prev = dict(previous_outputs)

            def _agent_step(
                entity: entity_lib.Entity,
                spec: entity_lib.ActionSpec = action_spec,
                prev_outputs: dict[str, str] = _prev,
                _round: int = round_num,
            ) -> dict[str, Any]:
                """Deliver topology-filtered observation, then act."""
                neighbors = self._topology_graph.get(entity.name, [])
                observation = ""
                if neighbors and prev_outputs:
                    parts = [f"{n}: {prev_outputs[n]}" for n in neighbors if n in prev_outputs]
                    if parts:
                        observation = "\n".join(parts)

                if observation:
                    entity.observe(observation)

                raw_action = entity.act(spec)
                return {
                    "entity": entity.name,
                    "round": _round,
                    "observation": observation,
                    "raw_action": raw_action,
                }

            tasks = {entity.name: functools.partial(_agent_step, entity) for entity in entities}
            step_results = concurrency.run_tasks(tasks)

            # Update previous outputs (new dict — does not affect captured _prev)
            previous_outputs = {name: result["raw_action"] for name, result in step_results.items()}

            if verbose:
                print(termcolor.colored("Terminate? No", _PRINT_COLOR))
                for entity in entities:
                    result = step_results[entity.name]
                    if result["observation"]:
                        print(
                            termcolor.colored(
                                f"Entity {entity.name} observed: {result['observation']}",
                                _PRINT_COLOR,
                            )
                        )
                    print(
                        termcolor.colored(
                            f"Entity {entity.name} chose action: {result['raw_action']}",
                            _PRINT_COLOR,
                        )
                    )
                    summary = result["raw_action"][:120]
                    print(
                        termcolor.colored(
                            f"The resolved event was: {entity.name} said: {summary}",
                            _PRINT_COLOR,
                        )
                    )

            if log is not None:
                log_entry: dict[str, Any] = {
                    "Step": round_num,
                    "Summary": f"Round {round_num} {gm_name}",
                    gm_name: {"round_outputs": step_results},
                }
                for entity_name, result in step_results.items():
                    log_entry[f"Entity [{entity_name}]"] = result
                log.append(log_entry)

            if checkpoint_callback is not None:
                checkpoint_callback(round_num)

        if verbose:
            print(termcolor.colored("Terminate? Yes", _PRINT_COLOR))

    @property
    def topology_graph(self) -> dict[str, list[str]]:
        """Return the topology graph (populated after run_loop is called)."""
        return self._topology_graph
