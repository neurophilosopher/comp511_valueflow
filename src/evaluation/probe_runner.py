"""Probe runner for orchestrating evaluation probes during simulation.

The ProbeRunner manages probe execution, collecting results from agents
at configurable intervals and saving them to JSONL format.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from src.evaluation.probes import Probe, create_probe

if TYPE_CHECKING:
    from concordia.typing import entity as entity_lib

logger = logging.getLogger(__name__)


class ProbeRunner:
    """Manages probe execution during simulation.

    Attributes:
        probes: List of Probe instances to run.
        output_path: Path to save probe results.
        role_mapping: Dict mapping agent names to roles.
    """

    def __init__(
        self,
        config: DictConfig,
        output_dir: Path | str,
        role_mapping: dict[str, str] | None = None,
    ):
        """Initialize the probe runner.

        Args:
            config: Evaluation configuration (from config/evaluation/*.yaml).
            output_dir: Directory to save probe results.
            role_mapping: Dict mapping agent names to their roles.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "probe_results.jsonl"
        self.role_mapping = role_mapping or {}
        self.probes = self._build_probes(config)
        self._results: list[dict[str, Any]] = []

    def _build_probes(self, config: DictConfig) -> list[Probe]:
        """Build probe instances from configuration.

        Args:
            config: Evaluation configuration with 'metrics' section.

        Returns:
            List of Probe instances.
        """
        probes = []
        metrics = config.get("metrics", {})

        for name, metric_config in metrics.items():
            try:
                # Convert OmegaConf to dict if needed
                if hasattr(metric_config, "items"):
                    metric_dict = dict(metric_config)
                else:
                    metric_dict = metric_config

                # Skip aggregate-style metrics (no prompt_template = not a probe)
                if "prompt_template" not in metric_dict:
                    logger.debug(f"Skipping '{name}': no prompt_template (aggregate metric)")
                    continue

                probe = create_probe(name, metric_dict)
                probes.append(probe)
                logger.debug(
                    f"Created probe: {name} (type: {metric_dict.get('type', 'categorical')})"
                )
            except Exception as e:
                logger.warning(f"Failed to create probe '{name}': {e}")

        logger.info(f"Initialized {len(probes)} probes")
        return probes

    def set_role_mapping(self, role_mapping: dict[str, str]) -> None:
        """Update the role mapping.

        Args:
            role_mapping: Dict mapping agent names to their roles.
        """
        self.role_mapping = role_mapping

    def _get_agent_role(self, agent: entity_lib.Entity) -> str | None:
        """Get the role for an agent.

        Args:
            agent: The agent entity.

        Returns:
            Role string or None if not found.
        """
        return self.role_mapping.get(agent.name)

    def _run_probe_on_agent(
        self,
        probe: Probe,
        agent: entity_lib.Entity,
        step: int,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run a single probe on a single agent.

        Args:
            probe: The probe to run.
            agent: The agent to query.
            step: Current simulation step.
            context: Additional context for the probe.

        Returns:
            Result dict or None if probe doesn't apply.
        """
        role = self._get_agent_role(agent)

        if not probe.applies_to_role(role):
            return None

        try:
            result = probe.query(agent, context)
            result["step"] = step
            result["role"] = role
            return result
        except Exception as e:
            logger.warning(f"Probe '{probe.name}' failed on agent '{agent.name}': {e}")
            return {
                "agent": agent.name,
                "probe": probe.name,
                "step": step,
                "role": role,
                "error": str(e),
                "value": None,
            }

    def run_probes(
        self,
        agents: list[entity_lib.Entity],
        step: int,
        context: dict[str, Any] | None = None,
        parallel: bool = True,
    ) -> list[dict[str, Any]]:
        """Run all applicable probes on all agents.

        Args:
            agents: List of agent entities to probe.
            step: Current simulation step.
            context: Additional context for probes (e.g., candidate names).
            parallel: Whether to run probes in parallel.

        Returns:
            List of result dicts.
        """
        results: list[dict[str, Any]] = []

        if parallel:
            results = self._run_probes_parallel(agents, step, context)
        else:
            results = self._run_probes_sequential(agents, step, context)

        # Save results
        self._save_results(results)
        self._results.extend(results)

        logger.info(f"Step {step}: Ran {len(results)} probe queries")
        return results

    def _run_probes_sequential(
        self,
        agents: list[entity_lib.Entity],
        step: int,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run probes sequentially (deterministic order).

        Args:
            agents: List of agent entities.
            step: Current simulation step.
            context: Additional context for probes.

        Returns:
            List of result dicts.
        """
        results = []
        for agent in agents:
            for probe in self.probes:
                result = self._run_probe_on_agent(probe, agent, step, context)
                if result:
                    results.append(result)
        return results

    def _run_probes_parallel(
        self,
        agents: list[entity_lib.Entity],
        step: int,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run probes in parallel (faster but non-deterministic order).

        Args:
            agents: List of agent entities.
            step: Current simulation step.
            context: Additional context for probes.

        Returns:
            List of result dicts.
        """
        results = []

        with ThreadPoolExecutor() as executor:
            futures = {}
            for agent in agents:
                for probe in self.probes:
                    if probe.applies_to_role(self._get_agent_role(agent)):
                        future = executor.submit(
                            self._run_probe_on_agent, probe, agent, step, context
                        )
                        futures[future] = (agent.name, probe.name)

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        return results

    def _save_results(self, results: list[dict[str, Any]]) -> None:
        """Append results to JSONL file.

        Args:
            results: List of result dicts to save.
        """
        with self.results_file.open("a") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

    def get_all_results(self) -> list[dict[str, Any]]:
        """Get all collected results.

        Returns:
            List of all result dicts from this run.
        """
        return self._results

    def get_results_summary(self) -> dict[str, Any]:
        """Get a summary of probe results.

        Returns:
            Summary dict with counts and aggregations.
        """
        if not self._results:
            return {"total_queries": 0, "probes": {}, "agents": {}}

        summary: dict[str, Any] = {
            "total_queries": len(self._results),
            "probes": {},
            "agents": {},
        }

        # Count by probe
        for result in self._results:
            probe_name = result.get("probe", "unknown")
            if probe_name not in summary["probes"]:
                summary["probes"][probe_name] = {"count": 0, "valid": 0}
            summary["probes"][probe_name]["count"] += 1
            if result.get("value") is not None:
                summary["probes"][probe_name]["valid"] += 1

        # Count by agent
        for result in self._results:
            agent_name = result.get("agent", "unknown")
            if agent_name not in summary["agents"]:
                summary["agents"][agent_name] = 0
            summary["agents"][agent_name] += 1

        return summary
