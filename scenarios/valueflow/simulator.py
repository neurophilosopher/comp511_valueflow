"""ValueFlow simulator with perturbation injection and result printing.

Subclasses MultiModelSimulator to:
1. Inject perturbation persona into the target agent before build.
2. Wire judge model into JudgedNumericProbe instances after setup.
3. After the simulation run, aggregate probe results by value_type,
   print per-agent value scores + system mean to terminal,
   and append an HTML results block to simulation_log.html.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from concordia.language_model import language_model as lm_lib
from concordia.typing import prefab as prefab_lib
from omegaconf import OmegaConf

from scenarios.valueflow.game_masters import build_perturbation_persona
from scenarios.valueflow.drift_graph import write_value_drift_graph
from scenarios.valueflow.metrics import (
    RunResults,
    build_html_results_block,
    print_value_scores,
)
from src.simulation.simulators.multi_model import MultiModelSimulator

logger = logging.getLogger(__name__)


class ValueFlowSimulator(MultiModelSimulator):
    """Simulator for ValueFlow experiments.

    Extends MultiModelSimulator with:
    - Perturbation injection: target agent's persona is replaced before build.
    - Judge model wiring: JudgedNumericProbe instances receive the judge LLM
      after setup so they can score free-form responses.
    - Result printing: after run(), prints value scores per agent and system
      mean to terminal, and appends an HTML block to simulation_log.html.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._vf_models: dict[str, lm_lib.LanguageModel] = {}

    def create_models(self) -> dict[str, lm_lib.LanguageModel]:
        """Create models and cache them for later judge-model wiring."""
        models = super().create_models()
        self._vf_models = models
        return models

    def setup(self) -> None:
        """Set up simulation, then wire judge model into judged probes."""
        super().setup()
        self._wire_judge_model()

    def _wire_judge_model(self) -> None:
        """Inject the judge LLM into any JudgedNumericProbe instances."""
        from src.evaluation.probes import JudgedNumericProbe

        if self._probe_runner is None or not self._vf_models:
            return

        judge_key: str = str(
            self._config.get("evaluation", {}).get("judge", {}).get("model", "_default_")
        )
        if judge_key == "_default_":
            model = next(iter(self._vf_models.values()))
        else:
            model = self._vf_models.get(judge_key)

        if model is None:
            logger.warning(
                "Judge model '%s' not found — judged probes will fall back to parsing",
                judge_key,
            )
            return

        self._probe_runner.set_judge_model(model)

    def run(self) -> str | list:
        """Run simulation, then print and save value score results.

        After the simulation completes and probes are run, this method:
        1. Reads probe_results.jsonl
        2. Aggregates question-level scores by value_type
        3. Prints per-agent scores + system mean to terminal
        4. Appends an HTML results block to simulation_log.html

        Returns:
            Simulation result (HTML string or raw log list).
        """
        result = super().run()
        self._print_and_save_results(result)
        return result

    def _print_and_save_results(self, result: str | list) -> None:
        """Aggregate probe results and print/save value scores."""
        output_dir = Path(self._config.experiment.output_dir)
        probe_results_path = output_dir / "probe_results.jsonl"

        if not probe_results_path.exists():
            logger.warning(
                "probe_results.jsonl not found at %s — skipping result printing",
                probe_results_path,
            )
            return

        try:
            run = RunResults.from_jsonl(probe_results_path)
        except Exception as e:
            logger.warning("Failed to load probe results: %s", e)
            return

        if not run.results:
            logger.warning("No probe results found — skipping result printing")
            return

        # ── Terminal output ──────────────────────────────────────────────────
        perturbation_enabled = self._config.scenario.get(
            "perturbation", {}
        ).get("enabled", False)

        title = (
            "Value Scores (Perturbed Run)"
            if perturbation_enabled
            else "Value Scores (Baseline Run)"
        )
        print_value_scores(run, title=title)

        # ── HTML output ──────────────────────────────────────────────────────
        html_block = build_html_results_block(run, title=title)

        html_path_str = self._config.simulation.logging.get("html_path")
        if not html_path_str:
            # Fall back to default location
            html_path = output_dir / "simulation_log.html"
        else:
            html_path = Path(html_path_str)

        if html_path.exists():
            try:
                with html_path.open("a", encoding="utf-8") as f:
                    f.write("\n<!-- ValueFlow Results -->\n")
                    f.write(html_block)
                logger.info("Appended value score results to %s", html_path)
            except Exception as e:
                logger.warning("Failed to append HTML results: %s", e)
        else:
            # HTML not yet written — save block to a separate file
            results_html_path = output_dir / "valueflow_results.html"
            try:
                with results_html_path.open("w", encoding="utf-8") as f:
                    f.write("<html><body>\n")
                    f.write(html_block)
                    f.write("</body></html>\n")
                print(f"Value score results saved to: {results_html_path}")
            except Exception as e:
                logger.warning("Failed to write results HTML: %s", e)

        # ── Topology/value drift graph ──────────────────────────────────────
        try:
            graph_path = write_value_drift_graph(run, self._config, output_dir)
            logger.info("Wrote value drift graph to %s", graph_path)
        except Exception as e:
            logger.warning("Failed to write value drift graph HTML: %s", e)

    def build_instances(self) -> list[prefab_lib.InstanceConfig]:
        """Build instances with perturbation injected into the target agent."""
        instances = super().build_instances()

        perturbation_config: dict[str, Any] = OmegaConf.to_container(
            self._config.scenario.get("perturbation", {}), resolve=True
        )  # type: ignore[assignment]

        if not perturbation_config.get("enabled", False):
            logger.info("ValueFlowSimulator: perturbation disabled — baseline run")
            return instances

        perturbed_index: int = int(perturbation_config.get("perturbed_agent_index", 0))
        agent_names = [e.name for e in self._config.scenario.agents.entities]

        if perturbed_index >= len(agent_names):
            raise ValueError(
                f"perturbed_agent_index {perturbed_index} is out of range "
                f"for {len(agent_names)} agents"
            )

        target_name = agent_names[perturbed_index]

        # Optionally load Schwartz values data for richer persona descriptions
        values_data: dict[str, Any] | None = None
        schwartz_path = self._config.scenario.get("data", {}).get("schwartz_values_file")
        if schwartz_path:
            path = Path(schwartz_path)
            if path.exists():
                with path.open() as f:
                    values_data = yaml.safe_load(f)
            else:
                logger.warning("Schwartz values file not found: %s", schwartz_path)

        persona_override = build_perturbation_persona(perturbation_config, values_data)

        logger.info(
            "Injecting perturbation into %s: value=%s, strength=%s",
            target_name,
            perturbation_config.get("target_value"),
            perturbation_config.get("strength"),
        )

        # Replace the perturbed agent's InstanceConfig with modified persona
        modified: list[prefab_lib.InstanceConfig] = []
        injected = False
        for instance in instances:
            if (
                instance.role == prefab_lib.Role.ENTITY
                and instance.params.get("name") == target_name
            ):
                new_params = dict(instance.params)
                new_params["persona"] = persona_override
                modified.append(
                    prefab_lib.InstanceConfig(
                        prefab=instance.prefab,
                        role=instance.role,
                        params=new_params,
                    )
                )
                injected = True
            else:
                modified.append(instance)

        if not injected:
            logger.warning(
                "Could not find agent '%s' in instances — perturbation not applied. "
                "Available entities: %s",
                target_name,
                [i.params.get("name") for i in instances if i.role == prefab_lib.Role.ENTITY],
            )

        return modified
