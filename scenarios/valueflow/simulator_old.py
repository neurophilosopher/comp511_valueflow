"""ValueFlow simulator with perturbation injection.

Subclasses MultiModelSimulator to override build_instances(), replacing the
perturbed agent's persona with the strongly value-shifted override string
*before* Concordia builds the agent entity.
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
from src.simulation.simulators.multi_model import MultiModelSimulator

logger = logging.getLogger(__name__)


class ValueFlowSimulator(MultiModelSimulator):
    """Simulator for ValueFlow experiments.

    Extends MultiModelSimulator with:
    - Perturbation injection: target agent's persona is replaced before build.
    - Judge model wiring: JudgedNumericProbe instances receive the judge LLM
      after setup so they can score free-form responses.
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
                "Judge model '%s' not found — judged probes will fall back to parsing", judge_key
            )
            return

        self._probe_runner.set_judge_model(model)

    def build_instances(self) -> list[prefab_lib.InstanceConfig]:
        """Build instances with perturbation injected into the target agent.

        Calls the base implementation then replaces the perturbed agent's
        'persona' param with the override string if perturbation is enabled.

        Returns:
            List of InstanceConfig objects, with the perturbed agent's persona
            already modified.
        """
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
