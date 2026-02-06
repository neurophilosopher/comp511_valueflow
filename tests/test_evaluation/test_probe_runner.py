"""Tests for ProbeRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from concordia.typing import entity as entity_lib
from omegaconf import OmegaConf

from src.evaluation.probe_runner import ProbeRunner


def _make_eval_config(**overrides: object) -> dict:
    """Create a minimal evaluation config."""
    base = {
        "name": "test_eval",
        "metrics": {
            "vote_pref": {
                "type": "categorical",
                "categories": ["A", "B", "C"],
                "prompt_template": "Who does {agent_name} prefer?",
            },
        },
    }
    base.update(overrides)
    return base


def _make_agent(name: str, response: str = "A") -> MagicMock:
    agent = MagicMock(spec=entity_lib.Entity)
    agent.name = name
    agent.act.return_value = response
    return agent


class TestProbeRunnerInit:
    """Tests for ProbeRunner initialization."""

    def test_builds_probes_from_config(self, tmp_path: Path) -> None:
        config = OmegaConf.create(_make_eval_config())
        runner = ProbeRunner(config, tmp_path)

        assert len(runner.probes) == 1
        assert runner.probes[0].name == "vote_pref"

    def test_skips_metrics_without_prompt_template(self, tmp_path: Path) -> None:
        config = OmegaConf.create(
            {
                "name": "test",
                "metrics": {
                    "aggregate_score": {"type": "numeric", "min": 0, "max": 100},
                    "vote_pref": {
                        "type": "categorical",
                        "categories": ["A", "B"],
                        "prompt_template": "Choose: {categories}",
                    },
                },
            }
        )
        runner = ProbeRunner(config, tmp_path)

        # aggregate_score has no prompt_template, should be skipped
        assert len(runner.probes) == 1
        assert runner.probes[0].name == "vote_pref"

    def test_empty_metrics(self, tmp_path: Path) -> None:
        config = OmegaConf.create({"name": "test", "metrics": {}})
        runner = ProbeRunner(config, tmp_path)
        assert len(runner.probes) == 0

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "dir"
        config = OmegaConf.create(_make_eval_config())
        runner = ProbeRunner(config, output)

        assert output.exists()
        assert runner.results_file == output / "probe_results.jsonl"


class TestProbeRunnerRunProbes:
    """Tests for run_probes execution."""

    def test_run_probes_sequential(self, tmp_path: Path) -> None:
        config = OmegaConf.create(_make_eval_config())
        runner = ProbeRunner(config, tmp_path)

        agents = [_make_agent("Alice", "A"), _make_agent("Bob", "B")]
        results = runner.run_probes(agents, step=1, parallel=False)

        assert len(results) == 2
        assert results[0]["agent"] == "Alice"
        assert results[0]["value"] == "A"
        assert results[0]["step"] == 1
        assert results[1]["agent"] == "Bob"
        assert results[1]["value"] == "B"

    def test_run_probes_parallel(self, tmp_path: Path) -> None:
        config = OmegaConf.create(_make_eval_config())
        runner = ProbeRunner(config, tmp_path)

        agents = [_make_agent("Alice", "A"), _make_agent("Bob", "B")]
        results = runner.run_probes(agents, step=2, parallel=True)

        assert len(results) == 2
        agent_names = {r["agent"] for r in results}
        assert agent_names == {"Alice", "Bob"}

    def test_role_filtering(self, tmp_path: Path) -> None:
        eval_cfg = _make_eval_config()
        eval_cfg["metrics"]["vote_pref"]["applies_to"] = ["voter"]
        config = OmegaConf.create(eval_cfg)
        runner = ProbeRunner(
            config,
            tmp_path,
            role_mapping={"Alice": "voter", "Bob": "candidate"},
        )

        agents = [_make_agent("Alice", "A"), _make_agent("Bob", "B")]
        results = runner.run_probes(agents, step=0, parallel=False)

        # Only Alice (voter) should be probed
        assert len(results) == 1
        assert results[0]["agent"] == "Alice"

    def test_saves_results_to_jsonl(self, tmp_path: Path) -> None:
        config = OmegaConf.create(_make_eval_config())
        runner = ProbeRunner(config, tmp_path)

        agents = [_make_agent("Alice", "A")]
        runner.run_probes(agents, step=0, parallel=False)

        lines = runner.results_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["agent"] == "Alice"
        assert data["step"] == 0

    def test_appends_across_steps(self, tmp_path: Path) -> None:
        config = OmegaConf.create(_make_eval_config())
        runner = ProbeRunner(config, tmp_path)

        agents = [_make_agent("Alice", "A")]
        runner.run_probes(agents, step=0, parallel=False)
        runner.run_probes(agents, step=1, parallel=False)

        lines = runner.results_file.read_text().strip().split("\n")
        assert len(lines) == 2


class TestProbeRunnerSummary:
    """Tests for get_results_summary."""

    def test_empty_summary(self, tmp_path: Path) -> None:
        config = OmegaConf.create(_make_eval_config())
        runner = ProbeRunner(config, tmp_path)

        summary = runner.get_results_summary()
        assert summary["total_queries"] == 0

    def test_summary_counts(self, tmp_path: Path) -> None:
        config = OmegaConf.create(_make_eval_config())
        runner = ProbeRunner(config, tmp_path)

        agents = [_make_agent("Alice", "A"), _make_agent("Bob", "unknown")]
        runner.run_probes(agents, step=0, parallel=False)

        summary = runner.get_results_summary()
        assert summary["total_queries"] == 2
        assert summary["probes"]["vote_pref"]["count"] == 2
        # "A" matches, "unknown" does not
        assert summary["probes"]["vote_pref"]["valid"] == 1
        assert summary["agents"]["Alice"] == 1
        assert summary["agents"]["Bob"] == 1

    def test_get_all_results(self, tmp_path: Path) -> None:
        config = OmegaConf.create(_make_eval_config())
        runner = ProbeRunner(config, tmp_path)

        agents = [_make_agent("Alice", "A")]
        runner.run_probes(agents, step=0, parallel=False)

        all_results = runner.get_all_results()
        assert len(all_results) == 1
        assert all_results[0]["agent"] == "Alice"
