"""Tests for JudgedNumericProbe and related ProbeRunner wiring (Phase 4)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from concordia.typing import entity as entity_lib
from omegaconf import OmegaConf

from src.evaluation.probe_runner import ProbeRunner
from src.evaluation.probes import (
    JudgedNumericProbe,
    NumericProbe,
    create_probe,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_probe(**overrides: object) -> JudgedNumericProbe:
    config: dict = {
        "type": "judged_numeric",
        "min": 0,
        "max": 10,
        "prompt_template": "How does {agent_name} feel about freedom?",
        "_judge_system_prompt": "Rate endorsement 0-10.",
        **overrides,
    }
    return JudgedNumericProbe("freedom", config)


def _make_agent(name: str = "Agent_0", response: str = "I deeply value freedom.") -> MagicMock:
    agent = MagicMock(spec=entity_lib.Entity)
    agent.name = name
    agent.act.return_value = response
    return agent


def _make_judge(score_text: str = "8") -> MagicMock:
    judge = MagicMock()
    judge.sample_text.return_value = score_text
    return judge


# ── JudgedNumericProbe construction ──────────────────────────────────────────


class TestJudgedNumericProbeInit:
    def test_defaults(self) -> None:
        probe = _make_probe()
        assert probe.name == "freedom"
        assert probe.min_value == 0
        assert probe.max_value == 10
        assert probe.judge_model is None

    def test_judge_system_prompt_stored(self) -> None:
        probe = _make_probe(_judge_system_prompt="Be an expert.")
        assert probe.judge_system_prompt == "Be an expert."

    def test_missing_judge_system_prompt_defaults_empty(self) -> None:
        probe = JudgedNumericProbe("x", {"type": "judged_numeric", "prompt_template": "Q?"})
        assert probe.judge_system_prompt == ""

    def test_applies_to_role(self) -> None:
        probe = _make_probe(applies_to=["participant"])
        assert probe.applies_to_role("participant") is True
        assert probe.applies_to_role("observer") is False


# ── build_prompt ──────────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_substitutes_agent_name(self) -> None:
        probe = _make_probe()
        prompt = probe.build_prompt("Alice", {})
        assert "Alice" in prompt
        assert "{agent_name}" not in prompt

    def test_substitutes_context_variables(self) -> None:
        probe = _make_probe(prompt_template="Hello {agent_name} in {city}.")
        prompt = probe.build_prompt("Bob", {"city": "Paris"})
        assert "Paris" in prompt

    def test_no_numeric_scale_in_prompt(self) -> None:
        """Prompt should not mention 0-10 scale (that's the judge's concern)."""
        probe = _make_probe()
        prompt = probe.build_prompt("Agent_0", {})
        assert "0-10" not in prompt
        assert "scale" not in prompt.lower()


# ── parse_response (fallback) ─────────────────────────────────────────────────


class TestParseResponse:
    def test_extracts_integer_in_range(self) -> None:
        probe = _make_probe()
        assert probe.parse_response("I'd say 7 out of 10") == 7

    def test_returns_none_for_out_of_range(self) -> None:
        probe = _make_probe()
        assert probe.parse_response("100") is None

    def test_returns_none_for_no_number(self) -> None:
        probe = _make_probe()
        assert probe.parse_response("I have no idea") is None

    def test_picks_first_in_range(self) -> None:
        probe = _make_probe()
        # 42 out of range, 6 in range
        assert probe.parse_response("out of 42 possible, I'd say 6") == 6

    def test_boundary_values(self) -> None:
        probe = _make_probe()
        assert probe.parse_response("0") == 0
        assert probe.parse_response("10") == 10


# ── query() — silent observation guarantee ────────────────────────────────────


class TestQuerySilent:
    def test_observe_never_called(self) -> None:
        probe = _make_probe()
        agent = _make_agent()
        probe.query(agent)
        agent.observe.assert_not_called()

    def test_act_called_exactly_once(self) -> None:
        probe = _make_probe()
        agent = _make_agent()
        probe.query(agent)
        agent.act.assert_called_once()


# ── query() — without judge model (fallback) ─────────────────────────────────


class TestQueryFallback:
    def test_returns_parsed_value_from_free_response(self) -> None:
        probe = _make_probe()
        agent = _make_agent(response="I'd say about 6 out of 10.")
        result = probe.query(agent)
        assert result["value"] == 6

    def test_returns_none_when_no_number_in_response(self) -> None:
        probe = _make_probe()
        agent = _make_agent(response="Freedom is very important to me.")
        result = probe.query(agent)
        assert result["value"] is None

    def test_result_contains_required_keys(self) -> None:
        probe = _make_probe()
        result = probe.query(_make_agent())
        assert "agent" in result
        assert "probe" in result
        assert "raw_response" in result
        assert "judge_response" in result
        assert "value" in result

    def test_raw_response_stored(self) -> None:
        probe = _make_probe()
        agent = _make_agent(response="My free-form answer.")
        result = probe.query(agent)
        assert result["raw_response"] == "My free-form answer."


# ── query() — with judge model ────────────────────────────────────────────────


class TestQueryWithJudge:
    def test_judge_sample_text_called(self) -> None:
        probe = _make_probe()
        probe.judge_model = _make_judge("7")
        probe.query(_make_agent())
        probe.judge_model.sample_text.assert_called_once()

    def test_judge_score_returned_as_value(self) -> None:
        probe = _make_probe()
        probe.judge_model = _make_judge("8")
        result = probe.query(_make_agent())
        assert result["value"] == 8

    def test_judge_response_stored(self) -> None:
        probe = _make_probe()
        probe.judge_model = _make_judge("5")
        result = probe.query(_make_agent())
        assert result["judge_response"] == "5"

    def test_judge_prompt_includes_system_prompt(self) -> None:
        probe = _make_probe(_judge_system_prompt="Expert psychologist.")
        probe.judge_model = _make_judge("4")
        probe.query(_make_agent(response="Some free answer."))
        call_args = probe.judge_model.sample_text.call_args
        prompt_text = call_args.kwargs.get("prompt") or call_args.args[0]
        assert "Expert psychologist." in prompt_text

    def test_judge_prompt_includes_agent_response(self) -> None:
        probe = _make_probe()
        probe.judge_model = _make_judge("6")
        probe.query(_make_agent(response="I cherish freedom deeply."))
        call_args = probe.judge_model.sample_text.call_args
        prompt_text = call_args.kwargs.get("prompt") or call_args.args[0]
        assert "I cherish freedom deeply." in prompt_text

    def test_judge_out_of_range_returns_none(self) -> None:
        probe = _make_probe()
        probe.judge_model = _make_judge("99")  # out of range
        result = probe.query(_make_agent())
        assert result["value"] is None

    def test_observe_still_not_called_with_judge(self) -> None:
        probe = _make_probe()
        probe.judge_model = _make_judge("5")
        agent = _make_agent()
        probe.query(agent)
        agent.observe.assert_not_called()


# ── create_probe factory ──────────────────────────────────────────────────────


class TestCreateProbeFactory:
    def test_creates_judged_numeric(self) -> None:
        probe = create_probe(
            "test",
            {"type": "judged_numeric", "min": 0, "max": 10, "prompt_template": "Q?"},
        )
        assert isinstance(probe, JudgedNumericProbe)

    def test_judged_numeric_distinct_from_numeric(self) -> None:
        probe = create_probe("test", {"type": "judged_numeric", "prompt_template": "Q?"})
        assert not isinstance(probe, NumericProbe)


# ── ProbeRunner judge wiring ──────────────────────────────────────────────────


def _eval_config_with_judged(system_prompt: str = "Rate it.") -> dict:
    return {
        "name": "test_judged_eval",
        "judge": {"system_prompt": system_prompt, "model": "_default_"},
        "metrics": {
            "freedom": {
                "type": "judged_numeric",
                "min": 0,
                "max": 10,
                "prompt_template": "How does {agent_name} feel about freedom?",
                "applies_to": ["participant"],
            },
            "trust": {
                "type": "numeric",
                "min": 0,
                "max": 10,
                "prompt_template": "Rate {agent_name}'s trust.",
            },
        },
    }


class TestProbeRunnerJudgeWiring:
    def test_judge_system_prompt_injected_into_judged_probes(self, tmp_path: Path) -> None:
        config = OmegaConf.create(_eval_config_with_judged("Be an expert."))
        runner = ProbeRunner(config, tmp_path)

        judged = [p for p in runner.probes if isinstance(p, JudgedNumericProbe)]
        assert len(judged) == 1
        assert judged[0].judge_system_prompt == "Be an expert."

    def test_numeric_probes_unaffected_by_judge_config(self, tmp_path: Path) -> None:
        config = OmegaConf.create(_eval_config_with_judged())
        runner = ProbeRunner(config, tmp_path)

        numeric = [
            p
            for p in runner.probes
            if isinstance(p, NumericProbe) and not isinstance(p, JudgedNumericProbe)
        ]
        assert len(numeric) == 1

    def test_set_judge_model_wires_judged_probes(self, tmp_path: Path) -> None:
        config = OmegaConf.create(_eval_config_with_judged())
        runner = ProbeRunner(config, tmp_path)
        model = _make_judge()

        runner.set_judge_model(model)

        judged = [p for p in runner.probes if isinstance(p, JudgedNumericProbe)]
        assert all(p.judge_model is model for p in judged)

    def test_set_judge_model_does_not_modify_numeric_probes(self, tmp_path: Path) -> None:
        config = OmegaConf.create(_eval_config_with_judged())
        runner = ProbeRunner(config, tmp_path)
        runner.set_judge_model(_make_judge())

        numeric = [
            p
            for p in runner.probes
            if isinstance(p, NumericProbe) and not isinstance(p, JudgedNumericProbe)
        ]
        for p in numeric:
            assert not hasattr(p, "judge_model") or p.judge_model is None  # type: ignore[attr-defined]

    def test_judged_probe_no_system_prompt_when_judge_config_absent(self, tmp_path: Path) -> None:
        config = OmegaConf.create(
            {
                "name": "no_judge",
                "metrics": {
                    "val": {
                        "type": "judged_numeric",
                        "prompt_template": "Q?",
                    }
                },
            }
        )
        runner = ProbeRunner(config, tmp_path)
        judged = [p for p in runner.probes if isinstance(p, JudgedNumericProbe)]
        assert judged[0].judge_system_prompt == ""
