"""Tests for evaluation probes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from concordia.typing import entity as entity_lib

from src.evaluation.probes import (
    BooleanProbe,
    CategoricalProbe,
    NumericProbe,
    create_probe,
)


class TestCategoricalProbe:
    """Tests for CategoricalProbe."""

    def _make_probe(self, **overrides: object) -> CategoricalProbe:
        config: dict = {
            "type": "categorical",
            "categories": ["conservative", "progressive", "undecided"],
            "prompt_template": "What does {agent_name} prefer? Choose from: {categories}",
            **overrides,
        }
        return CategoricalProbe("vote_pref", config)

    def test_build_prompt_substitutes_name_and_categories(self) -> None:
        probe = self._make_probe()
        prompt = probe.build_prompt("Alice", {})

        assert "Alice" in prompt
        assert "conservative" in prompt
        assert "progressive" in prompt

    def test_build_prompt_substitutes_context_vars(self) -> None:
        probe = self._make_probe(
            prompt_template="{agent_name} in {location}: {categories}",
        )
        prompt = probe.build_prompt("Bob", {"location": "Springfield"})

        assert "Bob" in prompt
        assert "Springfield" in prompt

    def test_parse_exact_match(self) -> None:
        probe = self._make_probe()
        assert probe.parse_response("conservative") == "conservative"

    def test_parse_case_insensitive(self) -> None:
        probe = self._make_probe()
        assert probe.parse_response("PROGRESSIVE") == "progressive"

    def test_parse_substring_match(self) -> None:
        probe = self._make_probe()
        assert probe.parse_response("I think I am progressive in my views") == "progressive"

    def test_parse_no_match(self) -> None:
        probe = self._make_probe()
        assert probe.parse_response("I like pizza") is None

    def test_applies_to_role_empty_means_all(self) -> None:
        probe = self._make_probe(applies_to=[])
        assert probe.applies_to_role("voter") is True
        assert probe.applies_to_role(None) is True

    def test_applies_to_role_filters(self) -> None:
        probe = self._make_probe(applies_to=["voter"])
        assert probe.applies_to_role("voter") is True
        assert probe.applies_to_role("candidate") is False

    def test_query_calls_act_not_observe(self) -> None:
        probe = self._make_probe()
        agent = MagicMock(spec=entity_lib.Entity)
        agent.name = "Alice"
        agent.act.return_value = "conservative"

        result = probe.query(agent)

        agent.act.assert_called_once()
        agent.observe.assert_not_called()
        assert result["agent"] == "Alice"
        assert result["probe"] == "vote_pref"
        assert result["value"] == "conservative"


class TestNumericProbe:
    """Tests for NumericProbe."""

    def _make_probe(self, **overrides: object) -> NumericProbe:
        config: dict = {
            "type": "numeric",
            "min": 1,
            "max": 10,
            "prompt_template": "Rate {agent_name}'s trust from {min} to {max}.",
            **overrides,
        }
        return NumericProbe("trust", config)

    def test_build_prompt_substitutes_range(self) -> None:
        probe = self._make_probe()
        prompt = probe.build_prompt("Alice", {})

        assert "Alice" in prompt
        assert "1" in prompt
        assert "10" in prompt

    def test_parse_integer_in_range(self) -> None:
        probe = self._make_probe()
        assert probe.parse_response("I would say 7") == 7

    def test_parse_integer_at_boundaries(self) -> None:
        probe = self._make_probe()
        assert probe.parse_response("1") == 1
        assert probe.parse_response("10") == 10

    def test_parse_integer_out_of_range_skipped(self) -> None:
        probe = self._make_probe()
        # 42 is out of range [1, 10], no valid number found
        assert probe.parse_response("42") is None

    def test_parse_float(self) -> None:
        # The parser tries integers first via \b(\d+)\b.
        # "3.5" yields integer matches "3" and "5" — if both are out of the
        # integer range, the float path runs.  Use min/max where no bare
        # integer from the text matches but a float does.
        probe = self._make_probe(min=3.2, max=3.8)
        assert probe.parse_response("About 3.5 I think") == 3.5

    def test_parse_no_number(self) -> None:
        probe = self._make_probe()
        assert probe.parse_response("I don't know") is None

    def test_parse_picks_first_valid(self) -> None:
        probe = self._make_probe()
        # "100" is out of range, "5" is in range
        assert probe.parse_response("Out of 100, I'd say 5") == 5


class TestBooleanProbe:
    """Tests for BooleanProbe."""

    def _make_probe(self, **overrides: object) -> BooleanProbe:
        config: dict = {
            "type": "boolean",
            "prompt_template": "Does {agent_name} support the proposal?",
            **overrides,
        }
        return BooleanProbe("support", config)

    def test_parse_positive(self) -> None:
        probe = self._make_probe()
        assert probe.parse_response("Yes, definitely") is True

    def test_parse_negative(self) -> None:
        probe = self._make_probe()
        assert probe.parse_response("No, I don't think so") is False

    def test_parse_ambiguous(self) -> None:
        probe = self._make_probe()
        assert probe.parse_response("I'm thinking about it") is None

    def test_parse_true_keyword(self) -> None:
        probe = self._make_probe()
        assert probe.parse_response("That is true") is True

    def test_parse_false_keyword(self) -> None:
        probe = self._make_probe()
        assert probe.parse_response("That is false") is False

    def test_build_prompt(self) -> None:
        probe = self._make_probe()
        prompt = probe.build_prompt("Charlie", {})
        assert "Charlie" in prompt


class TestCreateProbe:
    """Tests for create_probe factory."""

    def test_creates_categorical(self) -> None:
        probe = create_probe("test", {"type": "categorical", "categories": ["a", "b"]})
        assert isinstance(probe, CategoricalProbe)

    def test_creates_numeric(self) -> None:
        probe = create_probe("test", {"type": "numeric", "min": 1, "max": 5})
        assert isinstance(probe, NumericProbe)

    def test_creates_boolean(self) -> None:
        probe = create_probe("test", {"type": "boolean"})
        assert isinstance(probe, BooleanProbe)

    def test_defaults_to_categorical(self) -> None:
        probe = create_probe("test", {"categories": ["x", "y"]})
        assert isinstance(probe, CategoricalProbe)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown probe type"):
            create_probe("test", {"type": "imaginary"})
