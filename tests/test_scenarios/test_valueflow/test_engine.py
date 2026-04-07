"""Tests for ValueFlowEngine topology-aware observation delivery (Phase 2)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib

from scenarios.valueflow.engine import ValueFlowEngine


def _make_entity(name: str, response: str = "I value this deeply.") -> MagicMock:
    e = MagicMock(spec=entity_lib.Entity)
    e.name = name
    e.act.return_value = response
    return e


def _make_gm(name: str = "gm") -> MagicMock:
    gm = MagicMock(spec=entity_lib.Entity)
    gm.name = name
    return gm


def _chain_engine(num_rounds: int = 2) -> ValueFlowEngine:
    return ValueFlowEngine(
        topology_config={"type": "chain"},
        interaction_config={"num_rounds": num_rounds, "round_prompt": "Round {round_num}: speak."},
    )


class TestValueFlowEngineType:
    def test_inherits_from_engine_abc(self) -> None:
        assert isinstance(_chain_engine(), engine_lib.Engine)

    def test_abstract_methods_raise(self) -> None:
        engine = _chain_engine()
        gm = _make_gm()
        with pytest.raises(NotImplementedError):
            engine.make_observation(gm, _make_entity("A"))
        with pytest.raises(NotImplementedError):
            engine.next_acting(gm, [])
        with pytest.raises(NotImplementedError):
            engine.resolve(gm, "event")
        with pytest.raises(NotImplementedError):
            engine.terminate(gm)
        with pytest.raises(NotImplementedError):
            engine.next_game_master(gm, [gm])


class TestRunLoopBasics:
    def test_empty_entities_raises(self) -> None:
        engine = _chain_engine()
        with pytest.raises(ValueError, match="No entities"):
            engine.run_loop(game_masters=[_make_gm()], entities=[])

    def test_empty_game_masters_raises(self) -> None:
        engine = _chain_engine()
        with pytest.raises(ValueError, match="No game masters"):
            engine.run_loop(game_masters=[], entities=[_make_entity("A")])

    def test_premise_observed_by_all_agents(self) -> None:
        engine = _chain_engine(num_rounds=1)
        entities = [_make_entity("A0"), _make_entity("A1")]
        engine.run_loop(
            game_masters=[_make_gm()],
            entities=entities,
            premise="Welcome to the discussion.",
        )
        for e in entities:
            e.observe.assert_any_call("Welcome to the discussion.")

    def test_each_agent_acts_once_per_round(self) -> None:
        n_rounds = 3
        engine = _chain_engine(num_rounds=n_rounds)
        entities = [_make_entity("A0"), _make_entity("A1"), _make_entity("A2")]
        engine.run_loop(game_masters=[_make_gm()], entities=entities, premise="")
        for e in entities:
            assert e.act.call_count == n_rounds

    def test_zero_rounds_no_act_calls(self) -> None:
        engine = _chain_engine(num_rounds=0)
        entity = _make_entity("A0")
        engine.run_loop(game_masters=[_make_gm()], entities=[entity], premise="")
        entity.act.assert_not_called()


class TestStepZeroProbe:
    """Verify the pre-interaction step-0 probe fires before any agent acts.

    This is the key fix for paper §5.3 compliance: agents must be probed
    before they have acted, not after round 1.
    """

    def test_step_zero_callback_called_before_any_act(self) -> None:
        """checkpoint_callback(0) must be the very first callback call."""
        engine = _chain_engine(num_rounds=2)
        entity = _make_entity("A0")
        callback_calls: list[int] = []

        def tracking_callback(step: int) -> None:
            # Record act count at the moment of each callback
            callback_calls.append((step, entity.act.call_count))

        engine.run_loop(
            game_masters=[_make_gm()],
            entities=[entity],
            checkpoint_callback=tracking_callback,
            premise="",
        )

        # First callback must be step 0 and act must not have been called yet
        assert callback_calls[0][0] == 0, "First callback must be step 0"
        assert callback_calls[0][1] == 0, "Step-0 callback must fire before any act()"

    def test_step_zero_fires_even_with_zero_rounds(self) -> None:
        """Step-0 probe fires even when num_rounds=0."""
        engine = _chain_engine(num_rounds=0)
        entity = _make_entity("A0")
        calls: list[int] = []
        engine.run_loop(
            game_masters=[_make_gm()],
            entities=[entity],
            checkpoint_callback=calls.append,
            premise="",
        )
        assert 0 in calls

    def test_callback_sequence_is_0_then_1_through_n(self) -> None:
        """Full callback sequence: [0, 1, 2, ..., num_rounds]."""
        n_rounds = 3
        engine = _chain_engine(num_rounds=n_rounds)
        calls: list[int] = []
        engine.run_loop(
            game_masters=[_make_gm()],
            entities=[_make_entity("A0")],
            checkpoint_callback=calls.append,
            premise="",
        )
        assert calls == [0, 1, 2, 3]

    def test_no_observations_delivered_at_step_zero(self) -> None:
        """Agents have not yet received any neighbor messages at step 0."""
        engine = _chain_engine(num_rounds=1)
        a0 = _make_entity("A0", response="A0 output")
        a1 = _make_entity("A1", response="A1 output")

        step0_a1_observe_count = 0

        def callback(step: int) -> None:
            nonlocal step0_a1_observe_count
            if step == 0:
                # Record how many observe calls have happened on A1 so far
                # (only the premise observe should have happened, not neighbor obs)
                step0_a1_observe_count = a1.observe.call_count

        engine.run_loop(
            game_masters=[_make_gm()],
            entities=[a0, a1],
            checkpoint_callback=callback,
            premise="",  # no premise → 0 observe calls before step-0
        )

        # With no premise, A1 should have 0 observe calls at step 0
        assert step0_a1_observe_count == 0

    def test_premise_in_memory_at_step_zero(self) -> None:
        """Premise IS observed before step-0 probe — consistent with paper."""
        engine = _chain_engine(num_rounds=1)
        entity = _make_entity("A0")
        observe_count_at_step0 = 0

        def callback(step: int) -> None:
            nonlocal observe_count_at_step0
            if step == 0:
                observe_count_at_step0 = entity.observe.call_count

        engine.run_loop(
            game_masters=[_make_gm()],
            entities=[entity],
            checkpoint_callback=callback,
            premise="Welcome.",
        )

        # Exactly 1 observe call (the premise) should exist at step 0
        assert observe_count_at_step0 == 1


class TestTopologyFilteredObservations:
    """Verify that agents only observe their topological neighbors' outputs."""

    def test_chain_first_round_no_neighbor_observations(self) -> None:
        """In round 1 there are no previous outputs, so no neighbor observe calls
        (premise is absent, so no observe at all in round 1)."""
        engine = _chain_engine(num_rounds=1)
        entities = [_make_entity("A0"), _make_entity("A1"), _make_entity("A2")]
        engine.run_loop(game_masters=[_make_gm()], entities=entities, premise="")
        for e in entities:
            e.observe.assert_not_called()

    def test_chain_second_round_delivers_neighbor_output(self) -> None:
        """In round 2, A1 should observe A0's round-1 output; A0 gets nothing."""
        engine = _chain_engine(num_rounds=2)
        a0 = _make_entity("A0", response="A0 says something.")
        a1 = _make_entity("A1", response="A1 says something.")

        engine.run_loop(game_masters=[_make_gm()], entities=[a0, a1], premise="")

        # A0 has no neighbors in chain topology — never observed
        a0.observe.assert_not_called()

        # A1 should have been called with a string containing A0's output
        a1_observe_calls = [str(c) for c in a1.observe.call_args_list]
        assert any("A0 says something." in c for c in a1_observe_calls)

    def test_fully_connected_all_agents_see_all_others_in_round2(self) -> None:
        engine = ValueFlowEngine(
            topology_config={"type": "fully_connected"},
            interaction_config={"num_rounds": 2},
        )
        agents = [_make_entity(f"A{i}", response=f"A{i} output") for i in range(3)]
        engine.run_loop(game_masters=[_make_gm()], entities=agents, premise="")

        for i, agent in enumerate(agents):
            others_output = [f"A{j} output" for j in range(3) if j != i]
            obs_calls = [str(c.args[0]) for c in agent.observe.call_args_list]
            assert any(all(o in call for o in others_output) for call in obs_calls)

    def test_star_hub_observes_all_spokes(self) -> None:
        engine = ValueFlowEngine(
            topology_config={"type": "star"},
            interaction_config={"num_rounds": 2},
        )
        hub = _make_entity("Hub", response="Hub output")
        spoke1 = _make_entity("Spoke1", response="Spoke1 output")
        spoke2 = _make_entity("Spoke2", response="Spoke2 output")

        engine.run_loop(
            game_masters=[_make_gm()],
            entities=[hub, spoke1, spoke2],
            premise="",
        )

        hub_obs_calls = [str(c.args[0]) for c in hub.observe.call_args_list]
        assert any("Spoke1 output" in c for c in hub_obs_calls)
        assert any("Spoke2 output" in c for c in hub_obs_calls)


class TestLoggingAndCallbacks:
    def test_log_entries_appended_per_round(self) -> None:
        n_rounds = 3
        engine = _chain_engine(num_rounds=n_rounds)
        log: list[Any] = []
        engine.run_loop(
            game_masters=[_make_gm()],
            entities=[_make_entity("A0")],
            log=log,
            premise="",
        )
        assert len(log) == n_rounds

    def test_log_entry_has_step_and_summary(self) -> None:
        engine = _chain_engine(num_rounds=1)
        log: list[Any] = []
        engine.run_loop(
            game_masters=[_make_gm("narrator")],
            entities=[_make_entity("A0")],
            log=log,
            premise="",
        )
        entry = log[0]
        assert entry["Step"] == 1
        assert "narrator" in entry["Summary"]

    def test_checkpoint_callback_called_for_step0_and_all_rounds(self) -> None:
        """Total callbacks = 1 (step 0) + num_rounds."""
        n_rounds = 4
        engine = _chain_engine(num_rounds=n_rounds)
        calls: list[int] = []
        engine.run_loop(
            game_masters=[_make_gm()],
            entities=[_make_entity("A0")],
            checkpoint_callback=calls.append,
            premise="",
        )
        assert calls == [0, 1, 2, 3, 4]

    def test_topology_graph_populated_after_run(self) -> None:
        engine = _chain_engine(num_rounds=1)
        agents = [_make_entity("A0"), _make_entity("A1")]
        engine.run_loop(game_masters=[_make_gm()], entities=agents, premise="")
        assert engine.topology_graph == {"A0": [], "A1": ["A0"]}

    def test_topology_graph_empty_before_run(self) -> None:
        engine = _chain_engine()
        assert engine.topology_graph == {}
