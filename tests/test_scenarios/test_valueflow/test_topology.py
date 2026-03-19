"""Tests for ValueFlow topology graph construction (Phase 2)."""

from __future__ import annotations

import pytest

from scenarios.valueflow.game_masters import build_topology_graph

AGENTS = ["A0", "A1", "A2", "A3", "A4"]


class TestChainTopology:
    def test_first_agent_has_no_neighbors(self) -> None:
        g = build_topology_graph(AGENTS, "chain")
        assert g["A0"] == []

    def test_each_subsequent_agent_observes_predecessor(self) -> None:
        g = build_topology_graph(AGENTS, "chain")
        assert g["A1"] == ["A0"]
        assert g["A2"] == ["A1"]
        assert g["A3"] == ["A2"]
        assert g["A4"] == ["A3"]

    def test_all_agents_present_as_keys(self) -> None:
        g = build_topology_graph(AGENTS, "chain")
        assert set(g.keys()) == set(AGENTS)

    def test_single_agent_chain(self) -> None:
        g = build_topology_graph(["Solo"], "chain")
        assert g["Solo"] == []


class TestRingTopology:
    def test_each_agent_observes_predecessor_mod_n(self) -> None:
        g = build_topology_graph(AGENTS, "ring")
        assert g["A0"] == ["A4"]  # wraps around
        assert g["A1"] == ["A0"]
        assert g["A4"] == ["A3"]

    def test_all_agents_have_exactly_one_neighbor(self) -> None:
        g = build_topology_graph(AGENTS, "ring")
        for neighbors in g.values():
            assert len(neighbors) == 1

    def test_two_agent_ring(self) -> None:
        g = build_topology_graph(["X", "Y"], "ring")
        assert g["X"] == ["Y"]
        assert g["Y"] == ["X"]


class TestStarTopology:
    def test_hub_is_first_agent(self) -> None:
        g = build_topology_graph(AGENTS, "star")
        hub = AGENTS[0]
        assert set(g[hub]) == set(AGENTS[1:])

    def test_spokes_observe_only_hub(self) -> None:
        g = build_topology_graph(AGENTS, "star")
        hub = AGENTS[0]
        for agent in AGENTS[1:]:
            assert g[agent] == [hub]

    def test_two_agent_star(self) -> None:
        g = build_topology_graph(["Hub", "Spoke"], "star")
        assert g["Hub"] == ["Spoke"]
        assert g["Spoke"] == ["Hub"]


class TestFullyConnectedTopology:
    def test_each_agent_observes_all_others(self) -> None:
        g = build_topology_graph(AGENTS, "fully_connected")
        for agent in AGENTS:
            others = [a for a in AGENTS if a != agent]
            assert set(g[agent]) == set(others)

    def test_no_self_loops(self) -> None:
        g = build_topology_graph(AGENTS, "fully_connected")
        for agent, neighbors in g.items():
            assert agent not in neighbors

    def test_neighbor_count(self) -> None:
        g = build_topology_graph(AGENTS, "fully_connected")
        for neighbors in g.values():
            assert len(neighbors) == len(AGENTS) - 1


class TestCustomTopology:
    def test_explicit_adjacency_respected(self) -> None:
        adj = {"A0": ["A1", "A2"], "A1": [], "A2": ["A0"]}
        g = build_topology_graph(["A0", "A1", "A2"], "custom", custom_adjacency=adj)
        assert set(g["A0"]) == {"A1", "A2"}
        assert g["A1"] == []
        assert g["A2"] == ["A0"]

    def test_missing_adjacency_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_adjacency"):
            build_topology_graph(AGENTS, "custom", custom_adjacency=None)

    def test_unknown_agent_in_adjacency_raises(self) -> None:
        bad_adj = {"A0": ["UNKNOWN"]}
        with pytest.raises(ValueError, match="Unknown agent"):
            build_topology_graph(["A0", "A1"], "custom", custom_adjacency=bad_adj)

    def test_unknown_key_in_adjacency_raises(self) -> None:
        bad_adj = {"GHOST": ["A0"]}
        with pytest.raises(ValueError, match="Unknown agent"):
            build_topology_graph(["A0", "A1"], "custom", custom_adjacency=bad_adj)


class TestUnknownTopology:
    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown topology"):
            build_topology_graph(AGENTS, "hexagonal")
