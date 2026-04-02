"""Tests for ValueFlow topology graph construction (Phase 2)."""

from __future__ import annotations

import pytest

from scenarios.valueflow.game_masters import build_topology_graph

AGENTS = ["A0", "A1", "A2", "A3", "A4"]
COMMUNITY_AGENTS = [f"A{i}" for i in range(15)]


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


class TestUndirectedCycleTopology:
    def test_each_agent_observes_immediate_neighbors(self) -> None:
        g = build_topology_graph(AGENTS, "undirected_cycle")
        assert g["A0"] == ["A4", "A1"]
        assert g["A1"] == ["A0", "A2"]
        assert g["A4"] == ["A3", "A0"]

    def test_all_agents_have_two_neighbors_in_longer_cycle(self) -> None:
        g = build_topology_graph(AGENTS, "undirected_cycle")
        for neighbors in g.values():
            assert len(neighbors) == 2

    def test_two_agent_cycle_degrades_to_mutual_observation(self) -> None:
        g = build_topology_graph(["X", "Y"], "undirected_cycle")
        assert g["X"] == ["Y"]
        assert g["Y"] == ["X"]

    def test_single_agent_cycle_has_no_neighbors(self) -> None:
        g = build_topology_graph(["Solo"], "undirected_cycle")
        assert g["Solo"] == []


class TestSmallWorldTopology:
    def test_preserves_local_cycle_neighbors(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "small_world")
        assert {"A14", "A1"}.issubset(set(g["A0"]))
        assert {"A0", "A2"}.issubset(set(g["A1"]))
        assert {"A13", "A0"}.issubset(set(g["A14"]))

    def test_adds_expected_shortcuts(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "small_world")
        assert "A7" in g["A0"]
        assert "A0" in g["A7"]
        assert "A10" in g["A3"]
        assert "A3" in g["A10"]
        assert "A12" in g["A6"]
        assert "A6" in g["A12"]

    def test_shortcut_endpoints_have_three_neighbors(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "small_world")
        for agent in ("A0", "A3", "A6", "A7", "A10", "A12"):
            assert len(g[agent]) == 3

    def test_non_shortcut_agents_keep_two_neighbors(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "small_world")
        for agent in ("A1", "A2", "A4", "A5", "A8", "A9", "A11", "A13", "A14"):
            assert len(g[agent]) == 2


class TestCommunityTopology:
    def test_agents_observe_everyone_in_their_own_cluster(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "community")
        assert set(g["A1"]) == {"A0", "A2", "A3", "A4"}
        assert set(g["A6"]) == {"A5", "A7", "A8", "A9"}
        assert set(g["A11"]) == {"A10", "A12", "A13", "A14"}

    def test_bridge_agents_observe_the_other_two_bridge_agents(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "community")
        assert set(g["A0"]) == {"A1", "A2", "A3", "A4", "A5", "A10"}
        assert set(g["A5"]) == {"A6", "A7", "A8", "A9", "A0", "A10"}
        assert set(g["A10"]) == {"A11", "A12", "A13", "A14", "A0", "A5"}

    def test_non_bridge_agents_have_no_cross_cluster_neighbors(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "community")
        assert all(neighbor in {"A0", "A2", "A3", "A4"} for neighbor in g["A1"])
        assert all(neighbor in {"A5", "A7", "A8", "A9"} for neighbor in g["A6"])
        assert all(neighbor in {"A10", "A12", "A13", "A14"} for neighbor in g["A11"])

    def test_requires_exactly_fifteen_agents(self) -> None:
        with pytest.raises(ValueError, match="exactly 15 agents"):
            build_topology_graph(AGENTS, "community")


class TestCorePeripheryTopology:
    def test_core_is_fully_connected(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "core_periphery")
        assert set(g["A0"]) == {"A1", "A2", "A3", "A4"}
        assert set(g["A4"]) == {"A0", "A1", "A2", "A3"}

    def test_periphery_observes_assigned_core_agents(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "core_periphery")
        assert g["A5"] == ["A0", "A1"]
        assert g["A9"] == ["A4", "A0"]
        assert g["A14"] == ["A4", "A1"]

    def test_core_does_not_observe_periphery(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "core_periphery")
        assert all(neighbor in {"A1", "A2", "A3", "A4"} for neighbor in g["A0"])

    def test_requires_exactly_fifteen_agents(self) -> None:
        with pytest.raises(ValueError, match="exactly 15 agents"):
            build_topology_graph(AGENTS, "core_periphery")


class TestCorePeripheryBidirectionalTopology:
    def test_core_is_fully_connected(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "core_periphery_bidirectional")
        assert {"A1", "A2", "A3", "A4"}.issubset(set(g["A0"]))
        assert {"A0", "A1", "A2", "A3"}.issubset(set(g["A4"]))

    def test_periphery_observes_assigned_core_agents(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "core_periphery_bidirectional")
        assert g["A5"] == ["A0", "A1"]
        assert g["A9"] == ["A4", "A0"]
        assert g["A14"] == ["A4", "A1"]

    def test_core_observes_attached_periphery_agents(self) -> None:
        g = build_topology_graph(COMMUNITY_AGENTS, "core_periphery_bidirectional")
        assert {"A5", "A9", "A10", "A13"}.issubset(set(g["A0"]))
        assert {"A8", "A9", "A12", "A14"}.issubset(set(g["A4"]))

    def test_requires_exactly_fifteen_agents(self) -> None:
        with pytest.raises(ValueError, match="exactly 15 agents"):
            build_topology_graph(AGENTS, "core_periphery_bidirectional")


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
