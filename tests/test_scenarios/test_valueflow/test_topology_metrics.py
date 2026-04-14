from __future__ import annotations

import math

import pytest

from scenarios.valueflow.topology_metrics import compute_topology_metrics, normalize_topology_name


def test_fully_connected_metrics_match_closed_form() -> None:
    metrics = compute_topology_metrics("fully_connected", 6)

    assert metrics["topology"] == "fully_connected"
    assert metrics["num_nodes"] == 6
    assert metrics["num_edges"] == 15
    assert math.isclose(metrics["density"], 1.0)
    assert math.isclose(metrics["avg_shortest_path"], 1.0)
    assert math.isclose(metrics["fiedler_value"], 6.0)


def test_community_metrics_have_expected_edge_count() -> None:
    metrics = compute_topology_metrics("community", 15)

    assert metrics["num_nodes"] == 15
    assert metrics["num_edges"] == 33
    assert math.isclose(metrics["density"], 66 / 210)
    assert metrics["avg_shortest_path"] > 1.0
    assert metrics["fiedler_value"] > 0.0


def test_exact_topology_name_is_required() -> None:
    with pytest.raises(ValueError, match="Unsupported topology"):
        normalize_topology_name("inner core periphery undirected")


@pytest.mark.parametrize(
        ("topology_name", "num_nodes"),
    [
        ("community", 12),
        ("small_world", 10),
        ("core_periphery_bidirectional", 8),
    ],
)
def test_fixed_15_node_topologies_validate_node_count(
    topology_name: str,
    num_nodes: int,
) -> None:
    with pytest.raises(ValueError, match="only supports 15 nodes"):
        compute_topology_metrics(topology_name, num_nodes)
