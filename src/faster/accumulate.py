from typing import Iterable, Optional

import networkx as nx

from src.faster.node_types import NodeType


def calc_pop_accum_nodes(graph: nx.DiGraph, idx_nodes: Optional[Iterable[int]] = None) -> dict[int, dict[str, float]]:
    if idx_nodes is None:
        idx_nodes = graph.nodes()
    # Note: this assumes all nodes have the same data fields.
    n_towns = len([key for key in graph.nodes[0].keys()
                  if key.startswith("population_served__town_")])
    caches: list[dict[int, float]] = [dict() for _ in range(n_towns)]
    return {idx_node: {f"population_served__town_{i_town}": calc_pop_accum_cached(graph, idx_node, i_town, cache_town) for i_town, cache_town in enumerate(caches)}
            for idx_node in idx_nodes}


def calc_pop_accum(graph: nx.DiGraph, i_start: int, i_town: int) -> tuple[float, dict[int, float]]:
    cache: dict[int, float] = dict()
    pop_accum = calc_pop_accum_cached(graph, i_start, i_town, cache)
    return pop_accum, cache


def calc_pop_accum_cached(graph: nx.DiGraph, i_start: int, i_town: int, cache: dict[int, float]) -> float:
    if i_start in cache:
        return cache[i_start]
    pop_accum: float = graph.nodes[i_start][f"population_served__town_{i_town}"]
    for i_pred in graph.predecessors(i_start):
        if graph.nodes[i_pred]["node_type"] == NodeType.WWTP:
            # Do not need to serve population if it is already served by the predecessor
            continue
        else:
            # "wwps" or None
            w_edge = graph.edges[i_pred, i_start]["weight"]
            pop_accum += w_edge * calc_pop_accum_cached(graph, i_pred, i_town, cache)
    cache[i_start] = pop_accum
    return pop_accum


def calc_networklength_accum_nodes(graph: nx.DiGraph, idx_nodes: Optional[Iterable[int]] = None) -> dict[int, float]:
    if idx_nodes is None:
        idx_nodes = graph.nodes()
    cache: dict[int, float] = dict()
    return {idx_node: calc_networklength_accum_cached(graph, idx_node, cache) for idx_node in graph.nodes()}


def calc_networklength_accum(graph: nx.DiGraph, i_start: int) -> tuple[float, dict[int, float]]:
    cache: dict[int, float] = dict()
    pop_accum = calc_networklength_accum_cached(graph, i_start, cache)
    return pop_accum, cache


def calc_networklength_accum_cached(graph: nx.DiGraph, i_start: int, cache: dict[int, float]) -> float:
    if i_start in cache:
        return cache[i_start]
    pop_accum: float = graph.nodes[i_start]["network_length"]
    for i_pred in graph.predecessors(i_start):
        node_type = graph.nodes[i_pred]["node_type"]
        if node_type == NodeType.WWTP or node_type == NodeType.WWPS:
            # Network does not keep running once it has reached a WWTP
            # Network only runs up to the WWPS, it does not run to the child of the WWPS
            continue
        else:
            # Node is nothing
            w_edge = graph.edges[i_pred, i_start]["weight"]
            pop_accum += w_edge * \
                calc_networklength_accum_cached(graph, i_pred, cache)
    cache[i_start] = pop_accum
    return pop_accum
