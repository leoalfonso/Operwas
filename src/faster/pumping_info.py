from typing import Generator, Iterable, Optional

import networkx as nx
import numpy as np

from src.faster import config


class PumpingInfo:
    def __init__(self, pumping_height, pumping_lengths_pump, pumping_lengths_grav):
        self._pumping_height = pumping_height
        self._pumping_lengths_pump = pumping_lengths_pump
        self._pumping_lengths_grav = pumping_lengths_grav

    def get_pumping_height(self, i_start, i_end):
        return self._pumping_height[i_start, i_end]

    def get_pumping_length_pump(self, i_start, i_end):
        return self._pumping_lengths_pump[i_start, i_end]

    def get_pumping_length_grav(self, i_start, i_end):
        return self._pumping_lengths_grav[i_start, i_end]


def get_pumping_info() -> PumpingInfo:
    pumping_height = np.load(config.EXP_PUMPING_HEIGHT_FILE_PATH)
    pumping_lengths_pump = np.load(config.EXP_PUMPING_LENGTH_PUMP_FILE_PATH)
    pumping_lengths_grav = np.load(config.EXP_PUMPING_LENGTH_GRAV_FILE_PATH)
    return PumpingInfo(pumping_height, pumping_lengths_pump, pumping_lengths_grav)


def check_if_feasible_pumping_connection(idx_node_start: int, idx_node_end: int, graph: nx.DiGraph, pumping_info: PumpingInfo) -> bool:
    if idx_node_start == idx_node_end:
        # Can't pump to yourself
        return False
    simple_paths = nx.all_simple_paths(graph, idx_node_start, idx_node_end)
    if next(simple_paths, False):
        # If there is a "downhill" path through the rivers from start to end
        # it makes no sense to pump.
        return False
    pumping_height = pumping_info.get_pumping_height(idx_node_start, idx_node_end)
    if not (config.PUMPING_HEIGHT_MIN <= pumping_height <= config.PUMPING_HEIGHT_MAX):
        # Pumping height is outside min-max bounds.
        return False
    pumping_length_pump = pumping_info.get_pumping_length_pump(idx_node_start, idx_node_end)
    if not (config.PUMPING_LENGTH_PUMP_MIN <= pumping_length_pump <= config.PUMPING_LENGTH_PUMP_MAX):
        # Pumping path length (pumping) is outside min-max bounds.
        return False
    pumping_length_grav = pumping_info.get_pumping_length_grav(idx_node_start, idx_node_end)
    if not (config.PUMPING_LENGTH_GRAV_MIN <= pumping_length_grav <= config.PUMPING_LENGTH_GRAV_MAX):
        # Pumping path length (gravitational) is outside min-max bounds.
        return False
    pumping_length_total = pumping_length_pump + pumping_length_grav
    if not (config.PUMPING_LENGTH_TOTAL_MIN <= pumping_length_total <= config.PUMPING_LENGTH_TOTAL_MAX):
        # Pumping path length (gravitational) is outside min-max bounds.
        return False
    return True


def get_feasible_pumping_connections(graph: nx.DiGraph, pumping_info: PumpingInfo) -> list[tuple[int, int]]:
    feasible_connections = []
    for idx_start in graph.nodes():
        for idx_end in graph.nodes():
            if check_if_feasible_pumping_connection(idx_start, idx_end, graph, pumping_info):
                feasible_connections.append((idx_start, idx_end))
    return feasible_connections


def check_if_simple_path_exists(idx_node_start: int, idx_node_end: int, graph: nx.DiGraph) -> bool:
    simple_paths = nx.all_simple_paths(graph, idx_node_start, idx_node_end)
    if next(simple_paths, False):
        return True
    return False


def check_if_simple_path_exists_along_path(path: list[int], graph: nx.DiGraph) -> bool:
    for i, idx_node_start in enumerate(path):
        for idx_node_end in path[i:]:
            if check_if_simple_path_exists(idx_node_start, idx_node_end, graph):
                return True
    return False


def get_feasible_pumping_graph(graph: nx.DiGraph, pumping_info: PumpingInfo) -> nx.DiGraph:
    feasible_connections = get_feasible_pumping_connections(graph, pumping_info)
    graph_feasible = nx.DiGraph()
    graph_feasible.add_edges_from(feasible_connections)
    return graph_feasible


def all_feasible_simple_pumping_paths(graph: nx.DiGraph,
                                      pumping_info: PumpingInfo,
                                      idx_nodes_start: Optional[Iterable[int]] = None,
                                      idx_nodes_end: Optional[Iterable[int]] = None) -> Generator[list[int], None, None]:
    graph_feasible_pumping = get_feasible_pumping_graph(graph, pumping_info)
    if idx_nodes_start is None:
        idx_nodes_start = graph_feasible_pumping.nodes()
    if idx_nodes_end is None:
        idx_nodes_end = graph_feasible_pumping.nodes()
    for idx_node_start in idx_nodes_start:
        for idx_node_end in idx_nodes_end:
            if idx_node_start == idx_node_end:
                # Can't pump to yourself
                continue
            for simple_path in nx.all_simple_paths(graph_feasible_pumping, idx_node_start,
                                                   idx_node_end, cutoff=config.PUMPING_SIMPLE_PATH_LENGTH_MAX):
                # Final check to see if we are not pumping to end up somewhere that is actually
                # just along the river path downstream. If the pumping path is direct (length of 2) then
                # we don't need to check, since we can assume that check is already done during the
                # creation of the feasible pumping paths.
                if len(simple_path) <= 2 or (not check_if_simple_path_exists_along_path(simple_path, graph)):
                    yield simple_path


if __name__ == "__main__":
    from src.faster.graph import create_graph
    import math
    graph = create_graph()
    pumping_info = get_pumping_info()
    feasible_conns = list(get_feasible_pumping_connections(graph, pumping_info))
    node_paths_map = {node: [conn for conn in feasible_conns if conn[0] == node]
                      for node in graph.nodes()}
    conns_per_node = [len(node_paths_map[node]) for node in graph.nodes()]
    average_conns_per_node = sum(conns_per_node)/len(conns_per_node)
    total_number_combinations = math.prod([x + 2 for x in conns_per_node])
    equivalent_choices_per_node = total_number_combinations ** (1/graph.number_of_nodes())
    print(f"Connections per node: {conns_per_node}")
    print(f"Number of feasible connections found: {len(feasible_conns)}")
    print(f"Min number of connections per node: {min(conns_per_node)}")
    print(f"Max number of connections per node: {max(conns_per_node)}")
    print(f"Average number of connections per node: {average_conns_per_node:.2f}")
    print(f"Num options total: {total_number_combinations}")
    print(f"Equivalent number of choices per node: {equivalent_choices_per_node:.2f}")
