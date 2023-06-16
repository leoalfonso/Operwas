import copy
import random

import networkx as nx
import platypus
from platypus.core import Mutation, Variator
from platypus.operators import SSX, GAOperator, Replace

from src.faster.node_types import NodeType

DEFAULT_VARIATOR = GAOperator(SSX(), Replace())

NODE_TYPES_ALL = [NodeType.NOTHING, NodeType.WWTP, NodeType.WWPS]
NODE_TYPES_NO_WWPS = [NodeType.NOTHING, NodeType.WWTP]


class PumpingGraphType(platypus.types.Type):
    def __init__(self, graph_base: nx.DiGraph, feasible_pump_conns_list: list[list[int]],
                 w_nt_nothing: float = 1.0, w_nt_wwtp: float = 1.0, w_nt_wwps: float = 1.0):
        self._graph_base = copy.deepcopy(graph_base)
        self._feasible_connections = copy.deepcopy(feasible_pump_conns_list)

        self._weights_all = [w_nt_nothing, w_nt_wwtp, w_nt_wwps]
        self._weights_no_wwps = [w_nt_nothing, w_nt_wwtp]

    def rand(self) -> nx.DiGraph:
        mutation_rate = 1.0
        graph = copy.deepcopy(self._graph_base)
        mutate_graph(graph, self._graph_base, self._feasible_connections,
                     mutation_rate=mutation_rate, weights=self._weights_all, weights_no_wwps=self._weights_no_wwps)
        return graph


class PumpingGraphVariator(GAOperator):
    def __init__(self, graph_base: nx.DiGraph, feasible_conns_list: list[list[int]], mutation_rate: float = 0.1,
                 w_nt_nothing: float = 1.0, w_nt_wwtp: float = 1.0, w_nt_wwps: float = 1.0, prob_crossover: float = 0.3):
        variator = GraphSSX(probability=prob_crossover)
        mutator = GraphMutator(graph_base, feasible_conns_list, mutation_rate=mutation_rate,
                               w_nt_nothing=w_nt_nothing, w_nt_wwtp=w_nt_wwtp, w_nt_wwps=w_nt_wwps)
        super().__init__(variator, mutator)


class GraphMutator(Mutation):
    def __init__(self, graph_base: nx.DiGraph, feasible_pump_conns_list: list[list[int]], mutation_rate: float = 1e-1,
                 w_nt_nothing: float = 1.0, w_nt_wwtp: float = 1.0, w_nt_wwps: float = 1.0):
        super().__init__()

        self._graph_base = copy.deepcopy(graph_base)
        self._feasible_connections = copy.deepcopy(feasible_pump_conns_list)

        self._mutation_rate = mutation_rate

        self._weights_all = [w_nt_nothing, w_nt_wwtp, w_nt_wwps]
        self._weights_no_wwps = [w_nt_nothing, w_nt_wwtp]

    def mutate(self, parent):
        result = copy.deepcopy(parent)
        graph_mutate = result.variables[0]
        mutate_graph(graph_mutate, self._graph_base, self._feasible_connections,
                     mutation_rate=self._mutation_rate, weights=self._weights_all, weights_no_wwps=self._weights_no_wwps)
        result.evaluated = False
        return result


class GraphSSX(Variator):

    def __init__(self, probability=1.0):
        n_parents = 2
        super().__init__(n_parents)
        self.probability = probability

    def evolve(self, parents):
        result1 = copy.deepcopy(parents[0])
        result2 = copy.deepcopy(parents[1])

        graph1 = result1.variables[0]
        graph2 = result2.variables[0]

        graph1, graph2 = ssx_graphs(graph1, graph2, self.probability)
        result1.variables[0] = graph1
        result2.variables[0] = graph2

        result1.evaluated = False
        result2.evaluated = False

        return [result1, result2]


def mutate_graph(graph_mutate: nx.DiGraph, graph_base: nx.DiGraph, feasible_pump_conns_list: list[list[int]],
                 mutation_rate: float = 1e-1, weights: list[float] = [1., 1., 1.], weights_no_wwps: list[float] = [1., 1.]):
    # TODO: switch to unweighted graphs for speed and simplicity

    mutated_nodes = {node: random.choices(NODE_TYPES_ALL, weights=weights, k=1)[0]
                     for node in graph_mutate.nodes() if random.random() < mutation_rate}

    mutated_nodes_list = [(node, new_node_type) for node, new_node_type in mutated_nodes.items()]
    random.shuffle(mutated_nodes_list)

    # Check if new pumping stations even have somewhere to pump, if not, set them to either nothing or WWTP
    for node, new_node_type in mutated_nodes_list:
        if new_node_type == NodeType.WWPS and len(feasible_pump_conns_list[node]) == 0:
            mutated_nodes[node] = random.choices(
                NODE_TYPES_NO_WWPS, weights=weights_no_wwps, k=1)[0]

    # First set WWTPs because it is always safe to do so
    for node, new_node_type in mutated_nodes_list:
        if new_node_type == NodeType.WWTP:
            graph_mutate.nodes[node]["node_type"] = new_node_type

    # Now handle new NOTHINGS and WWPSes
    new_wwps_nodes = []
    for node, new_node_type in mutated_nodes_list:
        old_node_type = graph_mutate.nodes[node]["node_type"]
        out_edges_mutate: list[tuple[int, int, float]] = [
            (start, end, 1.0) for start, end in graph_mutate.out_edges(node)]
        out_edges_base: list[tuple[int, int, float]] = [
            (start, end, 1.0) for start, end in graph_base.out_edges(node)]

        if new_node_type == NodeType.NOTHING:
            graph_mutate.nodes[node]["node_type"] = new_node_type
            graph_mutate.remove_edges_from(out_edges_mutate)
            graph_mutate.add_weighted_edges_from(out_edges_base)
            if not check_graph_correctness(graph_mutate):
                # TODO: better cycle breaking than just giving up?
                # Undo operation and keep node the same
                graph_mutate.remove_edges_from(out_edges_base)
                graph_mutate.add_weighted_edges_from(out_edges_mutate)
                graph_mutate.nodes[node]["node_type"] = old_node_type

        elif new_node_type == NodeType.WWPS:
            graph_mutate.nodes[node]["node_type"] = new_node_type
            graph_mutate.remove_edges_from(out_edges_mutate)
            new_wwps_nodes.append(node)
            feasible_pump_conns = feasible_pump_conns_list[node]
            random.shuffle(feasible_pump_conns)
            success = False
            for node_pump_to in feasible_pump_conns:
                # We can only pump towards a node that is also a pumping station, or a WWTP
                node_pump_to_type = graph_mutate.nodes[node_pump_to]["node_type"]
                if not (node_pump_to_type == NodeType.WWTP or node_pump_to_type == NodeType.WWPS):
                    continue
                graph_mutate.add_weighted_edges_from([(node, node_pump_to, 1.0)])
                if check_graph_correctness(graph_mutate):
                    success = True
                    break
                else:
                    graph_mutate.remove_edge(node, node_pump_to)
            if not success:
                # Revert back to original state
                graph_mutate.nodes[node]["node_type"] = old_node_type
                graph_mutate.add_weighted_edges_from(out_edges_mutate)

        elif new_node_type == NodeType.WWTP:
            continue
        else:
            raise ValueError(f"Unknown node type: {new_node_type} for node: {node}.")

    return graph_mutate


def check_graph_correctness(graph: nx.DiGraph) -> bool:
    # Check for cycles in the graph
    if not nx.is_directed_acyclic_graph(graph):
        return False

    for node, node_data in graph.nodes.data():
        if node_data["node_type"] == NodeType.WWPS:
            # Check if the WWPS is pumping at all
            if len(graph.out_edges(node)) == 0:
                return False
            # Check if there is pumping to a "nothing" node
            for out_edge in graph.out_edges(node):
                node_receiving = out_edge[1]
                node_receiving_type = graph.nodes[node_receiving]["node_type"]
                if node_receiving_type == NodeType.NOTHING:
                    return False

    return True


def ssx_graphs(graph1: nx.DiGraph, graph2: nx.DiGraph, prob: float = 0.2) -> nx.DiGraph:
    # if not (check_graph_correctness(graph1) and check_graph_correctness(graph2)):
    #     raise ValueError("graph2 is bad.")

    # Assume both graphs have the same nodes
    for node in graph1.nodes():
        if random.random() > prob:
            continue

        node_type1, node_type2 = graph1.nodes[node]["node_type"], graph2.nodes[node]["node_type"]
        out_edges1, out_edges2 = list(graph1.out_edges(node)), list(graph2.out_edges(node))

        # if not (check_graph_correctness(graph1) and check_graph_correctness(graph2)):
        #     raise ValueError('ohoho')

        # Switch values
        graph1.remove_edges_from(out_edges1)
        graph1.add_weighted_edges_from([(i, j, 1.0) for i, j in out_edges2])
        graph1.nodes[node]["node_type"] = node_type2
        graph2.remove_edges_from(out_edges2)
        graph2.add_weighted_edges_from([(i, j, 1.0) for i, j in out_edges1])
        graph2.nodes[node]["node_type"] = node_type1

        if not (check_graph_correctness(graph1) and check_graph_correctness(graph2)):
            # Cycle detected, undo everything and continue
            # TODO: better cycle handling (instead of just giving up)
            graph1.remove_edges_from(out_edges2)
            graph1.add_weighted_edges_from([(i, j, 1.0) for i, j in out_edges1])
            graph1.nodes[node]["node_type"] = node_type1
            graph2.remove_edges_from(out_edges1)
            graph2.add_weighted_edges_from([(i, j, 1.0) for i, j in out_edges2])
            graph2.nodes[node]["node_type"] = node_type2
            # if not (check_graph_correctness(graph1) and check_graph_correctness(graph2)):
            #     raise ValueError('ohoho')

    # if not check_graph_correctness(graph1):
    #     raise ValueError("graph1 is bad.")
    # if not check_graph_correctness(graph2):
    #     raise ValueError("graph2 is bad.")

    return graph1, graph2


if __name__ == "__main__":
    # graph = random_directed_acyclic_graph(5)
    # print(f"Is graph acyclic directed? {nx.is_directed_acyclic_graph(graph)}")
    import copy

    import matplotlib.pyplot as plt

    # plt.figure()
    # nx.draw(graph, with_labels=True)
    # plt.show()
    from src.faster.graph import create_graph
    from src.faster.pumping_info import (get_feasible_pumping_connections,
                                         get_pumping_info)

    graph = create_graph()
    graph_base = copy.deepcopy(graph)
    pumping_info = get_pumping_info()

    conns_pump_feas = get_feasible_pumping_connections(graph, pumping_info)
    conns_pump_feas_aap = [[j for i2, j in conns_pump_feas if i == i2] for i in graph.nodes()]

    graphtype = PumpingGraphType(graph_base, conns_pump_feas_aap)

    variator = PumpingGraphVariator(graph_base, conns_pump_feas_aap)

    class Solution:
        def __init__(self, variables):
            self.variables = variables

    sol1 = Solution([graphtype.rand()])
    sol2 = Solution([graphtype.rand()])

    sol3, sol4 = variator.evolve([sol1, sol2])

    import time
    from datetime import timedelta

    from src.faster.plotting import plot_with_graph

    n_samples = int(1e4)
    t_start = time.time()

    for _ in range(n_samples):
        graph_mutated = mutate_graph(graph, graph_base, conns_pump_feas_aap,
                                     mutation_rate=1e-1, weights=[1, 1, 1], weights_no_wwps=[1, 1])
        # print(graph_mutated.edges())
        # plot_with_graph(graph_mutated)

    t_delta = time.time() - t_start
    t_delta_pretty = timedelta(seconds=t_delta)
    iter_per_sec = n_samples / t_delta
    print(f"Completed {n_samples} iterations, which took {t_delta_pretty}. This amounts to an average of {iter_per_sec:.2f} per second.")
