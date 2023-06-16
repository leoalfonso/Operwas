import copy
import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from platypus import NSGAII, EpsMOEA, Problem

from src.faster import config
from src.faster.accumulate import (calc_networklength_accum_nodes,
                                   calc_pop_accum_nodes)
from src.faster.custom_platypus import (PumpingGraphType, PumpingGraphVariator,
                                        check_graph_correctness)
from src.faster.graph import feasibility_check
from src.faster.node_types import NodeType
from src.faster.operwa_library import Join_Calcul
from src.faster.optimization.base import OperwaFastBase, store_results_archive
from src.faster.pumping_info import PumpingInfo

SCRIPT_NAME = pathlib.Path(__file__).stem


class OperwaFastGeneticPumping(OperwaFastBase):
    def __init__(self):
        n_objectives = 2
        n_vars = 1
        super().__init__(n_vars, n_objectives, SCRIPT_NAME)

        self.types[0] = PumpingGraphType(
            copy.deepcopy(self._graph_base),
            self._feasible_pump_conns_list,
            w_nt_nothing=config.WEIGHT_NODE_TYPE_NOTHING,
            w_nt_wwtp=config.WEIGHT_NODE_TYPE_WWTP,
            w_nt_wwps=config.WEIGHT_NODE_TYPE_WWPS)

        self.directions[0] = Problem.MAXIMIZE
        self.directions[1] = Problem.MAXIMIZE

        self._idx_evaluation = -1

    def evaluate(self, solution):
        self._idx_evaluation += 1
        if self._idx_evaluation % 100 == 0:
            print(f"Evaluation #{self._idx_evaluation}")

        graph: nx.DiGraph = solution.variables[0]
        if not check_graph_correctness(graph):
            raise ValueError("Graph is bad.")

        data_dict = prepare_optimization_data(
            graph, self._data_dict_base, self._pumping_info)

        feasibility_check(data_dict)

        do_store_total_results = (not config.RUN_RESULTS_STORE_RESULTS_TOTAL_INTERVAL is None) and (
            self._idx_evaluation % config.RUN_RESULTS_STORE_RESULTS_TOTAL_INTERVAL == 0)

        do_store_nodes_results = (not config.RUN_RESULTS_STORE_RESULTS_NODES_INTERVAL is None) and (
            self._idx_evaluation % config.RUN_RESULTS_STORE_RESULTS_NODES_INTERVAL == 0)

        benf_cost_ratio, coverage, results_total, results_nodes = Join_Calcul(
            data_dict,
            self._interpolators,
            store_total_results=do_store_total_results,
            store_node_results=do_store_nodes_results)

        solution.objectives[:] = (benf_cost_ratio, coverage)
        solution._idx_evaluation = self._idx_evaluation

        # Write the best results to file
        connect_to = None
        if not config.RUN_RESULTS_STORE_RESULTS_SOLUTIONS_INTERVAL is None:
            if self._idx_evaluation % config.RUN_RESULTS_STORE_RESULTS_SOLUTIONS_INTERVAL == 0:
                connect_to = self._data_dict_base["connects_to"].copy()
                connect_to[data_dict["idx_node"]] = data_dict["connects_to"]

        if not results_nodes is None:
            results_nodes["idx_evaluation"] = np.full(
                (data_dict["node_type"].shape[0],), self._idx_evaluation)
        if not results_total is None:
            results_total["idx_evaluation"] = self._idx_evaluation

        self.store_results(connect_to, results_total, results_nodes)


def prepare_optimization_data(graph: nx.DiGraph, data_dict_base: dict[str, np.ndarray],
                              pumping_info: PumpingInfo) -> dict[str, np.ndarray]:
    """
    Note: mutates `graph`.
    """
    data_dict = {key: val.copy() for key, val in data_dict_base.items()}

    for idx_node, data_node in graph.nodes.items():
        node_type = data_node["node_type"]
        if node_type == NodeType.NOTHING:
            data_dict["node_type"][idx_node] = node_type
            data_dict["activated"][idx_node] = False
            data_dict["connects_to"][idx_node] = -2

            # graph.add_weighted_edges_from([(start, end, graph_base.get_edge_data(
            #     start, end)["weight"]) for start, end in graph_base.out_edges(idx_node)])
        elif node_type == NodeType.WWTP:
            data_dict["node_type"][idx_node] = node_type
            data_dict["activated"][idx_node] = True
            data_dict["connects_to"][idx_node] = -1
            # Do not add connections to graph if node is WWTP, since the flow stops
            # at a WWTP; downstream nodes do not accumulate anything from it.

        elif node_type == NodeType.WWPS:
            node_connects_to = list(graph.successors(idx_node))[0]
            data_dict["node_type"][idx_node] = node_type
            data_dict["activated"][idx_node] = True
            data_dict["connects_to"][idx_node] = node_connects_to
            data_dict["pumping_height"][idx_node] = pumping_info.get_pumping_height(
                idx_node, node_connects_to)
            data_dict["pumping_length_pump"][idx_node] = pumping_info.get_pumping_length_pump(
                idx_node, node_connects_to)
            data_dict["pumping_length_grav"][idx_node] = pumping_info.get_pumping_length_grav(
                idx_node, node_connects_to)

    data_dict = {key: val[data_dict["activated"]] for key, val in data_dict.items()}

    # Perform graph accumulations
    population_served_dict = calc_pop_accum_nodes(graph, data_dict["idx_node"])
    network_length_dict = calc_networklength_accum_nodes(graph, data_dict["idx_node"])

    # Put graph accumulations inside data dictionary
    for idx_data_dict, idx_node in enumerate(data_dict["idx_node"]):
        pop_served_total = 0.0
        for col_name, pop_served_town in population_served_dict[idx_node].items():
            data_dict[col_name][idx_data_dict] = pop_served_town
            pop_served_total += pop_served_town
        data_dict["population_served"][idx_data_dict] = pop_served_total
        data_dict["network_length"][idx_data_dict] = network_length_dict[idx_node]

    return data_dict


def run_operwa_fast(n_generations: int, population_size: int):
    n_runs = n_generations * population_size
    assert n_runs > 0, "Number of runs should be larger than zero."

    problem = OperwaFastGeneticPumping()
    variator = PumpingGraphVariator(copy.deepcopy(problem._graph_base), problem._feasible_pump_conns_list,
                                    mutation_rate=config.MUTATION_PROBABILITY,
                                    w_nt_nothing=config.WEIGHT_NODE_TYPE_NOTHING,
                                    w_nt_wwtp=config.WEIGHT_NODE_TYPE_WWTP,
                                    w_nt_wwps=config.WEIGHT_NODE_TYPE_WWPS,
                                    prob_crossover=config.CROSSOVER_PROBABILITY)
    algorithm = NSGAII(problem, population_size=population_size, variator=variator, archive=[])
    # algorithm = EpsMOEA(problem, 1e-2, population_size=population_size, variator=variator, archive=[])

    algorithm.run(n_runs)

    store_results_archive(algorithm)

    return

    # Plot data in graph
    x_coverage = [s.objectives[1] for s in algorithm.result]
    y_bencost = [s.objectives[0] for s in algorithm.result]

    alphas = np.linspace(3e-1, 3e-1, len(x_coverage))

    plt.scatter(x_coverage, y_bencost, alpha=alphas)

    x_lims = (0, max(x_coverage) * 1.1)
    y_lims = (0, max(y_bencost) * 1.1)

    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.xlabel("$Coverage$")
    plt.ylabel("$Benefit/Costs$")

    plt.grid()
    plt.show()


if __name__ == "__main__":
    for _ in range(1):
        run_operwa_fast(n_generations=config.OPT_NUM_GENERATIONS,
                        population_size=config.OPT_POPULATION_SIZE)
