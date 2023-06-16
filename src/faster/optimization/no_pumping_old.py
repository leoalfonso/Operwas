import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from platypus import NSGAII, EpsMOEA, Problem, Solution, Subset

from src.faster import config
from src.faster.accumulate import (calc_networklength_accum_nodes,
                                   calc_pop_accum_nodes)
from src.faster.custom_typing import Scalar
from src.faster.graph import create_graph, feasibility_check
from src.faster.node_types import NodeType
from src.faster.operwa_library import Join_Calcul
from src.faster.optimization.base import OperwaFastBase, store_results_archive

SCRIPT_NAME = pathlib.Path(__file__).stem


class OperwaFast(OperwaFastBase):
    def __init__(self):
        n_objectives = 2
        n_vars = create_graph().number_of_nodes()
        super().__init__(n_vars, n_objectives, SCRIPT_NAME)

        for i in range(len(self.types)):
            self.types[i] = Subset([-2, -1], 1)

        self.directions[0] = Problem.MAXIMIZE
        self.directions[1] = Problem.MAXIMIZE

        self._idx_evaluation = -1

    def evaluate(self, solution: Solution):
        self._idx_evaluation += 1
        if self._idx_evaluation % 100 == 0:
            print(f"Evaluation #{self._idx_evaluation}")

        node_types = [vs[0] for vs in solution.variables]
        node_types = [NodeType.NOTHING if v == -2 else NodeType.WWTP for v in node_types]

        data_dict = prepare_optimization_data(node_types,  self._graph, self._data_dict_base)

        feasibility_check(data_dict)

        do_store_total_results = (not config.RUN_RESULTS_STORE_RESULTS_TOTAL_INTERVAL is None) and (
            self._idx_evaluation % config.RUN_RESULTS_STORE_RESULTS_TOTAL_INTERVAL == 0)

        do_store_nodes_results = (not config.RUN_RESULTS_STORE_RESULTS_NODES_INTERVAL is None) and (
            self._idx_evaluation % config.RUN_RESULTS_STORE_RESULTS_NODES_INTERVAL == 0)

        # Calculate benefits, costs, coverage
        benf_cost_ratio, coverage, results_total, results_nodes = Join_Calcul(
            data_dict,
            self._interpolators,
            store_total_results=do_store_total_results,
            store_node_results=do_store_nodes_results,
        )

        solution.objectives[:] = (benf_cost_ratio, coverage)
        solution._idx_evaluation = self._idx_evaluation

        solution_store = None
        if not config.RUN_RESULTS_STORE_RESULTS_SOLUTIONS_INTERVAL is None:
            if self._idx_evaluation % config.RUN_RESULTS_STORE_RESULTS_SOLUTIONS_INTERVAL == 0:
                solution_store = np.array(node_types)

        if not results_nodes is None:
            results_nodes["idx_evaluation"] = np.full(
                (data_dict["node_type"].shape[0],), self._idx_evaluation)
        if not results_total is None:
            results_total["idx_evaluation"] = self._idx_evaluation

        self.store_results(solution_store, results_total, results_nodes)


def prepare_optimization_data(node_types: list[NodeType], graph: nx.DiGraph, data_dict_base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Note: mutates `graph`.
    """
    data_dict = {key: val.copy() for key, val in data_dict_base.items()}

    for idx_node, node_type in zip(graph.nodes(), node_types):
        node_data = graph.nodes[idx_node]
        node_data["node_type"] = node_type
        data_dict["node_type"][idx_node] = node_type
        if node_type == NodeType.NOTHING:
            data_dict["activated"][idx_node] = False
            data_dict["connects_to"][idx_node] = -2

        elif node_type == NodeType.WWTP:
            data_dict["activated"][idx_node] = True
            data_dict["connects_to"][idx_node] = -1

        else:
            raise ValueError(f"Unknown node type: {node_type}, for node: {idx_node}")

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

    problem = OperwaFast()
    algorithm = NSGAII(problem, population_size=population_size, archive=[])
    # algorithm = EpsMOEA(problem, 1e-2, population_size=population_size, archive=[])

    algorithm.run(n_runs)

    store_results_archive(algorithm)

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
