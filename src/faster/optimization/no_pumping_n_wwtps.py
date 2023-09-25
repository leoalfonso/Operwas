# This does exactly the same as operwas, but using the database produced with __init__.py in scr/faster/experiments
# so it is faster.

import pathlib
import ast
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from platypus import NSGAII, EpsMOEA, Problem, Solution, Subset
from platypus import Integer

from src.faster import config
from src.faster.accumulate import (calc_networklength_accum_nodes,
                                   calc_pop_accum_nodes)
from src.faster.custom_typing import Scalar
from src.faster.graph import create_graph, feasibility_check
from src.faster.node_types import NodeType
from src.faster.operwa_library import Join_Calcul
from src.faster.optimization.base import OperwaFastBase, store_results_archive

SCRIPT_NAME = pathlib.Path(__file__).stem

THERE_IS_INITIAL_SEED = False

# The optimisation problem in no_pumping.py is set up as one with 33 decision variables that are binary.
# This script is to allow the user to select n WWTPs so that the problem has n decision variables that correspond to
# n indexes of the 33 candidates. The optimization problem has, therefore, n integer decision variables

class OperwaFastN(OperwaFastBase):
    def __init__(self):
        n_objectives = 2
        n_vars = 5  # Set the number of decision variables (number of WWTPs to be located)

        super().__init__(n_vars, n_objectives, SCRIPT_NAME)

        for i in range(len(self.types)):
            # Use Integer type for the n_vars decision variables
            self.types[i] = Integer(0, 32)  # Assuming indexes can range from 0 to 32

        self.directions[0] = Problem.MAXIMIZE
        self.directions[1] = Problem.MAXIMIZE

        self._idx_evaluation = -1

    def evaluate(self, solution: Solution):
        self._idx_evaluation += 1
        if self._idx_evaluation % 100 == 0:
            print(f"Evaluation #{self._idx_evaluation}")

        # Extract the integer decision variables (indexes)
        indexes = solution.variables

        data_dict = prepare_optimization_data_n(indexes, self._graph, self._data_dict_base)

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
                solution_store = np.array(indexes)

        if not results_nodes is None:
            results_nodes["idx_evaluation"] = np.full(
                (data_dict["node_type"].shape[0],), self._idx_evaluation)
        if not results_total is None:
            results_total["idx_evaluation"] = self._idx_evaluation

        self.store_results(solution_store, results_total, results_nodes)


def prepare_optimization_data_n(node_indexes: list[int], graph: nx.DiGraph, data_dict_base: dict[str, np.ndarray]) -> \
        dict[str, np.ndarray]:
    """
    Note: mutates `graph`.
    """
    data_dict = {key: val.copy() for key, val in data_dict_base.items()}

    # Create a list to store NodeType enum values for all nodes
    node_type_list = [NodeType.NOTHING] * len(graph.nodes)

    # Assign NodeType.WWTP to selected indexes
    for node_index in node_indexes:
        node_type_list[node_index-1] = NodeType.WWTP

    # Create a list of individual NodeType values
    node_type_values = [node_type for node_type in node_type_list]
    # Convert the node_type_list to a flat numpy array of integer values
    data_dict["node_type"] = np.array([node_type.value for node_type in node_type_list], dtype=object)
    # Convert the node_type_list to a numpy array
    data_dict["node_type"] = np.array(node_type_list, dtype=NodeType)

    for idx_node, node_type_value in enumerate(data_dict["node_type"]):
        node_data = graph.nodes[idx_node]
        node_type = NodeType(node_type_value)
        node_data["node_type"] = node_type

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

def my_logging_n(algorithm):
    print(f"Generation: {algorithm.nfe}")
    for solution in algorithm.population:
        print(f"Variable values: {solution.variables[0]}")
    print("--------------")

def run_operwa_fast_n(n_generations: int, population_size: int):
    n_runs = n_generations * population_size
    assert n_runs > 0, "Number of runs should be larger than zero."

    problem = OperwaFastN()

    if THERE_IS_INITIAL_SEED:
    # - ------------------------ Script to use initial solutions(seeds) ------------------------------
    # if there is a fileini_n.txt with initial solutions in the form (number of columns is the number of WWTPs
        # and the values are the indexes of the WWTPs)
        # 1,2,3
        # 1,2,4
        # 1,2,5
        # 1,2,6, etc.

        # Create a mapping dictionary for numeric values to NodeType
        value_to_node_type = {
            0: NodeType.NOTHING,
            1: NodeType.WWTP,
            # Add more mappings as needed
        }

        # Read the solutions from the text file
        with open(r'D:\OP_pycharm\Operwas_pump\inputs\ini_n.txt', "r") as file:
            solution_strings = file.readlines()

        # Define a function to parse a solution string and create a solution

        def parse_solution_string(solution_string):
            initial_values = ast.literal_eval(solution_string)  # Safely evaluate the string to get the list of values
            if len(initial_values) != problem.nvars:
                raise Exception("The seed file contains solutions for " + str(len(initial_values)) + " WWTPs, but n_vars was set to n_vars = " + str(problem.nvars) + ". Both should be the same, please check.")

            initial_solution_n = Solution(problem)

            initial_solution_n.variables[:] = initial_values
            problem.evaluate(initial_solution_n)
            return initial_solution_n

        # Create a list of initial solutions by parsing the solution strings
        initial_solutions = [parse_solution_string(solution_str) for solution_str in solution_strings]
        initial_results = []

        # Generate and store initial solutions along with their results
        for s in range(len(initial_solutions)):  # Adjust the number as needed
            initial_solution = initial_solutions[s]
            initial_solutions.append(initial_solution)
            initial_results.append(initial_solution.objectives)

        # Create and save the initial_results in a text file
        with open(r"D:\OP_pycharm\Operwas_pump\inputs\initial_results.txt", "w") as file:
            for result in initial_results:
                file.write(f"{result[0]}, {result[1]}\n")

        algorithm = NSGAII(problem, population=[initial_solutions], population_size=population_size, archive=[])
    # - ------------------------ end script to use initial solutions(seeds) ------------------------------
    else:  # , so, if THERE_IS_INITIAL_SEED is False
        algorithm = NSGAII(problem, population_size=population_size, archive=[])

    algorithm.run(n_runs)  # , callback=my_logging)  #
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
    plt.title("Solutions for " + str(problem.nvars) + " WWTPs")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    for _ in range(1):
        run_operwa_fast_n(n_generations=config.OPT_NUM_GENERATIONS,
                        population_size=config.OPT_POPULATION_SIZE)
print("Stop")
