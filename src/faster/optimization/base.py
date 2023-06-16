import copy
import os
from typing import Optional

import numpy as np
import pandas as pd
from platypus import Algorithm, Problem, Solution

from src.faster import config
from src.faster.custom_typing import Scalar
from src.faster.graph import create_graph, graph_to_df
from src.faster.interpolation import get_all_interpolators_combined
from src.faster.pumping_info import (get_feasible_pumping_connections,
                                     get_pumping_info)
from src.faster.utils import (create_dir_with_unique_suffix,
                              describe_data_dict,
                              get_generation_from_evaluations,
                              parse_config_to_json)


class OperwaFastBase(Problem):
    def __init__(self, n_vars: int, n_objectives: int, script_name: str):
        super().__init__(n_vars, n_objectives)

        # Read in graph
        self._graph_base = create_graph()
        self._graph = copy.deepcopy(self._graph_base)

        # Store graph data inside self
        df = graph_to_df(self._graph_base)
        df["activated"] = np.full((df.shape[0],), False)
        df["connects_to"] = np.full((df.shape[0],), -2)
        self._data_dict_base = {str(col_name): values.to_numpy() for col_name, values in df.items()}

        # Read in interpolators
        self._interpolators = get_all_interpolators_combined()

        # Read in pumping info
        self._pumping_info = get_pumping_info()
        self._data_dict_base["pumping_height"] = np.zeros((df.shape[0],))
        self._data_dict_base["pumping_length_pump"] = np.zeros((df.shape[0],))
        self._data_dict_base["pumping_length_grav"] = np.zeros((df.shape[0],))
        feasible_pumping_conns = list(
            get_feasible_pumping_connections(self._graph, self._pumping_info))

        feasible_pump_conns_list = [[j for i2, j in feasible_pumping_conns if i == i2]
                                    for i in self._graph_base.nodes()]
        self._feasible_pump_conns_list = feasible_pump_conns_list

        self._idx_evaluation = -1

        results_dir_path, unique_suffix = create_dir_with_unique_suffix(
            os.path.join(config.RUN_RESULTS_DIR, script_name), "run")

        configuration_file_name = f"config_{unique_suffix}.json"
        configuration_file_path = os.path.join(results_dir_path, configuration_file_name)
        with open(configuration_file_path, "w") as file:
            file.write(parse_config_to_json())

        solutions_file_name = f"solutions_{unique_suffix}.csv"
        self._solutions_file_path = os.path.join(results_dir_path, solutions_file_name)
        self._solutions_file_headers = ["idx_run"] + \
            [f"connect_to_{i}" for i in range(self._graph_base.number_of_nodes())]

        results_nodes_file_name = f"results_nodes_{unique_suffix}.csv"
        self._results_nodes_file_path = os.path.join(results_dir_path, results_nodes_file_name)

        results_nodes_summary_file_name = f"results_nodes_summary_{unique_suffix}.csv"
        self._results_nodes_summary_file_path = os.path.join(
            results_dir_path, results_nodes_summary_file_name)

        results_total_file_name = f"results_total_{unique_suffix}.csv"
        self._results_total_file_path = os.path.join(results_dir_path, results_total_file_name)

        results_archive_file_name = f"results_archive_{unique_suffix}.csv"
        self._results_archive_file_path = os.path.join(results_dir_path, results_archive_file_name)

    def evaluate(self, solution: Solution) -> None:
        raise NotImplementedError()

    def store_results(self, solution: Optional[np.ndarray],
                      results_total: Optional[dict[str, Scalar]],
                      results_nodes: Optional[dict[str, np.ndarray]]) -> None:
        if self._idx_evaluation == 0:
            open_mode = 'w'
            do_write_headers = True
        else:
            open_mode = 'a'
            do_write_headers = False

        if not solution is None:
            vals_to_write = [self._idx_evaluation] + solution.tolist()
            string_to_write = ','.join(map(str, vals_to_write))
            with open(self._solutions_file_path, 'a') as file:
                if do_write_headers:
                    file.write(','.join(self._solutions_file_headers) + '\n')
                file.write(string_to_write + '\n')

        if not results_total is None:
            with open(self._results_total_file_path, open_mode) as file:
                if do_write_headers:
                    header_str = ','.join(results_total.keys())
                    file.write(header_str + '\n')
                values_str = ','.join(map(str, results_total.values()))
                file.write(values_str + '\n')

        if not results_nodes is None:
            df_results_nodes = pd.DataFrame(data=results_nodes)
            with open(self._results_nodes_file_path, open_mode) as file:
                df_results_nodes.to_csv(file, header=do_write_headers, index=False)

            if not config.RUN_RESULTS_STORE_RESULTS_NODES_SUMMARY_INTERVAL is None:
                if self._idx_evaluation % config.RUN_RESULTS_STORE_RESULTS_NODES_SUMMARY_INTERVAL == 0:
                    results_nodes_summary = describe_data_dict(results_nodes)
                    results_nodes_summary["idx_evaluation"] = self._idx_evaluation
                    with open(self._results_nodes_summary_file_path, open_mode) as file:
                        if do_write_headers:
                            header_str = ','.join(results_nodes_summary.keys())
                            file.write(header_str + '\n')
                        values_str = ','.join(map(str, results_nodes_summary.values()))
                        file.write(values_str + '\n')


def store_results_archive(algorithm: Algorithm) -> None:
    # Store the population information over the entire run
    idx_archive = np.arange(len(algorithm.archive))
    idx_eval_archive = np.array([result._idx_evaluation for result in algorithm.archive])
    idx_generation_birth_archive = get_generation_from_evaluations(
        idx_eval_archive, algorithm.population_size)
    idx_generation_archive = get_generation_from_evaluations(idx_archive, algorithm.population_size)
    df_archive = pd.DataFrame(data={
        "idx_archive": idx_archive,
        "idx_evaluation": idx_eval_archive,
        "idx_generation_birth": idx_generation_birth_archive,
        "idx_generation": idx_generation_archive,
    })
    df_archive.to_csv(algorithm.problem._results_archive_file_path, index=False)
