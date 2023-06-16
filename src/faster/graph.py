import os

import networkx as nx
import numpy as np
import pandas as pd

from src.faster import config
from src.faster.custom_typing import Connection, Coordinate
from src.faster.node_types import NodeType
from src.faster.plotting import plot_with_graph
from src.faster.simplified_aokp import read_outlet_locations


def graph_to_df(graph: nx.DiGraph) -> pd.DataFrame:
    idx_nodes_sorted = sorted(graph.nodes())
    node_datas = [graph.nodes[idx_node] for idx_node in idx_nodes_sorted]
    df = pd.DataFrame(data=node_datas)
    return df


def parse_coordinate_string(coordinate_str: str) -> Coordinate:
    x_str, y_str = coordinate_str.strip('()').split(',')
    return (float(x_str), float(y_str))


def create_graph_from_data_and_connections(subcatchment_data: pd.DataFrame, connections: list[Connection]) -> nx.DiGraph:
    node_attrs = subcatchment_data.to_dict(orient="index")
    for node_attr in node_attrs.values():
        node_attr["node_type"] = NodeType.NOTHING
    graph = nx.DiGraph()
    graph.add_nodes_from([(node_attr["idx_node"], node_attr)
                         for node_attr in node_attrs.values()])
    graph.add_weighted_edges_from(connections)
    return graph


def get_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if isinstance(df["coordinates"][0], str):
        df["coordinates"] = list(
            map(parse_coordinate_string, df["coordinates"]))
    return df


def get_connections_from_data_one_missing(col_to_use: str = "population_served") -> list[Connection]:
    df_all = get_data(config.GRAPH_DATA_FILE_PATH)

    connections = []
    for idx_node_missing in range(df_all.shape[0]):
        file_path_one_missing = os.path.join(config.EXP_SUBCATCHMENTS_RESULTS_DIR_PATH,
                                             f"{config.EXP_SUBCATCHMENTS_RESULTS_FILE_PREFIX}__missing_outlet_{idx_node_missing:03}.csv")
        df_one_missing = get_data(file_path_one_missing)
        suffix_right = "__missing_{idx_node_missing}"
        df_joined = pd.merge(df_all, df_one_missing, how="left",
                             on="coordinates", suffixes=("", suffix_right))

        values_diff = df_joined[col_to_use + suffix_right] - df_joined[col_to_use]
        ratios = values_diff / df_joined[col_to_use][idx_node_missing]

        for idx_node, ratio in zip(df_joined["idx_node"], ratios):
            if ratio != 0.0 and not np.isnan(ratio):
                connections.append((idx_node_missing, idx_node, ratio))

    return connections


def create_graph_from_data_one_missing(col_to_use: str = "population_served") -> nx.DiGraph:
    df = get_data(config.GRAPH_DATA_FILE_PATH)
    connections = get_connections_from_data_one_missing(col_to_use=col_to_use)
    graph = create_graph_from_data_and_connections(df, connections)
    return graph


def create_graph() -> nx.DiGraph:
    subcatchment_data = get_data(config.GRAPH_DATA_FILE_PATH)
    _, connections = read_outlet_locations(config.OUTLETS_FILE_PATH)
    graph = create_graph_from_data_and_connections(subcatchment_data, connections)
    return graph


GRAPH_DF = graph_to_df(create_graph())
POPULATION_SERVED_MAX = GRAPH_DF["population_served"].sum()
NETWORK_LENGTH_MAX = GRAPH_DF["network_length"].sum()


def feasibility_check(data_dict: dict[str, np.ndarray]) -> None:
    # Small value to account for numerical differences
    epsilon = 1e-1
    # Check population served by treatment plants (wwps populations do not count since
    # ultimately they are served by the treatment plants)
    if data_dict["population_served"][data_dict["node_type"] == NodeType.WWTP].sum() > POPULATION_SERVED_MAX + epsilon:
        print("population of ", data_dict["population_served"].sum(), "detected")
        raise ValueError("Population served by treatment plants exceeds maximum in graph.")
    # Check network length of all activated nodes
    if data_dict["network_length"].sum() > NETWORK_LENGTH_MAX + epsilon:
        print("population of ", data_dict["population_served"].sum(), "detected")
        raise ValueError("Network length exceeds maximum in graph.")


if __name__ == "__main__":
    # graph = create_graph()
    graph = create_graph_from_data_one_missing(col_to_use="population_served")
    fig = plot_with_graph(graph)
