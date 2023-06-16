import os

import numpy as np
import pandas as pd

from src.faster import config
from src.faster.custom_typing import Coordinate
from src.faster.simplified_aokp import (get_feature_grid,
                                        join_attribute_data_at_outlet,
                                        join_attribute_data_in_boundary,
                                        join_attribute_network, pre_AOKP,
                                        read_outlet_locations)
from src.user_inputs import (no_data_value_land_price, path_channel,
                             path_ordered_outlets, path_population,
                             path_subcatchments, path_temp, no_data_value_loc_index)


def get_subcatchment_data(idx_nodes: list[int], coordinates: list[Coordinate]) -> pd.DataFrame:
    sc2outlet = pre_AOKP(coordinates)
    outlet2sc = {idx_outlet: idx_sc for idx_sc,
                 idx_outlet in sc2outlet.items()}

    # Get the features index in the grid file, to be used in further calculations
    feature_grid_data = get_feature_grid(
        path_population, 'feature_grid', path_temp)
    network_grid_data = get_feature_grid(
        path_channel, 'channel_grid', path_temp)

    # Calculation of population in each subcatchment
    pop_per_town_per_subcatchment = join_attribute_data_in_boundary(path_to_datafile=path_population,
                                                                    feature_grid_data=feature_grid_data,
                                                                    polygonized_file=path_subcatchments,
                                                                    field_name='sumInhabit',
                                                                    location_index_field_name='Loc_index',
                                                                    num_types_data=4)

    networklength_per_subcatchment = join_attribute_network(
        path_channel, network_grid_data, path_subcatchments, 'LENGTH')

    land_cost_at_outlets = join_attribute_data_at_outlet(path_ordered_outlets, feature_grid_data,
                                                         path_population, "Land_price", no_data_value_land_price)

    town_at_outlets = join_attribute_data_at_outlet(
        path_ordered_outlets, feature_grid_data, path_population, 'Loc_index', no_data_value_loc_index)

    idx_scs = np.array([outlet2sc[idx_node] for idx_node in range(len(coordinates))])

    n_towns = pop_per_town_per_subcatchment.shape[0]

    pop_data = dict()
    pop_served_total = np.zeros((len(idx_nodes,)))
    for i_town in range(n_towns):
        pop_per_subcatchment_for_town_i = pop_per_town_per_subcatchment[i_town, :]
        pop_per_subcatchment_for_town_i_ordered = pop_per_subcatchment_for_town_i[idx_scs]
        pop_data[f"population_served__town_{i_town}"] = pop_per_subcatchment_for_town_i_ordered
        pop_served_total += pop_per_subcatchment_for_town_i_ordered
    pop_data["population_served"] = pop_served_total

    networklength_per_subcatchment_ordered = networklength_per_subcatchment[idx_scs]
    land_cost_at_outlets_ordered = np.array(land_cost_at_outlets)[idx_scs]
    town_at_outlets_ordered = np.array([int(x) for x in town_at_outlets])[idx_scs]

    df_data = {
        "idx_node": idx_nodes,
        "coordinates": coordinates,
        "network_length": networklength_per_subcatchment_ordered,
        "land_cost": land_cost_at_outlets_ordered,
        "idx_town": town_at_outlets_ordered,
    }

    # Add population served per town columns
    df_data.update(pop_data)

    return pd.DataFrame(data=df_data)


def save_df(df: pd.DataFrame, save_path: str) -> None:
    save_dir, _ = os.path.split(save_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    df.to_csv(save_path, index=False)


def run_subcatchment_experiments():
    print("Running subcatchment experiments...")
    coordinates, _ = read_outlet_locations(config.OUTLETS_FILE_PATH)
    idx_nodes = list(range(len(coordinates)))
    df_data_all = get_subcatchment_data(idx_nodes, coordinates)
    save_df(df_data_all, config.GRAPH_DATA_FILE_PATH)
    for i_missing in range(len(coordinates)):
        print(f"Starting experiments with node #{i_missing} missing...")
        idx_nodes_one_missing = [i for i in range(len(coordinates)) if i != i_missing]
        coordinates_one_missing = [coordinates[i] for i in idx_nodes_one_missing]
        df_data_one_missing = get_subcatchment_data(idx_nodes_one_missing, coordinates_one_missing)
        save_path_one_missing = os.path.join(config.EXP_SUBCATCHMENTS_RESULTS_DIR_PATH,
                                             f"{config.EXP_SUBCATCHMENTS_RESULTS_FILE_PREFIX}__missing_outlet_{i_missing:03}.csv")
        save_df(df_data_one_missing, save_path_one_missing)
    print("Done.")


if __name__ == "__main__":
    run_subcatchment_experiments()
