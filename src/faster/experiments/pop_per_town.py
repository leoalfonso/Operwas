import os

import numpy as np
import pandas as pd

from src.faster import config
from src.faster.simplified_aokp import (create_temporary_circular_buffer,
                                        get_feature_grid,
                                        join_attribute_data_in_boundary,
                                        pre_AOKP, read_outlet_locations)
from src.user_inputs import (path_ordered_outlets, path_outputBufferfn,
                             path_population, path_temp)


def get_pop_per_town_in_radius(idx_sc: int, radius: float, path_ds_in: str, path_buffer_out: str, feature_grid_data) -> np.ndarray:
    # Create circular buffer
    temporary_buffer_file = create_temporary_circular_buffer(
        idx_sc, radius, path_ds_in, path_buffer_out)

    # Calculation of population in each buffer
    pop_per_town_in_buffer = join_attribute_data_in_boundary(path_to_datafile=path_population,
                                                             feature_grid_data=feature_grid_data,
                                                             polygonized_file=temporary_buffer_file,
                                                             field_name='sumInhabit',
                                                             location_index_field_name='Loc_index',
                                                             num_types_data=4)

    # There is only one circular buffer, so there is only one column
    return pop_per_town_in_buffer[:]


def get_pop_per_town_in_radiuses(idx_sc: int, radiuses: np.ndarray, path_ds_in: str, path_buffer_out: str, feature_grid_data) -> np.ndarray:
    n_samples = len(radiuses)
    n_towns = get_pop_per_town_in_radius(
        idx_sc, 1.0, path_ds_in, path_buffer_out, feature_grid_data).shape[0]
    pop_per_town_in_radiuses = np.full((n_samples, n_towns), np.nan)
    for i_radius, radius in enumerate(radiuses):
        pop_per_town_in_radius = get_pop_per_town_in_radius(
            idx_sc, radius, path_ds_in, path_buffer_out, feature_grid_data)
        for i_town, pop in enumerate(pop_per_town_in_radius):
            pop_per_town_in_radiuses[i_radius, i_town] = pop
    return pop_per_town_in_radiuses


def get_pop_per_town_data(idx_sc: int, radiuses: np.ndarray, path_ds_in: str, path_buffer_out: str, feature_grid_data) -> pd.DataFrame:
    pop_per_town_in_radiuses = get_pop_per_town_in_radiuses(
        idx_sc, radiuses, path_ds_in, path_buffer_out, feature_grid_data)
    n_towns = pop_per_town_in_radiuses.shape[1]
    town_names = [f"town_{i}" for i in range(n_towns)]
    data_dict = {"radius": radiuses}
    data_dict.update({
        f"population__{name}": pop_per_town_in_radiuses[:, i] for i, name in enumerate(town_names)})
    return pd.DataFrame(data=data_dict)


def run_pop_per_town_experiments():
    coordinates, _ = read_outlet_locations(config.OUTLETS_FILE_PATH)

    sc2outlet = pre_AOKP(coordinates)
    outlet2sc = {idx_outlet: idx_sc for idx_sc,
                 idx_outlet in sc2outlet.items()}

    feature_grid_data = get_feature_grid(
        path_population, 'feature_grid', path_temp)

    radiuses = np.linspace(0, config.EXP_POPULATION_RADIUS_MAX, config.EXP_POPULATION_N_SAMPLES)

    if not os.path.isdir(config.EXP_POPULATION_RESULTS_DIR_PATH):
        os.makedirs(config.EXP_POPULATION_RESULTS_DIR_PATH)

    n_outlets = len(coordinates)
    print("Running population per town experiments:")
    for idx_outlet in range(n_outlets):
        idx_sc = outlet2sc[idx_outlet]
        df_pop_per_town = get_pop_per_town_data(
            idx_sc, radiuses, path_ordered_outlets, path_outputBufferfn, feature_grid_data)
        file_name = f"{config.EXP_POPULATION_RESULTS_FILE_PREFIX}__outlet_{idx_outlet:003}.csv"
        save_path = os.path.join(config.EXP_POPULATION_RESULTS_DIR_PATH, file_name)
        df_pop_per_town.to_csv(save_path, index=False)
        print(
            f"\tFinished experiments for outlet {idx_outlet+1}/{n_outlets} ({(idx_outlet+1)/n_outlets * 100:.1f}%)")


if __name__ == "__main__":
    run_pop_per_town_experiments()
