import os

import numpy as np
import pandas as pd

from src.faster import config
from src.faster.simplified_aokp import (create_temporary_circular_buffer,
                                        get_feature_grid,
                                        join_attribute_data_in_boundary,
                                        pre_AOKP, read_outlet_locations)
from src.user_inputs import (path_ordered_outlets,
                             path_outputBufferfn, path_population,
                             path_temp)


def get_area_per_reuse_in_radius(idx_sc: int, radius: float, path_ds_in: str, path_buffer_out: str, feature_grid_data) -> np.ndarray:
    """
                    This function calculates the optimal flow to be used in an area.

                    :param:
                    :return:
    """
    # Create circular buffer
    temporary_buffer_file = create_temporary_circular_buffer(
        idx_sc, radius, path_ds_in, path_buffer_out)

    # Get area per reuse type in buffer
    area_per_reuse = join_attribute_data_in_boundary(path_to_datafile=path_population,
                                                     feature_grid_data=feature_grid_data,
                                                     polygonized_file=temporary_buffer_file,
                                                     field_name='area_pixel',
                                                     location_index_field_name='Type_R_ind',
                                                     num_types_data=3)
    # There is only one circular buffer, so there is only one column
    return area_per_reuse[:]


def get_area_per_reuse_in_radiuses(idx_sc: int, radiuses: np.ndarray, path_ds_in: str, path_buffer_out: str, feature_grid_data) -> np.ndarray:
    n_samples = len(radiuses)
    # Get the number of reuse types by running the calculation with small radius and
    # checking the size
    n_reuse_types = get_area_per_reuse_in_radius(
        idx_sc, 1.0, path_ds_in, path_buffer_out, feature_grid_data).shape[0]
    area_per_reuse_in_radiuses = np.full((n_samples, n_reuse_types), np.nan)
    for i_radius, radius in enumerate(radiuses):
        area_per_reuse_in_radius = get_area_per_reuse_in_radius(
            idx_sc, radius, path_ds_in, path_buffer_out, feature_grid_data)
        for i_town, pop in enumerate(area_per_reuse_in_radius):
            area_per_reuse_in_radiuses[i_radius, i_town] = pop
    return area_per_reuse_in_radiuses


def get_area_per_reuse_data(idx_sc: int, radiuses: np.ndarray, path_ds_in: str, path_buffer_out: str, feature_grid_data) -> pd.DataFrame:
    area_per_reuse_in_radiuses = get_area_per_reuse_in_radiuses(
        idx_sc, radiuses, path_ds_in, path_buffer_out, feature_grid_data)
    # Even though there are 3 apparent types of reuse we get from the calculation
    # we only are interested in the first two
    data_dict = {"radius": radiuses}
    types_reuse = ["agricultural", "urban"]
    data_dict.update({
        f"area__{type_reuse}": area_per_reuse_in_radiuses[:, i] for i, type_reuse in enumerate(types_reuse)})
    return pd.DataFrame(data=data_dict)


def run_area_per_reuse_experiments():
    coordinates, _ = read_outlet_locations(config.OUTLETS_FILE_PATH)

    sc2outlet = pre_AOKP(coordinates)
    outlet2sc = {idx_outlet: idx_sc for idx_sc,
                 idx_outlet in sc2outlet.items()}

    feature_grid_data = get_feature_grid(
        path_population, 'feature_grid', path_temp)

    radiuses = np.linspace(0, config.EXP_AREA_RADIUS_MAX, config.EXP_AREA_N_SAMPLES)

    if not os.path.isdir(config.EXP_AREA_RESULTS_DIR_PATH):
        os.makedirs(config.EXP_AREA_RESULTS_DIR_PATH)

    n_outlets = len(coordinates)
    print("Running area per reuse experiments:")
    for idx_outlet in range(n_outlets):
        idx_sc = outlet2sc[idx_outlet]
        df_area_per_reuse = get_area_per_reuse_data(
            idx_sc, radiuses, path_ordered_outlets, path_outputBufferfn, feature_grid_data)
        file_name = f"{config.EXP_AREA_RESULTS_FILE_PREFIX}__outlet_{idx_outlet:003}.csv"
        save_path = os.path.join(config.EXP_AREA_RESULTS_DIR_PATH, file_name)
        df_area_per_reuse.to_csv(save_path, index=False)
        print(
            f"\tFinished experiments for outlet {idx_outlet+1}/{n_outlets} ({(idx_outlet+1)/n_outlets * 100:.1f}%)")


if __name__ == "__main__":
    run_area_per_reuse_experiments()
