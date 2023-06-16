import os

import numpy as np
import pandas as pd

from src.faster import config
from src.faster.simplified_aokp import (create_temporary_circular_buffer,
                                        get_feature_grid,
                                        join_attribute_network, pre_AOKP,
                                        read_outlet_locations)
from src.user_inputs import (path_channel, path_ordered_outlets,
                             path_outputBufferfn, path_temp)


def get_reuse_network_length_in_radius(idx_sc: int, radius: float, path_ds_in: str, path_buffer_out: str, network_grid_data) -> np.ndarray:
    # Create circular buffer
    temporary_buffer_file = create_temporary_circular_buffer(
        idx_sc, radius, path_ds_in, path_buffer_out)

    # Calculate pipeline network inside of buffer area (reuse network)
    reuse_network_length_in_radius = join_attribute_network(
        path_channel, network_grid_data, temporary_buffer_file, 'Length')

    # There is only one circular buffer, so there is only one value, but we want
    # to keep it a numpy array for consistency with the other "get_smth_in_radius"
    # functions.
    return reuse_network_length_in_radius[:]


def get_reuse_network_length_in_radiuses(idx_sc: int, radiuses: np.ndarray, path_ds_in: str, path_buffer_out: str, network_grid_data) -> np.ndarray:
    n_samples = len(radiuses)
    reuse_network_lengths = np.full((n_samples, 1), np.nan)
    for i, radius in enumerate(radiuses):
        reuse_network_lengths[i] = get_reuse_network_length_in_radius(
            idx_sc, radius, path_ds_in, path_buffer_out, network_grid_data)[0]
    return reuse_network_lengths


def get_reuse_network_length_data(idx_sc: int, radiuses: np.ndarray, path_ds_in: str, path_buffer_out: str, network_grid_data) -> pd.DataFrame:
    reuse_network_length_in_radiuses = get_reuse_network_length_in_radiuses(
        idx_sc, radiuses, path_ds_in, path_buffer_out, network_grid_data)
    data_dict = {
        "radius": radiuses,
        "reuse_network_length": reuse_network_length_in_radiuses[:, 0],
    }
    return pd.DataFrame(data=data_dict)


def run_reuse_network_length_experiments():
    coordinates, _ = read_outlet_locations(config.OUTLETS_FILE_PATH)

    sc2outlet = pre_AOKP(coordinates)
    outlet2sc = {idx_outlet: idx_sc for idx_sc,
                 idx_outlet in sc2outlet.items()}

    network_grid_data = get_feature_grid(
        path_channel, 'channel_grid', path_temp)

    radiuses = np.linspace(0, config.EXP_NETWORKLENGTH_RADIUS_MAX,
                           config.EXP_NETWORKLENGTH_N_SAMPLES)

    if not os.path.isdir(config.EXP_NETWORKLENGTH_RESULTS_DIR_PATH):
        os.makedirs(config.EXP_NETWORKLENGTH_RESULTS_DIR_PATH)

    n_outlets = len(coordinates)
    print("Running population per town experiments:")
    for idx_outlet in range(n_outlets):
        idx_sc = outlet2sc[idx_outlet]
        df_reuse_network_length = get_reuse_network_length_data(
            idx_sc, radiuses, path_ordered_outlets, path_outputBufferfn, network_grid_data)
        file_name = f"{config.EXP_NETWORKLENGTH_RESULTS_FILE_PREFIX}__outlet_{idx_outlet:003}.csv"
        save_path = os.path.join(config.EXP_NETWORKLENGTH_RESULTS_DIR_PATH, file_name)
        df_reuse_network_length.to_csv(save_path, index=False)
        print(
            f"\tFinished experiments for outlet {idx_outlet+1}/{n_outlets} ({(idx_outlet+1)/n_outlets * 100:.1f}%)")


if __name__ == "__main__":
    run_reuse_network_length_experiments()
