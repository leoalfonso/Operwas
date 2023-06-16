import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from src.faster import config
from src.faster.custom_typing import DataInterpolator
from src.faster.utils import remove_duplicates_in_x, sort_by_x


def make_interpolator(xs: np.ndarray, ys: np.ndarray) -> DataInterpolator:
    if np.all(xs == 0.0):
        print("Warning: creating interpolator where x-values are all zero.")
        return lambda _: float('inf')

    xs, ys = sort_by_x(xs, ys)
    xs, ys = remove_duplicates_in_x(xs, ys)

    def interpolator(x: float) -> float:
        if xs[0] <= x <= xs[-1]:
            y: float = np.interp(x, xs, ys)
            return y
        else:
            raise ValueError("Input value lies outside of interpolator data bounds.")

    return interpolator


def create_interpolators_from_data(df: pd.DataFrame, reverse=False) -> dict[str, DataInterpolator]:
    radiuses = df.pop("radius").to_numpy(copy=True)
    if reverse:
        return {f"{col}_to_radius": make_interpolator(values.to_numpy(copy=True), radiuses) for col, values in df.items()}
    else:
        return {f"radius_to_{col}": make_interpolator(radiuses, values.to_numpy(copy=True)) for col, values in df.items()}


def combine_interpolators(list_interpolators: list[dict[int, dict[str, DataInterpolator]]]) -> dict[int, dict[str, DataInterpolator]]:
    interpolators_combined: dict[int, dict[str, DataInterpolator]] = dict()
    for interpolatorsa in list_interpolators:
        for idx, interpolators in interpolatorsa.items():
            if idx in interpolators_combined:
                # Check for overlapping keys. If found, make sure that the columns in the
                # result files have unique names (except for radiuses of course)
                if not interpolators_combined[idx].keys().isdisjoint(interpolators.keys()):
                    overlapping_keys = set(interpolators_combined[idx].keys()).intersection(
                        set(interpolators.keys()))
                    raise ValueError(
                        f"Found overlapping keys while combining interpolators: {overlapping_keys}")
                interpolators_combined[idx].update(interpolators)
            else:
                interpolators_combined[idx] = interpolators
    return interpolators_combined


def filename_to_outletidx(filename: str) -> Optional[int]:
    matcher = re.compile(r"__outlet_(\d+).csv")
    matches = matcher.findall(filename)
    if len(matches) == 0:
        return None
    else:
        return int(matcher.findall(filename)[0])


def create_interpolators_from_dir(dir_path: str, reverse=False) -> dict[int, dict[str, DataInterpolator]]:
    all_interpolators = dict()
    for file_name in os.listdir(dir_path):
        idx_outlet = filename_to_outletidx(file_name)
        if idx_outlet is None:
            print(f"Skipping file: {file_name}")
            continue
        file_path = os.path.join(dir_path, file_name)
        df = pd.read_csv(file_path)
        all_interpolators[idx_outlet] = create_interpolators_from_data(df, reverse=reverse)
    return all_interpolators


def get_all_interpolators() -> list[dict[int, dict[str, DataInterpolator]]]:
    dir_paths = {
        config.EXP_AREA_RESULTS_DIR_PATH: True,
        config.EXP_NETWORKLENGTH_RESULTS_DIR_PATH: False,
        config.EXP_POPULATION_RESULTS_DIR_PATH: False,
    }
    return [create_interpolators_from_dir(dir_path, reverse=reverse) for dir_path, reverse in dir_paths.items()]


def get_all_interpolators_combined() -> dict[int, dict[str, DataInterpolator]]:
    all_interpolators = get_all_interpolators()
    return combine_interpolators(all_interpolators)


if __name__ == "__main__":
    all_interpolators = get_all_interpolators_combined()
    value = 1e1
    idx_outlets = sorted(list(all_interpolators.keys()))
    for idx_outlet in idx_outlets:
        interpolators = all_interpolators[idx_outlet]
        for name, interpolator in interpolators.items():
            print(f"\tFor interpolator: {name}")
            interp_value = interpolator(value)
            print(f"\tFor interpolator: {name}, interpolated value = {interp_value}")
