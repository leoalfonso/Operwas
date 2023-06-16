import json
import os
import random
import string
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.faster import config


def parse_config_to_dict() -> dict[str, Any]:
    vars = [x for x in dir(config) if x.upper() == x and not x.startswith("_")]
    return {var: getattr(config, var) for var in vars}


def parse_config_to_json() -> str:
    return json.dumps(parse_config_to_dict(), indent=4, default=str)


def random_code(length: int) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choices(alphabet, k=length))


def create_dir_with_unique_suffix(parent_dir: str, prefix: str) -> tuple[str, str]:
    now_str = datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S-%f")[:-3]
    unique_suffix = now_str
    while os.path.exists(os.path.join(parent_dir, f"{prefix}_{unique_suffix}")):
        # Results dir already exists, need to create a unique name
        unique_suffix = f"{now_str}-{random_code(8)}"
    results_dir_path = os.path.join(parent_dir, f"{prefix}_{unique_suffix}")
    os.makedirs(results_dir_path)
    return results_dir_path, unique_suffix


def sort_by_x(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx_sorted = np.argsort(xs)
    return xs[idx_sorted], ys[idx_sorted]


def remove_duplicates_in_x(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diffs = np.diff(xs)
    is_non_dup = np.full(xs.shape, False)
    # Always keep the first item
    is_non_dup[0] = True
    is_non_dup[1:] = diffs != 0.0
    return xs[is_non_dup], ys[is_non_dup]


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_describe = df.describe()
    data_dict = dict()
    for colname, values in df_describe.items():
        for valname, value in values.items():
            data_dict[f"{colname}_{valname}"] = value
    return pd.DataFrame(data=data_dict, index=[0])


def describe_column(xs: np.ndarray) -> dict[str, int | float]:
    return {
        "count": xs.shape[0],
        "mean": xs.mean(),
        "std": xs.std(),
        "min": xs.min(),
        "max": xs.max(),
        "p25": np.percentile(xs, 25),
        "p50": np.median(xs),
        "p75": np.percentile(xs, 75),
        "count_zero": (xs == 0).sum(),
    }


def describe_data_dict(data_dict: dict[str, np.ndarray], keys: Optional[list[str]] = None) -> dict[str, int | float]:
    accepted_dtypes = [
        np.dtype('float64'),
        np.dtype('int64'),
    ]

    if keys is None:
        keys = list(data_dict.keys())

    data_dict_summary = dict()
    for key in keys:
        if data_dict[key].dtype in accepted_dtypes:
            value_dict_summary = describe_column(data_dict[key])
            data_dict_summary.update({
                f"{key}_{valname}": val for valname, val in value_dict_summary.items()})

    return data_dict_summary


def get_generation_from_evaluations(idx_evaluations: np.ndarray, population_size: int) -> np.ndarray:
    return idx_evaluations // population_size


def get_generation_from_evaluation(idx_evaluations: int, population_size: int) -> int:
    return idx_evaluations // population_size
