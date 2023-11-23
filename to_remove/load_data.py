import json
import os
from typing import Iterable, Optional

import pandas as pd

from src.faster.custom_typing import Scalar


def get_suffix_from_dir(dir_path: str) -> str:
    result_dir_name = os.path.split(dir_path)[-1]
    suffix = result_dir_name.removeprefix("run_")
    return suffix


def load_result(result_dir_path: str, prefix: str, keep_cols: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    suffix = get_suffix_from_dir(result_dir_path)
    file_name = f"{prefix}_{suffix}.csv"
    file_path = os.path.join(result_dir_path, file_name)
    if not os.path.isfile(file_path):
        print(f"Cannot find {file_name} in {result_dir_path}")
        return None
    df = pd.read_csv(file_path)
    if keep_cols is not None:
        df = df[keep_cols]
    df["id_run"] = suffix
    return df


def load_results(result_dir_paths: str | Iterable[str], prefix: str, keep_cols: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    if isinstance(result_dir_paths, str):
        result_dir_paths = [result_dir_paths]
    df_total = None
    for result_dir_path in result_dir_paths:
        df = load_result(result_dir_path, prefix, keep_cols=keep_cols)
        if df is not None:
            if df_total is None:
                df_total = df
            else:
                df_total = pd.concat([df_total, df], axis=0)
    if not df_total is None:
        df_total.reset_index(drop=True, inplace=True)
    return df_total


def load_results_solutions(result_dir_path: str | Iterable[str], keep_cols: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    return load_results(result_dir_path, "solutions", keep_cols=keep_cols)


def load_results_total(result_dir_path: str | Iterable[str], keep_cols: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    return load_results(result_dir_path, "results_total", keep_cols=keep_cols)


def load_results_nodes(result_dir_path: str | Iterable[str], keep_cols: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    return load_results(result_dir_path, "results_nodes", keep_cols=keep_cols)


def load_results_nodes_summary(result_dir_path: str | Iterable[str], keep_cols: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    return load_results(result_dir_path, "results_nodes_summary", keep_cols=keep_cols)


def load_results_archive(result_dir_path: str | Iterable[str], keep_cols: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    return load_results(result_dir_path, "results_archive", keep_cols=keep_cols)


def load_config_as_dict(result_dir_path: str) -> Optional[dict[str, Scalar]]:
    suffix = get_suffix_from_dir(result_dir_path)
    file_name = f"config_{suffix}.json"
    file_path = os.path.join(result_dir_path, file_name)
    if not os.path.isfile(file_path):
        print(f"Cannot find {file_name} in {result_dir_path}")
        return None
    with open(file_path, "r") as file:
        config_dict = dict(json.load(file))
        config_dict["id_run"] = suffix
        return config_dict


def load_config_as_df(result_dir_paths: str | Iterable[str]) -> Optional[pd.DataFrame]:
    if isinstance(result_dir_paths, str):
        result_dir_paths = [result_dir_paths]
    df_total = None
    for result_dir_path in result_dir_paths:
        config_dict = load_config_as_dict(result_dir_path)
        if config_dict is not None:
            df = pd.DataFrame(data=config_dict, index=[0])
            if df_total is None:
                df_total = df
            else:
                df_total = pd.concat([df_total, df], axis=0)
    if not df_total is None:
        df_total.reset_index(drop=True, inplace=True)
    return df_total
