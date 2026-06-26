import itertools
from typing import List
import logging
import mosek
from itertools import combinations
import pandas as pd
import os
import numpy as np
from fastsdp_tools import get_m_indexes_of_higher_values_in_list

from fastsdp_tools import (
    get_project_path,
    create_folder,
    remove_values_of_list_from_list,
    create_folder_benchmark,
)

logger_mosek = logging.getLogger("Mosek_logger")

all_possible_cuts = [
    "RLT",
    "triangularization",
    "McCormick_beta_z",
    "beta_logits_comparaison",
    "beta_logits_comparaison_big_M",
    "sum_beta_logits_equal_logit",
]


def compute_cuts_str(cuts: List):
    """
    Compute the cuts string.
    """
    cuts_str = "_".join(cuts)
    cuts_str = cuts_str.replace("^", "")
    return cuts_str


def create_all_cuts_to_test(self):
    """
    Create all cuts to test.
    """
    self.cuts_to_test = [[]]

    if self.cuts is not None:
        if self.all_combinations_cuts:
            if self.name == "TargetedSDP":
                self.cuts = remove_values_of_list_from_list(
                    self.cuts,
                    [
                        "betaibetaj",
                        "Adversariales",
                        "Tij",
                        "zbar",
                    ],
                )
            elif self.name == "UntargetedSDP":
                self.cuts = remove_values_of_list_from_list(self.cuts, ["zbar"])

            for r in range(1, len(self.cuts) + 1):
                for combo in combinations(self.cuts, r):
                    self.cuts_to_test.append(list(combo))
        else:
            self.cuts_to_test = [self.cuts]




def compute_number_RLT(self) -> int:
    """
    Compute the number of RLT constraints to be added.
    Returns:
        int: Number of RLT constraints.
    """
    n_rlt = 0
    for layer in range(1, self.K+1 if self.LAST_LAYER else self.K):
        for neuron_next in range(self.n[layer]):
            if (layer, neuron_next) in self.stable_inactives_neurons:
                
                continue
            if (layer, neuron_next) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or layer != self.K - 1
            ):
                
                continue
            n_cstr = int(self.RLT_prop * self.n[layer - 1])
            indexes_pruned = [
                j
                for j in range(self.n[layer - 1])
                if (layer - 1, j) in self.stable_inactives_neurons
                or (layer - 1, j) in self.stable_actives_neurons
            ]
            neurons_with_great_weights = get_m_indexes_of_higher_values_in_list(
                np.abs(self.W[layer - 1][neuron_next]), n_cstr, indexes_pruned
            )
            n_rlt += len(neurons_with_great_weights)
    return n_rlt

def adapt_number_RLT(self, max_n_rlt : int = 5e5):
    
    if "RLT" not in self.cuts:
        logger_mosek.debug("RLT not activated, skipping adaptation of number of RLT constraints.")
        return
    n_rlt = self.compute_number_RLT()
    logger_mosek.debug(f"RLT : Current number of RLT constraints to be added : {n_rlt}")
    if n_rlt > max_n_rlt:
        new_RLT_prop = round(self.RLT_prop * max_n_rlt / n_rlt, 2)
        logger_mosek.debug(f"RLT: Adapt number of RLT constraints from {self.RLT_prop} to {new_RLT_prop}")
        self.RLT_prop = new_RLT_prop


def print_solution_to_file_for_cb_solver(mat, index_matrix, dim, file_cb):
    """
    Print the solutions of the optimization to a file.
    """
    logger_mosek.info("Writing solutions to conic bundle file...")
    for j in range(dim):
        for i in range(j):
            file_cb.write(f"{index_matrix} {i} {j} {mat[i][j]} ")
            file_cb.write("\n")


def print_dual_variable_to_file_for_cb_solver(list_cstr, file_cb):
    """
    Print the dual variables of the optimization to a file.
    """
    logger_mosek.info("Writing dual variables to conic bundle file...")
    for ind, cstr in enumerate(list_cstr):
        name = cstr["name"]
        dual_value = cstr["dual_value"]
        file_cb.write(f"{name} : {dual_value} ")
        file_cb.write("\n")


def concat_dataframes_with_missing_columns(df1, df2):
    """
    Concatenate two DataFrames, filling missing columns with None.
    """
    if df1 is None and df2 is None:
        return pd.DataFrame()
    if df1 is None:
        return df2.copy()
    if df2 is None:
        return df1.copy()

    df1_copy = df1.copy()
    df2_copy = df2.copy()

    all_columns = set(df1_copy.columns).union(set(df2_copy.columns))

    for col in all_columns - set(df1_copy.columns):
        df1_copy[col] = None

    for col in all_columns - set(df2_copy.columns):
        df2_copy[col] = None

    df1_copy = df1_copy[sorted(all_columns)]
    df2_copy = df2_copy[sorted(all_columns)]

    result_df = pd.concat([df1_copy, df2_copy], ignore_index=True)

    return result_df


def replace_none_with_false(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Replace None/NaN values in a specific column with False.
    """
    df_copy = df.copy()

    if column_name not in df_copy.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    df_copy[column_name] = df_copy[column_name].fillna(False)

    return df_copy


def check_cuts(row):
    """
    Build a string representation of active cuts from a result row.
    """
    cuts_str = ""
    if row["Tij"]:
        cuts_str += "Tij, "
    if row["triangularization"]:
        cuts_str += "tri, "
    if row["RLT"]:
        cuts_str += "RLT, "
    if row["allMC"]:
        cuts_str += "allMC, "
    if (
        not row["Tij"]
        and not row["RLT"]
        and not row["triangularization"]
        and not row["allMC"]
    ):
        cuts_str += "$\\emptyset$"
    else:
        cuts_str = cuts_str[:-2]
    return cuts_str


def add_cuts_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all cut columns exist in the DataFrame (fill missing ones with False).
    """
    for cut in all_possible_cuts:
        if cut not in df.columns:
            df[cut] = False
    return df
