from typing import List
import numpy as np
import mosek
import logging
import os

from fastsdp_tools import get_project_path
from solve.sdp_solve.run_benchmark import compute_cuts_str


logger_mosek = logging.getLogger("Mosek_logger")


def add_all_infos_optimal_values_to_dic(self, cuts: List, verbose: bool = False):
    """
    Add all the information about the optimal values found to the dictionnary for the benchmark.
    """
    self.primal_obj_value = self.model.primalObjValue()

    logger_mosek.debug(
        "Optimal solution found with objective value: %s", self.primal_obj_value
    )
    self.dual_obj_value = self.model.dualObjValue()
    logger_mosek.info("Dual objective value: %s", self.dual_obj_value)
    
    self.optimal_value = self.primal_obj_value  
    self.is_robust = self.optimal_value >= 0

    gap = abs(self.dual_obj_value - self.primal_obj_value)
    if gap > 0.01 * max(1.0, abs(self.dual_obj_value)):
        
        try:
            self.diagnose_infeasibility()
        except Exception as e:
            logger_mosek.warning("Slater diagnostic failed : %s", e)

    if self.indexes_matrices.BETAS_Z:
        self.save_beta_values_fusion(cuts)
    dic_sol = {"optimal_value": self.optimal_value}
    dic_sol.update({"primal_obj_value": self.primal_obj_value})
    dic_sol.update({"dual_obj_value": self.dual_obj_value})
    return dic_sol


def is_status_optimal(self):
    return self.model.getPrimalSolutionStatus() == mosek.fusion.SolutionStatus.Optimal


def is_status_infeasible(self):
    return (
        self.model.getProblemStatus() == mosek.fusion.ProblemStatus.PrimalInfeasible
        or self.model.getProblemStatus() == mosek.fusion.ProblemStatus.DualInfeasible
    )


def is_status_unknown(self):
    return self.model.getPrimalSolutionStatus() == mosek.fusion.SolutionStatus.Unknown


def save_beta_values_fusion(self, cuts: List):
    """
    Extract betas solutions values and save them in a file.
    """
    cuts_str = compute_cuts_str(cuts)
    model_dir = get_project_path(f"{self.folder_name}/{self.name}")
    os.makedirs(model_dir, exist_ok=True)

    beta_mat_info = next(
        (m for m in self.indexes_matrices.current_matrices_variables if "betas" in m["name"]),
        None,
    )
    if beta_mat_info is None:
        logger_mosek.warning("save_beta_values_fusion: no matrix with betas found.")
        return

    dim = beta_mat_info["dim"]
    try:
        psd_var = self.model.getVariable(beta_mat_info["name"])
        X = np.array(psd_var.level()).reshape((dim, dim))
    except Exception as e:
        logger_mosek.warning("save_beta_values_fusion : impossible de lire la matrice betas : %s", e)
        return

    lines = []
    for class_label in self.indexes_matrices.ytargets:
        if class_label == self.indexes_matrices.ytrue:
            continue
        try:
            idx = self.indexes_matrices.index_variable_beta(class_label)
            beta_val = float(X[0, idx])
            lines.append(f"beta_{class_label} = {beta_val:.8f}")
        except Exception as e:
            lines.append(f"beta_{class_label} = ERROR ({e})")

    out_path = os.path.join(model_dir, f"betas_{cuts_str}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
