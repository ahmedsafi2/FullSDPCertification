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
    print("Optimal solution found with objective value: ", self.primal_obj_value)
    self.dual_obj_value = self.model.dualObjValue()
    logger_mosek.info("Dual objective value: %s", self.dual_obj_value)
    print("PRIMAL-DUAL : primal = ", self.primal_obj_value, " dual = ", self.dual_obj_value)
    self.optimal_value = self.primal_obj_value  # Pas de constante hors du model ici
    print(
        f"Optimal value (no added constant, already written in model): {self.optimal_value} : with cuts {cuts}"
    )
    self.is_robust = self.optimal_value >= 0

    gap = abs(self.dual_obj_value - self.primal_obj_value)
    if gap > 0.01 * max(1.0, abs(self.dual_obj_value)):
        print(f"⚠ Gap primal-dual persistant ({gap:.3e}) → déclenchement du diagnostic Slater")
        logger_mosek.warning("Gap primal-dual persistant (%.3e) → diagnostic Slater", gap)
        try:
            self.diagnose_infeasibility()
        except Exception as e:
            logger_mosek.warning("Diagnostic Slater échoué : %s", e)

    # self.compute_solutions(cuts, verbose)
    if self.indexes_matrices.BETAS_Z:
        self.save_beta_values_fusion(cuts)
    dic_sol = {"optimal_value": self.optimal_value}
    dic_sol.update({"primal_obj_value": self.primal_obj_value})
    dic_sol.update({"dual_obj_value": self.dual_obj_value})
    return dic_sol


def is_status_optimal(self):
    """
    Check if the status of the solver is optimal.

    Returns
    -------
    bool
        True if the status is optimal, False otherwise.
    """
    return self.model.getPrimalSolutionStatus() == mosek.fusion.SolutionStatus.Optimal


def is_status_infeasible(self):
    """
    Check if the status of the solver is infeasible.

    Returns
    -------
    bool
        True if the status is infeasible, False otherwise.
    """
    return (
        self.model.getProblemStatus() == mosek.fusion.ProblemStatus.PrimalInfeasible
        or self.model.getProblemStatus() == mosek.fusion.ProblemStatus.DualInfeasible
    )


def is_status_unknown(self):
    """
    Check if the status of the solver is unknown.

    Returns
    -------
    bool
        True if the status is unknown, False otherwise.
    """
    return self.model.getPrimalSolutionStatus() == mosek.fusion.SolutionStatus.Unknown


def save_beta_values_fusion(self, cuts: List):
    """
    Extrait les valeurs β_j depuis le modèle Fusion et les sauvegarde dans betas_{cuts_str}.txt.

    Équivalent de save_beta_values (API classic) mais utilise model.getVariable().level()
    au lieu de task.getbarxj(), car l'API Fusion n'expose pas directement les barx.

    La valeur de β_j est X[0, index_variable_beta(j)] dans la dernière matrice PSD
    (ligne 0 = variable constante 1, donc X[0, i] = β_j · 1 = β_j).
    """
    cuts_str = compute_cuts_str(cuts)
    model_dir = get_project_path(f"{self.folder_name}/{self.name}")
    os.makedirs(model_dir, exist_ok=True)

    beta_mat_info = next(
        (m for m in self.indexes_matrices.current_matrices_variables if "betas" in m["name"]),
        None,
    )
    if beta_mat_info is None:
        logger_mosek.warning("save_beta_values_fusion : aucune matrice avec betas trouvée.")
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
    print(f"CALLBACK : Valeurs beta sauvegardées dans {out_path}")
