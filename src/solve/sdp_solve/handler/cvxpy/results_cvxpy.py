import logging
import numpy as np

logger_mosek = logging.getLogger("Mosek_logger")

_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
_INFEASIBLE_STATUSES = {"infeasible", "infeasible_inaccurate"}


def add_all_infos_optimal_values_to_dic(self, cuts):
    """
    Équivalent CVXPY de add_all_infos_optimal_values_to_dic (results_classic.py).

    self.problem.value contient la valeur de la partie trace de l'objectif (sans la constante).
    La valeur certifiée est : primal_obj_value + Objective.constant.
    """
    raw = self.problem.value if self.problem is not None else None

    if raw is None or np.isinf(raw) or np.isnan(raw):
        self.primal_obj_value = None
        self.dual_obj_value = None
        self.optimal_value = None
        self.is_robust = False
        logger_mosek.warning(
            "CVXPY : problème mal résolu — status=%s, value=%s",
            self.status_cp,
            raw,
        )
    else:
        self.primal_obj_value = float(raw)
        self.dual_obj_value = None
        self.optimal_value = self.primal_obj_value + self.Objective.constant
        self.is_robust = self.optimal_value >= 0
        print(
            f"CALLBACK CVXPY : Optimal value (with constant) : {self.optimal_value}"
            f" — status={self.status_cp}"
            f", cuts={cuts}"
        )
        logger_mosek.info("CVXPY optimal value (with constant) : %s", self.optimal_value)

    self.compute_solutions(cuts, print_sol=False)

    return {
        "optimal_value": self.optimal_value,
        "primal_obj_value": self.primal_obj_value,
        "dual_obj_value": self.dual_obj_value,
    }


def is_status_optimal(self):
    return self.status_cp in _OPTIMAL_STATUSES


def is_status_infeasible(self):
    return self.status_cp in _INFEASIBLE_STATUSES


def is_status_unknown(self):
    return self.status_cp not in _OPTIMAL_STATUSES and self.status_cp not in _INFEASIBLE_STATUSES
