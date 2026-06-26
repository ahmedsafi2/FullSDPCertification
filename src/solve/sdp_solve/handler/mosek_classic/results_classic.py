from typing import List
import mosek
import logging
import numpy as np

from fastsdp_tools import get_project_path
from solve.sdp_solve.get_variables import print_solution_to_file_for_cb_solver


logger_mosek = logging.getLogger("Mosek_logger")


def add_all_infos_optimal_values_to_dic(self, cuts: List):
    """
    Add all the information about the optimal values found to the dictionnary for the benchmark.
    """
    self.primal_obj_value = self.task.getprimalobj(mosek.soltype.itr)
    logger_mosek.debug(
        "Optimal solution found with objective value: %s", self.primal_obj_value
    )    
    self.dual_obj_value = self.task.getdualobj(mosek.soltype.itr)
    if self.verbose : 
        pass
    logger_mosek.info("Dual objective value: %s", self.dual_obj_value)
    if self.verbose :
        pass
    self.optimal_value = self.primal_obj_value + self.Objective.constant
    self.is_robust = self.optimal_value >= 0
    if self.verbose :
        pass

    gap = abs(self.dual_obj_value - self.primal_obj_value)
    if gap > 0.01 * max(1.0, abs(self.dual_obj_value)):
        try:
            self.diagnose_infeasibility()
        except Exception as e:
            logger_mosek.warning("Slater diagnostic failed : %s", e)

    self.compute_solutions(cuts, print_sol = False)
    dic_sol = {"optimal_value": self.optimal_value}
    dic_sol.update({"primal_obj_value": self.primal_obj_value})
    dic_sol.update({"dual_obj_value": self.dual_obj_value})
    return dic_sol


@staticmethod
def reconstruct_matrix(size, tab_triang):
    """
            Reconstruct the symmetric matrix
    from the values of the lower triangular part given in a one-dimensional array.

    Args:
    size (int): dimension of the square matrix
    tab_triang (list): array of values from the lower triangular part of the matrix
    """
    mat = np.zeros((size, size))
    tri_indices = np.triu_indices(size)
    mat[tri_indices] = tab_triang
    mat = mat + mat.T - np.diag(mat.diagonal())
    return mat


def is_status_optimal(self):

    return self.status == mosek.solsta.optimal


def is_status_infeasible(self):

    return (
        self.status == mosek.solsta.dual_infeas_cer
        or self.status == mosek.solsta.prim_infeas_cer
    )


def is_status_unknown(self):

    return self.status == mosek.solsta.unknown
