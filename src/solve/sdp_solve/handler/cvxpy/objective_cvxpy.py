import numpy as np
import logging
from collections import defaultdict
from typing import List
import cvxpy as cp
import numba
from numba.typed import Dict

from ..indexes_matrices import Indexes_Matrixes_for_Mosek_Solver
from ..indexes_variables import Indexes_Variables_for_Mosek_Solver
from ..objective import Objective
from ..variable_elements import add_dict_linear_to_elements, add_dict_quad_to_elements

logger_mosek = logging.getLogger("Mosek_logger")


class ObjectiveCvxpy(Objective):

    def __init__(
        self,
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver,
        indexes_variables: Indexes_Variables_for_Mosek_Solver,
        **kwargs,
    ):
        super().__init__(indexes_matrices, indexes_variables, **kwargs)

    def add_var(self, dict1, value, dict2=None):
        if dict2 is None:
            add_dict_linear_to_elements(
                elements=self.elements.elements,
                dict=dict1,
                value=value,
                nb_index=self.indexes_variables.max_index,
                dividing_non_diag=True,
            )
        else:
            add_dict_quad_to_elements(
                elements=self.elements.elements,
                dict1=dict1,
                dict2=dict2,
                value=value,
                nb_index=self.indexes_variables.max_index,
                dividing_non_diag=True,
            )

    def to_cvxpy_expr(self, cp_vars: List[cp.Variable]) -> cp.Expression:
        """
        Construit l'expression CVXPY Σ_m trace(C_m @ P_m) représentant l'objectif.

        La constante self.constant n'est PAS incluse ici.
        Elle doit être ajoutée après résolution :
            optimal_value = problem.value + self.constant
        """
        self.format_obj()

        grouped = defaultdict(list)
        for m, i, j, v in zip(self.num_matrix, self.i, self.j, self.value):
            grouped[int(m)].append((int(i), int(j), float(v)))

        expr = 0.0
        for m, terms in grouped.items():
            P = cp_vars[m]
            dim = P.shape[0]
            C = np.zeros((dim, dim))
            for i, j, v in terms:
                C[i, j] += v
                if i != j:
                    C[j, i] += v
            expr = expr + cp.trace(C @ P)

        return expr

    def add_task(self, task):
        pass

    def add_to_task(self):
        pass
