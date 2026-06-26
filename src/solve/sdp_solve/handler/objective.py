from typing import List
import mosek
import numpy as np
import os
import sys
from .indexes_matrices import (
    Indexes_Matrixes_for_Mosek_Solver,
)
from .indexes_variables import (
    Indexes_Variables_for_Mosek_Solver,
)
from .sdp_variable_mapper import  SDPVariableMapper 
from .variable_elements import ElementsinConstraintsObjectives
from fastsdp_tools import deduplicate_and_sum


import logging
import time


logger_mosek = logging.getLogger("Mosek_logger")


class Objective( SDPVariableMapper ):
    def __init__(
        self,
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver,
        indexes_variables: Indexes_Variables_for_Mosek_Solver,
        **kwargs,
    ):
        super().__init__(
            indexes_matrices=indexes_matrices,
            indexes_variables=indexes_variables,
            **kwargs,
        )

        self.elements = ElementsinConstraintsObjectives(
            self.indexes_variables.max_index,
        )
        self.constant = 0

    def format_obj(self):
        i, j, num_matrix, val = self.elements.decode_key_vec()

        self.i = i
        self.j = j
        self.num_matrix = num_matrix
        self.value = val

    def add_constant(self, value: float):
        self.constant += value

    def add_var(self, **kwargs):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def reinitialize(self, verbose : bool):

        self.elements = ElementsinConstraintsObjectives(
            self.indexes_variables.max_index,
        )

        self.constant = 0
        self.verbose = verbose
