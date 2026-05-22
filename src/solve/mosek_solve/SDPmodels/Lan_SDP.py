import numpy as np
import mosek
import yaml
import os
import sys

from tools.utils import infinity, add_functions_to_class
import logging
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from ..mosek_generic_solver import MosekSolver
from networks import ReLUNN
from .certification_problem_objective import objective_Lan
from .certification_problem_constraints_bounds import (
    L2_ball_bounds,
    quad_bounds,
    McCormick_inter_layers,
    all_Mc_Cormick_all_layers,
    all_4_McCormick,
    is_front_of_matrix,
)
from .certification_problem_constraints_forward_pass import (
    ReLU_constraint_Lan,
    ReLU_constraint_stable_active_relaxation,
    ReLU_triangularization,
    last_layer_linear_equality
)
from .certification_problem_constraints_rlt import add_RLT_constraints
from .certification_problem_constraints_division_by_layers import matrix_by_layers_rec
from .certification_problem_constraints_sdp import first_term_equal_zero


logger_mosek = logging.getLogger("Mosek_logger")


@add_functions_to_class(
    objective_Lan,
    ReLU_constraint_Lan,
    ReLU_constraint_stable_active_relaxation,
    quad_bounds,
    ReLU_triangularization,
    add_RLT_constraints,
    McCormick_inter_layers,
    matrix_by_layers_rec,
    first_term_equal_zero,
    all_Mc_Cormick_all_layers,
    all_4_McCormick,
    is_front_of_matrix,
    L2_ball_bounds,
    last_layer_linear_equality
)
class LanSDP(MosekSolver):
    def __init__(self, **kwargs):
        # print("kwargs in LanSDP: ", kwargs)
        super().__init__(certification_model_name="LanSDP", **kwargs)

        print("STUDY : beginning LanSDP init")
        self.BETAS = False
        self.BETAS_Z = False

        self.possible_targets = [
            target for target in self.ytargets if target != self.ytrue
        ]
        if "ytarget" in kwargs:
            self.ytarget = kwargs["ytarget"]
        elif not self.is_trivially_solved:
            self.ytarget = np.random.choice(self.possible_targets)

        print("Neurones stables actives: ", self.stable_actives_neurons)
        print("Neurones stables inactives: ", self.stable_inactives_neurons)

        logger_mosek.debug(f"Bounds for the network :  {self.L} and {self.U}")
        print("STUDY : ending LanSDP init")

    def add_objective(self):
        """
        Add the objective to the Objective class.
        """
        self.objective_Lan()

    def add_constraints(self, cuts: List = []):
        """
        Add constraints to the task.
        """

        # print("L before adding constraints  :   ", self.L)
        # print("U before adding constraints  :   ", self.U)
        # RELU

        self.ReLU_constraint_Lan()

        self.ReLU_triangularization()

        # BOUNDS
        self.quad_bounds()
        if self.norm == "L2" and self.INPUT_IN_VARIABLES and len(self.pruned_input_neurons) == 0:
            self.L2_ball_bounds()

        # CUTS
        if "RLT" in cuts:
            self.add_RLT_constraints(p=self.RLT_prop)

        if "allMC" in cuts:
            self.all_Mc_Cormick_all_layers()

        if self.MATRIX_BY_LAYERS:
            self.matrix_by_layers_rec(only_linear_constraints=True)

        if self.LAST_LAYER:
            self.last_layer_linear_equality()

        self.first_term_equal_zero()

        self.handler.Constraints.end_constraints()
