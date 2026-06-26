import numpy as np
from typing import List, Dict
import sys
import os
import mosek
import logging
import time

from ..indexes import Indexes_Mosek_Solver
from .objective_classic import ObjectiveClassic
from .constraints_classic import ConstraintsClassic
from .results_classic import (
    add_all_infos_optimal_values_to_dic,
    is_status_optimal,
    is_status_infeasible,
    is_status_unknown,
    reconstruct_matrix,
)
from .callback_classic import makeUserCallback

from solve.sdp_solve.run_benchmark import compute_cuts_str

from ..common_handler_functions import (
    print_index_variables_matrices,
    num_matrices_variables,
    print_num_variables,
    initialize_variables,
    save_matrix_csv,
    save_matrix_png,
    Matrices_Solutions,
    get_matrices_variables,
    compute_solutions,
    save_beta_values,
    diagnose_infeasibility,
)

from fastsdp_tools.utils import count_calls, add_functions_to_class, get_project_path

logger_mosek = logging.getLogger("Mosek_logger")


@add_functions_to_class(
    initialize_variables,
    reconstruct_matrix,
    save_matrix_csv,
    save_matrix_png,
    add_all_infos_optimal_values_to_dic,
    get_matrices_variables,
    is_status_optimal,
    is_status_infeasible,
    is_status_unknown,
    compute_solutions,
    save_beta_values,
    diagnose_infeasibility,
    print_index_variables_matrices,
    num_matrices_variables,
    print_num_variables,
)
class MosekClassicHandler:
    def __init__(self, **kwargs):
        """
        Initialize the ConstraintHandler class.

        Parameters
        ----------
        n: List[int]
            List of the number of neurons in each layer.
        K: int
            Number of layers.
        matrix_by_layers: bool
            Whether to use matrix by layers or not.
        last_layer: bool
            Whether the last layer is included in the matrix of the z variables or not.
        betas: bool
            Whether to include the beta variables or not.
        betas_z: bool
            Whether to include the beta variables in the matrixes for z variables.
        zbar: bool
            Whether to include the zbar variables or not.
        MATRIX_BY_LAYERS: bool
            Whether to divide matrix variables by layers or not divide them.
        LAST_LAYER: bool
            Whether the last layer is included in the matrix of the z variables or not.
        """
        self.MATRIX_BY_LAYERS = kwargs.get("MATRIX_BY_LAYERS", False)

        self.LAST_LAYER = kwargs.get("LAST_LAYER", False)
        self.BETAS = kwargs.get("BETAS", False)
        self.BETAS_Z = kwargs.get("BETAS_Z", False)
        self.ZBAR = kwargs.get("ZBAR", False)
        self.stable_inactives_neurons = kwargs.get("stable_inactives_neurons", None)
        self.stable_active_neurons = kwargs.get("stable_active_neurons", None)

        self.n = kwargs.get("n", None)
        self.K = kwargs.get("K", None)

        self.folder_name = kwargs.pop("folder_name", None)
        self.name = kwargs.pop("name", None)

        self.epsilon = kwargs.pop("epsilon", None)
        self.solver_time_limit = kwargs.pop("solver_time_limit", None)
        self.rescode = None

        self.ytrue = kwargs.get("ytrue", None)
        self.ytarget = kwargs.get("ytarget", None)



        self.indexes_matrices = Indexes_Mosek_Solver(**kwargs)
        self.indexes_variables = self.indexes_matrices  # merged object (shared reference)

        self.vector_variables = []
        self.final_number_constraints = None

        self.Objective = ObjectiveClassic(
            self.indexes_matrices, self.indexes_variables, **kwargs
        )
        self.Constraints = ConstraintsClassic(
            self.indexes_matrices,
            self.indexes_variables,
            **kwargs,
        )

    def initiate_env(self, verbose: bool = False):
        logger_mosek.info("Initializing MOSEK solver")
        self.verbose = verbose
        if self.verbose:
            pass
        self.env = mosek.Env()
        self.task = self.env.Task(0, 0)
        self.env.__enter__()  
        self.task.__enter__()  

        usercallback = makeUserCallback(maxtime=20000, task=self.task)
        self.task.set_InfoCallback(usercallback)

        self.adjust_solver_parameters()

        self.indexes_matrices.current_matrices_variables = []
        self.vector_variables = []
        self.Objective.add_task(self.task)
        self.Objective.reinitialize(verbose)
        self.Constraints.add_task(self.task)
        self.Constraints.reinitialize(verbose)
        return self 

    def adjust_solver_parameters(self, **parameters):

        self.task.putdouparam(mosek.dparam.intpnt_tol_rel_gap, 1e-3)  
        self.task.putdouparam(mosek.dparam.intpnt_tol_pfeas, 1e-3)    
        self.task.putdouparam(mosek.dparam.intpnt_tol_dfeas, 1e-3)    
        
        self.task.putintparam(mosek.iparam.intpnt_max_iterations, 400)

        num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
        self.task.putintparam(mosek.iparam.num_threads, num_threads)

        if self.solver_time_limit is not None:
            self.task.putdouparam(mosek.dparam.optimizer_max_time, float(self.solver_time_limit))


    @count_calls(
        "init_variables"
    )  
    def add_matrix_variable(self, name: str, dim: int):
        """
        Add a matrix variable of dimension dim to the task.
        """
        logger_mosek.debug(f"Adding a variable matrix {name} of dimension %s", dim)
        if any(
            d["name"] == name for d in self.indexes_matrices.current_matrices_variables
        ):
            logger_mosek.debug(
                f"Variable matrix {name} already exists. Skipping addition."
            )
            return
        else:
            logger_mosek.debug(f"Variable matrix {name} added.")
            self.indexes_matrices.current_matrices_variables.append(
                {"name": name, "dim": dim, "value": Matrices_Solutions()}
            )
            self.task.appendbarvars([dim])

    def add_vector_variable(self, name: str, dim: int):
        """
        Add a vector variable of dimension dim to the task."""
        logger_mosek.info(f"Adding a variable vector {name} of dimension %s", dim)
        self.vector_variables.append(dim)
        self.task.appendvars(dim)

    def initialize_constraints(self):
        logger_mosek.info(
            f"Initializing {self.Constraints.current_num_constraint} constraints"
        )
        self.task.appendcons(self.Constraints.current_num_constraint)
        self.final_number_constraints = self.Constraints.current_num_constraint

    def cleanup_mosek(self):
        logger_mosek.info("Cleaning up MOSEK environment and task \n \n \n")
        if self.task:
            self.task.__exit__(None, None, None) 
            self.task = None
        if self.env:
            self.env.__exit__(None, None, None)  
            self.env = None

    def is_feasible(self, variables_matrices, precision: float = 1e-6) -> bool:
        for constraint in self.Constraints.list_cstr:
            try:
                val = 0
                for index in range(len(constraint["num_matrix"])):
                    num_matrix = constraint["num_matrix"][index]
                    i = constraint["i"][index]
                    j = constraint["j"][index]
                    coeff = constraint["value"][index]
                    val_matrix = variables_matrices[num_matrix][i][j]

                    if i != j:
                        coeff *= 2

                    val += coeff * val_matrix

                lb = constraint["lb"]
                ub = constraint["ub"]

                if val < lb - precision:
                    logger_mosek.debug(
                        f"Constraint {constraint['name']} is not feasible: {val} < {lb}"
                    )
                    return False
                if val > ub + precision:
                    logger_mosek.debug(
                        f"Constraint {constraint['name']} is not feasible: {val} > {ub}"
                    )
                    return False
                else:
                    logger_mosek.debug(
                        f"Constraint {constraint['name']} is feasible: {val} in [{lb}, {ub}]"
                    )
            except Exception as e:
                logger_mosek.error(f"Error in constraint {constraint['name']}: {e}")
        return True

    def value_solution(self, variables_matrices):
        try:
            val = self.Objective.constant
            for index in range(len(self.Objective.list_indexes_matrixes)):
                num_matrix = self.Objective.list_indexes_matrixes[index]
                i = self.Objective.list_indexes_variables_i[index]
                j = self.Objective.list_indexes_variables_j[index]

                coeff = self.Objective.list_values[index]
                val_matrix = variables_matrices[num_matrix][i][j]

                if i != j:
                    coeff *= 2
                val += coeff * val_matrix

            return val
        except Exception as e:
            logger_mosek.error(f"Error in computing the objective: {e}")
            return None

    def define_objective_sense(self):
        self.task.putobjsense(mosek.objsense.minimize)

    def optimize(self):
        logger_mosek.info("Optimizing the task")
        self.rescode = self.task.optimize()

    def is_time_limit(self):
        return self.rescode == mosek.rescode.trm_max_time

    def write_model(
        self,
        cuts: List = [],
        RLT_prop: float = 0.0,
        data_index: int = None,
        ytarget: int = None,
    ):
        """
        Write the results of the optimization to a file.
        """
        logger_mosek.info("Writing results to file...")
        cuts_str = compute_cuts_str(cuts)
        self.task.writedata(
            get_project_path(
                f"{self.folder_name}/{self.name}/{self.name}_{cuts_str}_ind={data_index}_ytarget={ytarget}_RLT={RLT_prop}_classic.ptf"
            )
        )
        logger_mosek.info(
            f"Results written to {get_project_path(f'{self.folder_name}/{self.name}/{self.name}_{cuts_str}_ind={data_index}_ytarget={ytarget}_RLT={RLT_prop}_classic.ptf')}"
        )

    def print_solver_info(self, verbose: bool = False):
        def mosek_to_logger(msg):
            msg = msg.rstrip("\n")
            if msg:
                pass

        if verbose:
            self.task.set_Stream(mosek.streamtype.log, mosek_to_logger)

    def get_solution_status(self):
        self.status = self.task.getsolsta(mosek.soltype.itr)
        return self.status

    def get_num_iterations(self):
        num_iterations = self.task.getintinf(mosek.iinfitem.intpnt_iter)
        return num_iterations

    def get_solution(self, **kwargs):
        ind_solution = kwargs.get("ind_solution", None)
        dim = kwargs.get("dim", None)
        mat = self.task.getbarxj(mosek.soltype.itr, ind_solution)
        return self.reconstruct_matrix(dim, mat)

    def get_dual_variables(self):
        dual_variables = self.task.gety(mosek.soltype.itr)

        prosta = self.task.getprosta(mosek.soltype.itr)
        solsta = self.task.getsolsta(mosek.soltype.itr)


        y = [0.0] * self.final_number_constraints

        assert len(dual_variables) == len(self.Constraints.list_cstr), "Le nombre de variables duales ne correspond pas au nombre de contraintes."
        for i in range(len(dual_variables)):
            self.Constraints.list_cstr[i]["dual_value"] = dual_variables[i]

            name = self.Constraints.list_cstr[i]["name"]

            val1 = dual_variables[i]


        return dual_variables
