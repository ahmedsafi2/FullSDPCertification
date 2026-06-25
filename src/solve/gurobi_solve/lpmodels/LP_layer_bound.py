import logging
import numpy as np

from ..gurobi_generic_solver import GurobiSolver
from ..objective import add_objective_bound_layer
from ..constraints import RELU_triangular_constraint, ball_constraint
from ..variables import add_variable_z, _add_variable_z
from fastsdp_tools.utils import add_functions_to_class
from ..callback import NonConvexQuadraticProgramCallback

logger_gurobi = logging.getLogger("Gurobi_logger")


@add_functions_to_class(
    add_objective_bound_layer,
    RELU_triangular_constraint,
    ball_constraint,
    add_variable_z,
    _add_variable_z,
)
class LPBoundLayer(GurobiSolver):
    """
    A solver that uses Gurobi to solve the optimization problem.
    """

    def __init__(
        self,
        **kwargs,
    ):
        #print("kwargs in init LP : ", kwargs)

        super().__init__(certification_model_type="LP_Bound_Layer", **kwargs)
        assert self.bounds_method == "IBP"
        assert self.use_inactive_neurons and self.use_active_neurons

        print("Initialization... DONE")

    def add_objective(self):
        self.add_objective_bound_layer()

    def add_variables(self):
        """
        Add variables to the model.
        """
        self.add_variable_z()

    def add_constraints(self):
        """
        Add constraints to the model.
        """
        # RELUMIX
        self.RELU_triangular_constraint()

        # BOUNDS
        self.ball_constraint()
        


    def solve(self, verbose : bool = True, **kwargs):
        for layer_obj in range(1,self.K+1):
            self.layer_obj = layer_obj
                    
            self.max_layer_z = self.layer_obj + 1
       
            for neuron_obj in range(self.n[layer_obj]):
                self.neuron_obj = neuron_obj
                for bound_obj in ["lower", "upper"]:
                    print(f"Solving for layer {layer_obj}, neuron {neuron_obj}, bound {bound_obj}...")
                    
                    self.bound_obj = bound_obj
                    self.max_layer_z = self.layer_obj + 1
                    super().run_optimization(verbose=False)
                    if self.opt is None :
                        raise ValueError("Cannot catch the bound computed with this")
                    if bound_obj == "lower" :
                        print(f"BORNES : updating L... with layer {layer_obj}, neuron {neuron_obj}, last value = {self.L[layer_obj][neuron_obj]}, new value = {self.opt}")
                        self.L[layer_obj][neuron_obj] = self.opt
                    elif bound_obj == "upper":
                        print(f"BORNES : updating U... with layer {layer_obj}, neuron {neuron_obj}, last value = {self.U[layer_obj][neuron_obj]}, new value = {self.opt}")
                        self.U[layer_obj][neuron_obj] = self.opt

                  
                    
            
