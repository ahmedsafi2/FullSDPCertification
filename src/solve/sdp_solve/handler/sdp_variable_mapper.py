import numpy as np
import random

from .indexes_matrices import (
    Indexes_Matrixes_for_Mosek_Solver,
)
from .indexes_variables import (
    Indexes_Variables_for_Mosek_Solver,
)
from .neuron_linearization import NeuronLinearization
from .variable_elements import (
    Equivalent_Neurons_Index,
    Equivalent_Betas_Index,
    _get_linear_indices_from_key,
    _get_key_from_layer_neuron_,
    _get_layer_neuron_from_key_,
    _get_key_linear_,
)
import logging
from typing import List

from numba import njit
import numba
from numba.typed import Dict

logger_mosek = logging.getLogger("Mosek_logger")

from fastsdp_tools import summing_values_two_dicts, change_to_zero_negative_values


def get_only_one_variable_kwargs(index: int = 1, **kwargs):
    assert index in [1, 2], "Index must be either 1 or 2."
    return {
        k.strip(f"{index}"): v for k, v in kwargs.items() if str(k).endswith(f"{index}")
    }


class  SDPVariableMapper :
    """
    A class to handle the variables call in the MOSEK solver.
    """
    def __init__(
        self,
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver,
        indexes_variables: Indexes_Variables_for_Mosek_Solver,
        **kwargs,
    ):
        """
        Initialize the  SDPVariableMapper  class.
        """
        self.indexes_matrices = indexes_matrices
        self.indexes_variables = indexes_variables

        self.stable_inactives_neurons = kwargs.pop("stable_inactives_neurons")
        self.stable_actives_neurons = kwargs.pop("stable_actives_neurons")
        self.ytargets = kwargs.get("ytargets")

        self.K = kwargs.pop("K")
        self.n = kwargs.pop("n")
        self.W = kwargs.pop("W")
        self.b = kwargs.pop("b")

        self.L = kwargs.get("L", None)
        self.U = kwargs.get("U", None)


        self.layers_values = NeuronLinearization (
            K=self.K,
            n=self.n,
            W=self.W,
            b=self.b,
            stable_actives_neurons=self.stable_actives_neurons,
            stable_inactives_neurons=self.stable_inactives_neurons,
            **kwargs,
        )

        self.L, self.U = self.layers_values.computing_bounds_based_on_stable_neurons(
            L=self.L, U=self.U
        )
        self.U_above_zero = change_to_zero_negative_values(
            self.U, dim=2
        ) 
        self.L_above_zero = change_to_zero_negative_values(
            self.L, dim=2
        ) 

        self.LAST_LAYER = kwargs.get("LAST_LAYER", None)
        self.MATRIX_BY_LAYERS = kwargs.get("MATRIX_BY_LAYERS", None)
        self.BETAS = kwargs.get("BETAS", None)
        self.INPUT_IN_VARIABLES = kwargs.get("INPUT_IN_VARIABLES", True)
        self.kept_input_neurons = kwargs.get("kept_input_neurons", set(range(int(self.n[0]))))
        self.pruned_input_neurons = kwargs.get("pruned_input_neurons", set())
        assert self.LAST_LAYER is not None, "LAST_LAYER must be specified."
        assert self.BETAS is not None, "BETAS must be specified."

        self.equivalent_neurons = Equivalent_Neurons_Index(K = self.K, LAST_LAYER=self.LAST_LAYER)
        self.equivalent_indexes_betas = Equivalent_Betas_Index(ytargets=self.ytargets)

        self.create_equivalent_indexes_matrices()

        self.study_indexes_equivalent_neurons() 

       

    def create_equivalent_indexes_matrices(self):

        for layer in range(self.K + 1):
            if layer == 0 and not self.INPUT_IN_VARIABLES:
                continue  
            for neuron in range(self.n[layer]):
                if layer == 0 and neuron in self.pruned_input_neurons:
                    continue  
                
                equivalent_values_neurons, constant = (
                    self.layers_values.get_equivalent_values(layer, neuron)
                )

                is_boundary_layer = len(self.indexes_matrices._layer_to_groups.get(layer, [])) > 1
                decomposed_in_front_and_back_matrix = ((not ((layer,neuron) in self.stable_actives_neurons)) and (layer < self.K or self.LAST_LAYER)) and self.MATRIX_BY_LAYERS and is_boundary_layer
                
                self.equivalent_neurons.create_dict(layer=layer, neuron=neuron, K = self.K, 
                                                    LAST_LAYER=self.LAST_LAYER, 
                                                    decomposed_in_front_and_back_matrix=decomposed_in_front_and_back_matrix)

                if decomposed_in_front_and_back_matrix:
                    
                    for (k, j), val in equivalent_values_neurons.items():
                        
                        k_groups = self.indexes_matrices._layer_to_groups.get(k, [])
                        k_can_be_front = any(
                            pos < len(self.indexes_matrices.layer_groups[g]) - 1
                            for g, pos in k_groups
                        )
                        k_can_be_back = any(
                            pos == len(self.indexes_matrices.layer_groups[g]) - 1
                            for g, pos in k_groups
                        )

                        if k_can_be_front:
                                
                                ind_i_front = self.indexes_variables._get_variable_index(
                                    "z", layer=k, neuron=j, front_of_matrix=True
                                )
                                ind_num_matrix_front = self.indexes_matrices._get_matrix_index(
                                    "z", layer=k, neuron=j, front_of_matrix=True
                                )
                                self.equivalent_neurons.add(
                                    layer=layer,
                                    neuron=neuron,
                                    i=ind_i_front,
                                    num_matrix=ind_num_matrix_front,
                                    value=1,
                                    front_of_matrix=True,
                                )
                        if k_can_be_back:
                            
                            ind_i_back = self.indexes_variables._get_variable_index(
                                "z", layer=k, neuron=j, front_of_matrix=False
                            )
                            ind_num_matrix_back = self.indexes_matrices._get_matrix_index(
                                "z", layer=k, neuron=j, front_of_matrix=False
                            )
                            self.equivalent_neurons.add(
                                layer=layer,
                                neuron=neuron,
                                i=ind_i_back,
                                num_matrix=ind_num_matrix_back,
                                value=1,
                                front_of_matrix=False,
                            )

                else : 
                    for (k, j), val in equivalent_values_neurons.items():

                        if (k < self.K - 1 and not self.LAST_LAYER) or (
                            k < self.K and self.LAST_LAYER
                        ):
                            ind_i_front = self.indexes_variables._get_variable_index(
                                "z", layer=k, neuron=j, front_of_matrix=True
                            )
                            ind_num_matrix_front = self.indexes_matrices._get_matrix_index(
                                "z", layer=k, neuron=j, front_of_matrix=True
                            )
                            self.equivalent_neurons.add(
                                layer=layer,
                                neuron=neuron,
                                i=ind_i_front,
                                num_matrix=ind_num_matrix_front,
                                value=val,
                            )
                        else:
                            ind_i_back = self.indexes_variables._get_variable_index(
                                "z", layer=k, neuron=j, front_of_matrix=False
                            )
                            ind_num_matrix_back = self.indexes_matrices._get_matrix_index(
                                "z", layer=k, neuron=j, front_of_matrix=False
                            )
                            self.equivalent_neurons.add(
                                layer=layer,
                                neuron=neuron,
                                i=ind_i_back,
                                num_matrix=ind_num_matrix_back,
                                value=val,
                            )



                self.equivalent_neurons.add_constant(
                    layer=layer,
                    neuron=neuron,
                   
                     value=constant,
                )

        if self.BETAS:

            for class_label in self.ytargets:

                i = self.indexes_variables._get_variable_index(
                    "beta", class_label=class_label
                )
                num_matrix = self.indexes_matrices._get_matrix_index(
                    "beta", class_label=class_label
                )
                self.equivalent_indexes_betas.add(
                    class_label=class_label,
                    i=i,
                    num_matrix=num_matrix,
                )

    def study_indexes_equivalent_neurons(self):
        for k in range(self.K + 1):
            for j in range(self.n[k]):
                key = self.equivalent_neurons.get_index(layer=k, neuron=j)
                layer, neuron = _get_layer_neuron_from_key_(key=key)
                assert (
                    layer == k and neuron == j
                ), f"ERROR in STUDY1: layer = {layer}, neuron = {neuron}, k = {k}, j = {j}"
        for num_matrix in range(self.indexes_matrices.nb_matrices):
            logger_mosek.debug(f"in study_indexes_equivalent_neurons: num_matrix = {num_matrix}")
            try:
                for i in range(self.indexes_variables.max_index):

                    key = _get_key_linear_(i, num_matrix)

                    i2, num_matrix2 = _get_linear_indices_from_key(key)

                    assert (
                        i == i2 and num_matrix == num_matrix2
                    ), f"ERROR in STUDY2: i = {i}, num_matrix = {num_matrix}, i2 = {i2}, num_matrix2 = {num_matrix2}"

            except ValueError as e:
                print("Error : ", e)
                pass

    def _print_equivalent_indexes_(self):
        line = ""

        for layer in range(self.K + 1):
            line += f"Layer {layer}:\n"
            for neuron in range(self.n[layer]):
                
                decomposed_in_front_and_back_matrix = (not ((layer,neuron) in self.stable_actives_neurons)) and (layer < self.K or self.LAST_LAYER) and self.MATRIX_BY_LAYERS

                constant = self.equivalent_neurons.get_constant(
                    layer=layer, neuron=neuron
                )
                line += f"\n  Layer {layer} Neuron {neuron}: \n"

                if ((layer <= self.K - 1 and not self.LAST_LAYER) or (
                      self.LAST_LAYER
                    )) and decomposed_in_front_and_back_matrix:
                    front = self.equivalent_neurons.get_equivalent(
                        layer=layer, neuron=neuron, front_of_matrix=True
                    )
                    line += f"    FRONT_OF_MATRIX \n"
                    for key, value in front.items():
                        i, num_matrix = _get_linear_indices_from_key(key, 13)
                        line += f"  {(num_matrix,i)} : {value};   "
                if layer > 0 and decomposed_in_front_and_back_matrix: 
                    line += f"\n    BACK_OF_MATRIX \n"
                    back = self.equivalent_neurons.get_equivalent(
                        layer=layer, neuron=neuron, front_of_matrix=False
                    )
                    for key, value in back.items():
                        i, num_matrix = _get_linear_indices_from_key(key, 13)
                        line += f"{(num_matrix,i)} : {value};   "
                elif not decomposed_in_front_and_back_matrix:
                    eq = self.equivalent_neurons.get_equivalent(
                        layer=layer, neuron=neuron, 
                    )
                    for key, value in eq.items():
                        i, num_matrix = _get_linear_indices_from_key(key, 13)
                        line += f"{(num_matrix,i)} : {value};   "
                line += f"\n    constant : {constant}\n"
        if self.BETAS:
            for class_label in self.ytargets:
                line += f"Class {class_label}:"
                dict_beta = self.equivalent_indexes_betas.get_equivalent(class_label)
                for key, value in dict_beta.items():
                    i, num_matrix = _get_linear_indices_from_key(key, 13)
                    line += f"  i : {i} ; "
                    line += f"  num_matrix : {num_matrix}\n"
        print(line)

        ""

    def add_constant(self, value: float):
        """
        Add a constant to the constraint.
        """
        raise NotImplementedError("This method should be implemented in the subclass COnstraint or Objective.")

    def add_var(self, **kwargs):
        raise NotImplementedError("This method should be implemented in the subclass Constraint or Objective.")

  

    def verify_variable_z(self, layer : int, neuron : int, front_of_matrix : bool= None):
        assert layer is not None, "Layer must be specified for z variable."
        assert neuron is not None, "Neuron must be specified for z variable."
        if layer == 0 and (not self.INPUT_IN_VARIABLES or neuron in self.pruned_input_neurons):
            raise ValueError(
                f"Cannot reference z_(0,{neuron}) as a SDP variable: "
                "this input neuron was removed from the SDP (INPUT_IN_VARIABLES=False or partial pruning). "
                "Check that no constraint explicitly uses this layer-0 variable."
            )
        assert (layer, neuron) not in self.stable_inactives_neurons, "Stable inactive neurons should not be added as variables."
        if (layer, neuron) in self.stable_actives_neurons :
            decomposed_in_front_and_back_matrix = False
        else : 
            if self.MATRIX_BY_LAYERS :
                if front_of_matrix is None:
                    if layer < self.K-1 :
                        front_of_matrix = True
                    elif layer == self.K-1 and self.LAST_LAYER :
                        front_of_matrix = True
                    else :
                        front_of_matrix = False
                is_boundary_layer = len(self.indexes_matrices._layer_to_groups.get(layer, [])) > 1
                decomposed_in_front_and_back_matrix = (layer < self.K or self.LAST_LAYER) and is_boundary_layer
            else : 
                decomposed_in_front_and_back_matrix = False
        return decomposed_in_front_and_back_matrix, front_of_matrix


              

    def add_linear_variable(self, var: str, value: float, **kwargs):
        """
        Add a linear variable to the constraint.
        Checks if the variable present corresponds to stable active neurons and divides it in this case to the precedent layer's z variables.
        """
        if value == 0:
            return
        if var == "z":
            layer = kwargs.get("layer", None)
            neuron = kwargs.get("neuron", None)
            front_of_matrix = kwargs.get("front_of_matrix", None)
            #print("STUDY COEFF layer : ", layer, "neuron : ", neuron, "front_of_matrix : ", front_of_matrix)

            decomposed_in_front_and_back_matrix, front_of_matrix = self.verify_variable_z(layer, neuron, front_of_matrix)
            dict1 = self.equivalent_neurons.get_equivalent(
                    layer=layer, neuron=neuron, front_of_matrix=front_of_matrix, decomposed_in_front_and_back_matrix=decomposed_in_front_and_back_matrix
                )
            assert len(dict1) > 0, "Dictionnary used in add_linear_variable is empty"

            self.add_var(
                dict1=dict1,
                value=value,
            )

            constant = self.equivalent_neurons.get_constant(layer, neuron)
            self.add_constant(value * constant)
        else:
            class_label = kwargs.get("class_label", None)
            assert (
                class_label is not None
            ), "Class label must be specified for beta variable."
            self.add_var(
                dict1=self.equivalent_indexes_betas.get_equivalent(
                    class_label=class_label
                ),
                value=value,
            )



    def add_quad_variable(self, var1: str, var2: str, value: float, **kwargs):
        """
        Add a product of two variables to the constraint.
        Checks if the variables present corresponds to stable active neurons and divides them in this case to the precedent layer's z variables.
        """
        if value == 0:
            return
        assert var1 in ["z", "beta"], "var1 must be either 'z' or 'beta'."
        assert var2 in ["z", "beta"], "var2 must be either 'z' or 'beta'."
        if var1 == "z" and var2 == "beta":
            layer1 = kwargs.get("layer1", None)
            neuron1 = kwargs.get("neuron1", None)
            front_of_matrix1 = kwargs.get("front_of_matrix1", None)
            decomposed_in_front_and_back_matrix, front_of_matrix1 = self.verify_variable_z(layer1, neuron1, front_of_matrix1)
            class_label = kwargs.get("class_label", None)
            assert (
                class_label is not None
            ), "Class label must be specified for beta variable."

            dict2 = self.equivalent_indexes_betas.get_equivalent(
                class_label=class_label
            )
            dict1 = self.equivalent_neurons.get_equivalent(
                layer1, neuron1, front_of_matrix1, decomposed_in_front_and_back_matrix
            )
            
            constant1 = self.equivalent_neurons.get_constant(layer1, neuron1)
            constant2 = 0

        elif var1 == "beta" and var2 == "z":
            layer2 = kwargs.get("layer2", None)
            neuron2 = kwargs.get("neuron2", None)
            front_of_matrix2 = kwargs.get("front_of_matrix2", None)
            decomposed_in_front_and_back_matrix, front_of_matrix2 = self.verify_variable_z(layer2, neuron2, front_of_matrix2)
            class_label = kwargs.get("class_label", None)
            assert (
                class_label is not None
            ), "Class label must be specified for beta variable."
           
            dict1 = self.equivalent_indexes_betas.get_equivalent(
                class_label=class_label
            )
            dict2 = self.equivalent_neurons.get_equivalent(
                layer2, neuron2, front_of_matrix2, decomposed_in_front_and_back_matrix
            )

            constant1 = 0
            constant2 = self.equivalent_neurons.get_constant(layer2, neuron2)

        elif var1 == "beta" and var2 == "beta":
            class_label1 = kwargs.get("class_label1", None)
            class_label2 = kwargs.get("class_label2", None)
            assert (
                class_label1 is not None
            ), "Class label1 must be specified for beta variable."
            assert (
                class_label2 is not None
            ), "Class label2 must be specified for beta variable."
            dict1 = self.equivalent_indexes_betas.get_equivalent(
                class_label=class_label1
            )
            dict2 = self.equivalent_indexes_betas.get_equivalent(
                class_label=class_label2
            )
            constant1 = 0
            constant2 = 0

        elif var1 == "z" and var2 == "z":
            layer1 = kwargs.get("layer1", None)
            neuron1 = kwargs.get("neuron1", None)
            front_of_matrix1 = kwargs.get("front_of_matrix1", None)
            layer2 = kwargs.get("layer2", None)
            neuron2 = kwargs.get("neuron2", None)
            front_of_matrix2 = kwargs.get("front_of_matrix2", None)
            
            decomposed_in_front_and_back_matrix1, front_of_matrix1 = self.verify_variable_z(layer1, neuron1, front_of_matrix1)
            decomposed_in_front_and_back_matrix2, front_of_matrix2 = self.verify_variable_z(layer2, neuron2, front_of_matrix2)
            constant1 = self.equivalent_neurons.get_constant(layer1, neuron1)
            constant2 = self.equivalent_neurons.get_constant(layer2, neuron2)
            dict1 = self.equivalent_neurons.get_equivalent(
                layer=layer1, neuron=neuron1, front_of_matrix=front_of_matrix1, decomposed_in_front_and_back_matrix=decomposed_in_front_and_back_matrix1
            )
            dict2 = self.equivalent_neurons.get_equivalent(
                layer=layer2, neuron=neuron2, front_of_matrix=front_of_matrix2, decomposed_in_front_and_back_matrix=decomposed_in_front_and_back_matrix2
            )

        assert len(dict1) > 0, "Dictionnary used in add_quad_variable is empty"
        assert len(dict2) > 0, "Dictionnary used in add_quad_variable is empty"
        self.add_var(
            dict1=dict1,
            value=value,
            dict2=dict2,
        )
        if constant1 != 0 and constant2 != 0:
            self.add_constant(value * constant1 * constant2)
        if constant1 != 0:
            self.add_var(dict1=dict2, value=value * constant1)
        if constant2 != 0:
            self.add_var(dict1=dict1, value=value * constant2)

    def add_z_quad_bound_composed(self, layer1, neuron1, front_of_matrix1,
                                   weight, coeff1, U_next, bound_sense):
        
        v = weight * coeff1
        if (bound_sense == "upper" and v>=0) or (bound_sense == "lower" and v<0):
            # Adding constraint z_next * z_{layer1} <= L_{layer1} * z_next + U_{next} * z_{layer1} - L_{layer1} * U_{next}
            if layer1 == 0: # L_layer1 is zero on all layer outputs (every variable except the one representing the input)
                coeff_next = -v * self.L[layer1][neuron1]
                cst        = v * self.L[layer1][neuron1] * U_next
            else :
                coeff_next = 0
                cst        = 0
           
        else: 
                # Adding constraint z_next * z_{layer1} >= U_{layer1} * z_next + U_next * z_{layer1} - U_{layer1} * U_{layer1}
                coeff_next = -v * self.U[layer1][neuron1]
                cst        = v * self.U[layer1][neuron1] * U_next
        
        self.add_linear_variable(
            var="z",
            layer=layer1,
            neuron=neuron1,
            value=-v * U_next,
            front_of_matrix=front_of_matrix1,
        )

         
        return coeff_next, cst
    
    
    def add_z_quad_bound_one_variable(self, layer1, neuron1, weight, coeff1, bound_sense):
        v = weight * coeff1
        if (bound_sense == "upper" and v>=0) or (bound_sense == "lower" and v<0):
            # Adding constraint z_next * z_{layer1} <= U_{layer1} * z_next
            coeff_next = -v * self.U[layer1][neuron1]
            
        else:  
            if layer1 == 0: # L_layer1 is zero on all layer outputs (every variable except the one representing the input)
                # Adding constraint z_next * z_{layer1} >= L_{layer1} * z_next
                coeff_next = -v * self.L[layer1][neuron1]
            else :
                coeff_next = 0

        return coeff_next, 0

        

    def add_z_quad_active_neuron(
        self,
        layer_prev: int,
        neuron_prev: int,
        layer_next: int,
        neuron_next: int,
        front_of_matrix_prev: bool,
        front_of_matrix_next: bool,
        weight: float,
        bound_sense: str = "upper",
        bound_type: str = "one_variable",
    ):
        assert bound_type in ["one_variable", "composed", "random"]
        assert bound_sense in ["lower", "upper"]

        assert (layer_prev, neuron_prev) in self.stable_actives_neurons
        equivalent_values_neurons, constant = self.layers_values.get_equivalent_values(
            layer_prev, neuron_prev
        )

        U_next = float(self.U[layer_next][neuron_next])
        coeff_next = 0.0  # accumulated coefficient for z_{layer_next}
        cst = 0.0         # accumulated constant

        for (layer1, neuron1), coeff1 in equivalent_values_neurons.items():
            # Check that the quadratic product z_next * z_{layer1} is present in the variable matrices
            groups_layer1 = {g for g, _ in self.indexes_matrices._layer_to_groups.get(layer1, [])}
            groups_layer_next = {g for g, _ in self.indexes_matrices._layer_to_groups.get(layer_next, [])}

            if groups_layer1 & groups_layer_next: # Product present, no bounding needed
                if layer_next - layer1 > 2:
                    logger_mosek.debug(f"RELU : ", f"Product of z_{layer_next} and z_{layer1} is PRESENT in the variable matrices, adding product without bounding.")
                decomposed1, front1 = self.verify_variable_z(layer1, neuron1, None)
                dict_layer1 = self.equivalent_neurons.get_equivalent(layer1, neuron1, front1, decomposed1)
                decomposed_next, front_next = self.verify_variable_z(layer_next, neuron_next, front_of_matrix_next)
                dict_layer_next = self.equivalent_neurons.get_equivalent(layer_next, neuron_next, front_next, decomposed_next)
                self.add_var(dict1=dict_layer1, dict2=dict_layer_next, value=-weight * coeff1)
            else:
                
                if bound_type == "random":
                    bound_type_ = random.choice(["one_variable", "composed"])
                else:
                    bound_type_ = bound_type

                if bound_type_ == "composed":
                    coeff_next_, cst_ = self.add_z_quad_bound_composed(layer1, neuron1, front_of_matrix_prev,
                                                                       weight, coeff1, U_next, bound_sense)
                else:
                    coeff_next_, cst_ = self.add_z_quad_bound_one_variable(layer1, neuron1, weight, coeff1, bound_sense)

                coeff_next += coeff_next_
                cst        += cst_
        
            
        if coeff_next != 0:
            self.add_linear_variable(
                var="z",
                layer=layer_next,
                neuron=neuron_next,
                value=coeff_next,
                front_of_matrix=front_of_matrix_next,
            )
        if cst != 0:
            self.add_constant(cst)

        # Linear term from the constant part of the stable-active decomposition
        if weight * constant != 0:
            self.add_linear_variable(
                var="z",
                layer=layer_next,
                neuron=neuron_next,
                value=-weight * constant,
                front_of_matrix=front_of_matrix_next,
            )


   
   