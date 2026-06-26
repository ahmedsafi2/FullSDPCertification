import numba
from numba.typed import Dict
import numpy as np


big_M_cst = 13


@numba.njit
def _get_key_quad_(i, j, num_matrix, n_index, M: int = big_M_cst):
    key = (i + 1) * n_index * M + (j + 1) * M + num_matrix

    return (i + 1) * n_index * M + (j + 1) * M + num_matrix


@numba.njit
def _get_quad_indices_from_key(index, n_index, M: int = big_M_cst):
    i = (index // (n_index * M)) - 1
    j = ((index // M) % n_index) - 1
    num_matrix = index % M

    return i, j, num_matrix


@numba.njit
def _get_key_linear_(i, num_matrix, M: int = big_M_cst):

    return (i + 1) * M + num_matrix


@numba.njit
def _get_linear_indices_from_key(key: int, M: int = big_M_cst):

    i = (key // M) - 1
    num_matrix = key % M
    return i, num_matrix


@numba.njit
def _get_key_from_layer_neuron_(layer: int, neuron: int, M: int = big_M_cst):

    return (neuron + 1) * M + layer


@numba.njit
def _get_layer_neuron_from_key_(key: int, M: int = big_M_cst):

    layer = key % M
    neuron = (key // M) - 1
    return layer, neuron


# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************


@numba.njit
def _add__co(i, j, num_matrix, value, n_index, elements):
    key = _get_key_quad_(i, j, num_matrix, n_index)
    if key in elements:
        elements[key] += value
    else:
        elements[key] = value


@numba.njit
def _get__co(index, elements):
    if index in elements:
        return elements[index]
    else:
        return 0.0


@numba.njit
def _decode_elements_numba_co(elements_dict, n_index):
    n = len(elements_dict)
    i_arr = np.empty(n, dtype=np.int32)
    j_arr = np.empty(n, dtype=np.int32)
    num_matrix_arr = np.empty(n, dtype=np.int32)
    val_arr = np.empty(n, dtype=np.float64)

    idx = 0
    for key, value in elements_dict.items():
        i, j, num_matrix = _get_quad_indices_from_key(
            key,
            n_index=n_index,
        )
        i_arr[idx] = i
        j_arr[idx] = j
        num_matrix_arr[idx] = num_matrix
        val_arr[idx] = value
        idx += 1

    return i_arr, j_arr, num_matrix_arr, val_arr


class ElementsinConstraintsObjectives:
    """
    Class to handle a variable in a constraint or objective.
    """
    def __init__(self, n_index: int):
        self.n_index = n_index
        self.elements = Dict.empty(
            key_type=numba.types.int64, value_type=numba.types.float64
        )

    def get_key(self, i, j, num_matrix):
        return _get_key_quad_(i, j, num_matrix, self.n_index)

    def get_i_j_num_matrix_from_key(self, index):
        return _get_quad_indices_from_key(index, self.n_index)

    def add(self, i, j, num_matrix, value):
        _add__co(i, j, num_matrix, value, self.n_index, self.elements)

    def get(self, key):
        return _get__co(key, self.elements)

    def decode_key_vec(self):
        return _decode_elements_numba_co(
            self.elements,
            n_index=self.n_index,
        )


# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************


@numba.njit
def _add_ni(
    i: int,
    num_matrix: int,
    value: float,
    equivalent_neurons_substract: numba.typed.Dict,
    M: int = big_M_cst,
):
    index_equivalent = _get_key_linear_(i, num_matrix, M)
    if index_equivalent in equivalent_neurons_substract:
        equivalent_neurons_substract[index_equivalent] += value
    else:
        equivalent_neurons_substract[index_equivalent] = value


class Equivalent_Neurons_Index:
    def __init__(self, K : int, LAST_LAYER : bool):
        self.M = big_M_cst
        self.K = K
        self.LAST_LAYER = LAST_LAYER
        self.equivalent_neurons = {}

    def get_index(self, layer: int, neuron: int):
        return _get_key_from_layer_neuron_(layer=layer, neuron=neuron, M=self.M)

    def create_dict(self, layer: int, neuron: int, K : int, LAST_LAYER : bool, decomposed_in_front_and_back_matrix : bool):
        """
        Create a dictionary for the equivalent neurons.
        """

        key = _get_key_from_layer_neuron_(layer=layer, neuron=neuron, M=self.M)
        
        assert key not in self.equivalent_neurons, f"Index {key} already exists."

        self.equivalent_neurons[key] = {"constant" : 0.0}
        if decomposed_in_front_and_back_matrix :
            if layer > 0:
                self.equivalent_neurons[key]["weights_back"] = Dict.empty(
                        key_type=numba.types.int64, value_type=numba.types.float64
                    )
            if layer <= K - 2 or (LAST_LAYER and layer == K - 1):
                self.equivalent_neurons[key]["weights_front"] = Dict.empty(
                        key_type=numba.types.int64, value_type=numba.types.float64
                    )
        else :
            self.equivalent_neurons[key]["weights"] = Dict.empty(
                        key_type=numba.types.int64, value_type=numba.types.float64
                    )
    
    def add_unstable_neuron(self, front_of_matrix : bool, i : int, num_matrix : int, value : float, key : int):
       
        weight_str = "weights_front" if front_of_matrix else "weights_back"
        _add_ni(
            i,
            num_matrix,
            value,
            equivalent_neurons_substract=self.equivalent_neurons[key][weight_str],
            M=self.M,
        )
    
    def add_stable_active_neuron(self, i : int, num_matrix : int, value : float, key : int):
        _add_ni(
            i,
            num_matrix,
            value,
            equivalent_neurons_substract=self.equivalent_neurons[key]["weights"],
            M=self.M,
        )

    def add(
        self,
        layer: int,
        neuron: int,
        i: int,
        num_matrix: int,
        value: float,
        **kwargs,
    ):
        key = _get_key_from_layer_neuron_(layer=layer, neuron=neuron, M=self.M)
        assert key in self.equivalent_neurons, f"Index {key} does not exist."
        
        front_of_matrix = kwargs.get("front_of_matrix", None)
        if front_of_matrix is not None : 
            self.add_unstable_neuron(front_of_matrix,i,num_matrix,value,key)
        else:
            self.add_stable_active_neuron(i,num_matrix,value,key)

        

    def add_constant(self, layer: int, neuron: int, value: float):

        index = self.get_index(layer, neuron)
        assert index in self.equivalent_neurons, f"Index {index} does not exist."
        self.equivalent_neurons[index]["constant"] += value

    def get_constant(self, layer: int, neuron: int):

        index = self.get_index(layer, neuron)
        assert index in self.equivalent_neurons, f"Index {index} does not exist."
        return self.equivalent_neurons[index]["constant"]

    def get_equivalent(self, layer: int, neuron: int, front_of_matrix: bool = None, decomposed_in_front_and_back_matrix : bool = False):

       
        index = self.get_index(layer, neuron)
        assert index in self.equivalent_neurons, f"Index {index} does not exist."
        if not decomposed_in_front_and_back_matrix : 
            return self.equivalent_neurons[index]["weights"]
        if front_of_matrix is None :
            raise ValueError(f"front_of_matrix must be specified for neuron {neuron} at layer {layer}")
        max_front_layer = self.K if self.LAST_LAYER else self.K - 1
        if not ( (not front_of_matrix and layer > 0) or (front_of_matrix and layer < max_front_layer) ):
            pass
        weights_str = "weights_front" if front_of_matrix else "weights_back"
        return self.equivalent_neurons[index][weights_str]


# ********************************************************************************************************************************
# ********************************************************************************************************************************


def _add_from_key(class_label: int, i: int, num_matrix: int, dict: numba.typed.Dict):

    index = _get_key_linear_(i, num_matrix, big_M_cst)
    if index in dict:
        raise ValueError(f"Class label {class_label} already exists in the dictionary index = {index}.")
    else:
        dict[index] = 1


class Equivalent_Betas_Index:


    def __init__(self, ytargets: list = None):

        self.equivalent_indexes_betas = {
            target: numba.typed.Dict.empty(
                key_type=numba.types.int64,
                value_type=numba.types.float64,
            )
            for target in ytargets
        }

    def add(self, class_label: int, i: int, num_matrix: int):

        assert (
            class_label in self.equivalent_indexes_betas
        ), f"Class label {class_label} not found in equivalent betas."
        _add_from_key(
            class_label,
            i=i,
            num_matrix=num_matrix,
            dict=self.equivalent_indexes_betas[class_label],
        )

    def get_equivalent(self, class_label: int):

        if class_label in self.equivalent_indexes_betas:
            return self.equivalent_indexes_betas[class_label]
        else:
            raise KeyError(f"Class label {class_label} not found in equivalent betas.")


# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************


@numba.njit
def add_dict_linear_to_elements(
    elements: numba.typed.Dict, dict: numba.typed.Dict, value: float, n_index: int, dividing_non_diag: bool = True
):
    for key in dict.keys():
        i, num_matrix = _get_linear_indices_from_key(key)
        key_in_element = _get_key_quad_(
            i=i,
            j=0,
            num_matrix=num_matrix,
            n_index=n_index,
        )

        if dividing_non_diag and i != 0:
            value_ = value/2
        else:
            value_ = value
        if value_ == 0:
            continue
        if key_in_element in elements:
            elements[key_in_element] += dict[key] * value_
        else:
            elements[key_in_element] = dict[key] * value_


@numba.njit
def add_dict_quad_to_elements(
    elements: numba.typed.Dict,
    dict1: numba.typed.Dict,
    dict2: numba.typed.Dict,
    value: float,
    n_index: int,
    dividing_non_diag: bool = True,
):
    for key1 in dict1.keys():
        i1, num_matrix1 = _get_linear_indices_from_key(key1)

        for key2 in dict2.keys():
            i2, num_matrix2 = _get_linear_indices_from_key(key2)
            assert (
                num_matrix1 == num_matrix2
            ), f"Matrix indices do not match: {num_matrix1} != {num_matrix2}"
            if i1 != i2 and dividing_non_diag:
                value_ = value/2 
            else:
                value_ = value 

            if value_ == 0:
                continue

            if i2 > i1:
                i1, i2 = i2, i1
            index_element = _get_key_quad_(i1, i2, num_matrix1, n_index=n_index)
            if index_element in elements:
                elements[index_element] += dict1[key1] * dict2[key2] * value_
            else:
                elements[index_element] = dict1[key1] * dict2[key2] * value_
