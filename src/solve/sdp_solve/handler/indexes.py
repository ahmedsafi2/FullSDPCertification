from typing import List, Union

from .variable_elements import big_M_cst


def resolve_layer_groups(
    MATRIX_BY_LAYERS: Union[bool, List[List[int]]],
    K: int,
    LAST_LAYER: bool = False,
    INPUT_IN_VARIABLES: bool = True,
) -> List[List[int]]:


    last = K if LAST_LAYER else K - 1
    start = 0 if INPUT_IN_VARIABLES else 1

    if isinstance(MATRIX_BY_LAYERS, bool):
        if MATRIX_BY_LAYERS:
            return [[k, k + 1] for k in range(start, last)]
        else:
            return [list(range(start, last + 1))]
    else:
        assert MATRIX_BY_LAYERS[0][0] == start, (
            f"First group must start at layer {start} "
            f"(INPUT_IN_VARIABLES={INPUT_IN_VARIABLES})"
        )

        assert MATRIX_BY_LAYERS[-1][-1] == last, (
            f"Last group must end at layer {last}, got {MATRIX_BY_LAYERS[-1][-1]}"
        )
        for i in range(len(MATRIX_BY_LAYERS) - 1):
            assert MATRIX_BY_LAYERS[i][-1] == MATRIX_BY_LAYERS[i + 1][0], (
                f"Groups {MATRIX_BY_LAYERS[i]} and {MATRIX_BY_LAYERS[i + 1]} "
                f"must share exactly one boundary layer"
            )
        return MATRIX_BY_LAYERS


class Indexes_Mosek_Solver:
    """
    Class to index variables in matrices.
    """

    def __init__(
        self,
        K: int,
        n: List[int],
        MATRIX_BY_LAYERS: Union[bool, List[List[int]]] = False,
        LAST_LAYER: bool = False,
        BETAS: bool = False,
        BETAS_Z: bool = False,
        ZBAR: bool = False,
        INPUT_IN_VARIABLES: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        K : int
            Number of network layers.
        n : List[int]
            Number of neurons per layer (length K+1).
        MATRIX_BY_LAYERS : Union[bool, List[List[int]]]
            True  → standard chordal decomposition (consecutive pairs).
            False → single matrix.
            List  → explicit groups, e.g. [[0,1],[1,2,3,4],[4,5]].
        LAST_LAYER : bool
            If True, the last layer (logits) is included as a variable.
        BETAS : bool
            Enables beta variables (untargeted formulation).
        BETAS_Z : bool
            If True, betas are embedded in the last z matrix.
        ZBAR : bool
            Enables the zbar variable.
        INPUT_IN_VARIABLES : bool
            If False, z_0 is removed from the SDP variables; the first block starts
            at layer 1. The L∞ ball is then implicitly absorbed into the
            pre-computed bounds L_1, U_1.
        **kwargs
            ytrue, ytargets, stable_inactives_neurons, stable_actives_neurons,
            keep_penultimate_actives.
        """
        self.n = n
        self.K = K
        self.MATRIX_BY_LAYERS = MATRIX_BY_LAYERS  # kept for reference
        self.LAST_LAYER = LAST_LAYER
        self.BETAS = BETAS
        self.BETAS_Z = BETAS_Z
        self.ZBAR = ZBAR
        self.INPUT_IN_VARIABLES = INPUT_IN_VARIABLES

        self.ytrue = kwargs.get("ytrue")
        self.ytargets = kwargs.get("ytargets")
        self.stable_inactives_neurons = kwargs.get("stable_inactives_neurons")
        self.stable_actives_neurons = kwargs.get("stable_actives_neurons")
        self.keep_penultimate_actives = kwargs.get("keep_penultimate_actives", False)

        self.layer_groups = resolve_layer_groups(MATRIX_BY_LAYERS, K, LAST_LAYER, INPUT_IN_VARIABLES)
        self._layer_to_groups = self._build_layer_to_groups()
        self._pruned_adv_before = self._build_pruned_adv_before()


        self.check_conformity()

        self.current_matrices_variables = []
        self.count_nb_matrices()
        self.count_max_indexes()

    def _build_layer_to_groups(self) -> dict:
        """
        Mapping layer → list of (group_idx, position_in_group).
        A boundary layer shared between two groups appears twice.
        """
        mapping = {}
        for group_idx, group in enumerate(self.layer_groups):
            for pos, layer in enumerate(group):
                mapping.setdefault(layer, []).append((group_idx, pos))
        return mapping

    def _build_pruned_adv_before(self) -> dict:
        """
        Precomputes, for each class c, the number of pruned adversarial classes
        strictly before c (i.e. neither ytrue, nor in ytargets, and < c).
        """
        if self.ytargets is None or self.ytrue is None:
            return {}
        count = 0
        result = {}
        for c in range(self.n[self.K]):
            result[c] = count
            if c != self.ytrue and c not in self.ytargets:
                count += 1
        return result


    def count_nb_matrices(self):
        self.nb_matrices = len(self.layer_groups)
        if not self.BETAS_Z and self.BETAS:
            self.nb_matrices += 1  # dedicated matrix for betas
        assert self.nb_matrices < big_M_cst, (
            f"nb_matrices={self.nb_matrices} >= big_M_cst={big_M_cst}: "
            f"the hash-key encoding in variable_elements.py would overflow silently. "
            f"Increase big_M_cst in variable_elements.py before proceeding."
        )

    def count_max_indexes(self):
        max_index = 0
        for group_idx, group in enumerate(self.layer_groups):
            size = 1
            for layer in group:
                if layer == self.K and self.LAST_LAYER:
                    # Output layer: only ytargets + ytrue
                    size += len(self.ytargets) + 1
                else:
                    size += self.n[layer]
            if group_idx == len(self.layer_groups) - 1 and self.BETAS_Z:
                if self.ZBAR:
                    size += 1
                size += len(self.ytargets)
            max_index = max(max_index, size)

        if not self.BETAS_Z and self.BETAS:
            max_index = max(max_index, len(self.ytargets) + 1)

        self.max_index = max_index + 1000

    def check_conformity(self):
        """
        Checks structural consistency: K, n, and presence of unstable neurons
        in each group.
        """
        assert self.K == len(self.n) - 1
        if len(self.layer_groups) > 1:
            for group_idx, group in enumerate(self.layer_groups):
                unstable_count = sum(
                    1
                    for layer in group
                    for j in range(self.n[layer])
                    if (layer, j) not in self.stable_inactives_neurons
                    and (layer, j) not in self.stable_actives_neurons
                )
                print(
                    f"Matrix size for the group {group_idx} {group} : "
                    f"{unstable_count}"
                )
                assert unstable_count > 0, (
                    f"Group {group} has no unstable neurons — a special treatment is needed."
                )

        if any(
            all(
                (layer, j) in self.stable_inactives_neurons
                for j in range(self.n[layer])
            )
            for layer in range(1, self.K)
        ):
            raise ValueError(
                "There are layers with only inactive neurons : "
                "a special treatment is needed : the output is fixed."
            )


    def get_number_pruned_neurons_on_layer(self, layer: int, neuron: int = None) -> int:
        """
        Number of pruned neurons in a layer up to neuron `neuron`.
        If neuron=None, counts all pruned neurons in the layer.
        """
        if (
            self.stable_inactives_neurons is None
            and self.stable_actives_neurons is None
        ):
            return 0

        if neuron is None:
            neuron = self.n[layer]

        if (layer == self.K - 1) and self.keep_penultimate_actives:
            return len(
                [
                    (k, j)
                    for k, j in self.stable_inactives_neurons
                    if k == layer and j < neuron
                ]
            )
        else:
            return len(
                [
                    (k, j)
                    for k, j in (
                        self.stable_inactives_neurons + self.stable_actives_neurons
                    )
                    if k == layer and j < neuron
                ]
            )

    def get_number_pruned_neurons_before_layer(
        self, layer: int, neuron: int = None
    ) -> int:
        """
        Pruned neurons strictly before `layer`, plus those in `layer` up to `neuron`.
        """
        if self.stable_inactives_neurons is None:
            return 0
        if neuron is None:
            neuron = self.n[layer]
        return (
            sum(self.get_number_pruned_neurons_on_layer(k) for k in range(layer))
            + self.get_number_pruned_neurons_on_layer(layer, neuron)
        )

    def get_number_pruned_adversarial_targets_before_target(self, ytarget) -> int:
        """
        Number of non-ytarget, non-ytrue classes strictly before `ytarget`.
        Used to adjust the index of z_K variables when LAST_LAYER=True.
        """
        return self._pruned_adv_before[ytarget]

    def _n_vars_in_layer(self, layer: int) -> int:
        """
        Number of z variables for a given layer, after pruning.

        For layer K (LAST_LAYER=True): ytargets + ytrue are the only
        variables, i.e. len(ytargets) + 1 minus any stable neurons.
        """
        if layer == self.K and self.LAST_LAYER:
            return (
                len(self.ytargets)
                + 1  # +1 for ytrue
                - self.get_number_pruned_neurons_on_layer(layer)
            )
        return self.n[layer] - self.get_number_pruned_neurons_on_layer(layer)

    def _offset_end_of_last_group(self) -> int:
        """
        1 + number of z variables in the last group.
        Used as base offset to position beta and zbar in the matrix.
        """
        return 1 + sum(self._n_vars_in_layer(l) for l in self.layer_groups[-1])

    def is_in_matrix_with_betas(self, layer: int) -> bool:
        """True si `layer` appartient au dernier groupe (qui contient les betas quand BETAS_Z=True)."""
        if self.BETAS_Z:
            return layer in self.layer_groups[-1]
        return False

    def index_matrix_z(self, layer: int, front_of_matrix: bool) -> int:
        """
        Index of the SDP matrix for variable z_{layer}.

        front_of_matrix=True  → looks for a group where layer is not in the last position.
        front_of_matrix=False → looks for a group where layer is in the last position.
        """
        if layer not in self._layer_to_groups:
            raise ValueError(f"Layer {layer} not found in any group.")

        entries = self._layer_to_groups[layer]

        if front_of_matrix:
            for group_idx, pos_in_group in entries:
                if pos_in_group < len(self.layer_groups[group_idx]) - 1:
                    return group_idx
            raise ValueError(
                f"Layer {layer} cannot be front_of_matrix=True: "
                f"it is always the last element of its group(s)."
            )
        else:
            for group_idx, pos_in_group in entries:
                if pos_in_group == len(self.layer_groups[group_idx]) - 1:
                    return group_idx
            raise ValueError(
                f"Layer {layer} cannot be front_of_matrix=False: "
                f"it is never the last element of any group."
            )

    def index_matrix_beta(self) -> int:
        """Index of the matrix containing beta variables."""
        assert self.BETAS
        if self.BETAS_Z:
            return len(self.layer_groups) - 1  
        else:
            return len(self.layer_groups) 

    def index_matrix_zbar(self) -> int:
        """Index of the matrix containing the zbar variable."""
        assert self.ZBAR
        assert self.BETAS_Z
        return len(self.layer_groups) - 1  

    def _get_matrix_index(self, var_type: str, is_first: bool = None, **kwargs) -> int:
        """
        Dispatcher: returns the matrix index for a variable of a given type.

        Parameters
        ----------
        var_type : str
            'z', 'beta' or 'zbar'.
        is_first : bool or None
            None for linear variables, True/False for quadratic pairs.
        """
        suffix = "" if is_first is None else ("1" if is_first else "2")
        front_of_matrix = kwargs.get(f"front_of_matrix{suffix}", None)

        if var_type == "z":
            layer_key = f"layer{suffix}" if f"layer{suffix}" in kwargs else "layer"
            layer = kwargs.get(layer_key)
            if layer is None:
                raise ValueError(
                    f"Layer required for z variable ({'first' if is_first else 'second'})"
                )
            if front_of_matrix is None:
                last_layer = self.layer_groups[-1][-1]
                front_of_matrix = (layer != last_layer)
            return self.index_matrix_z(layer, front_of_matrix)

        elif var_type == "beta":
            class_label_key = (
                f"class_label{suffix}" if f"class_label{suffix}" in kwargs else "class_label"
            )
            if kwargs.get(class_label_key) is None:
                raise ValueError(
                    f"Class label required for beta variable ({'first' if is_first else 'second'})"
                )
            return self.index_matrix_beta()

        elif var_type == "zbar":
            return self.index_matrix_zbar()

        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    def get_shape_matrix(self, num_matrix: int):
        """Dimension of matrix num_matrix."""
        return self.current_matrices_variables[num_matrix]["dim"]

    def get_name_matrix(self, num_matrix: int):
        """Name of matrix num_matrix."""
        return self.current_matrices_variables[num_matrix]["name"]


    def index_variable_z(self, layer: int, neuron: int, front_of_matrix: bool) -> int:
        """
        Index of z_{layer, neuron} in its SDP matrix.

        The matrix is identified by front_of_matrix (see index_matrix_z).
        The position in the matrix [1, z_{g_0}, z_{g_1}, ..., z_{g_m}] is:
          1 + Σ n_vars_in_layer(g_p) for p < pos + neuron offset in the layer.
        """
        if (layer == self.K and not self.LAST_LAYER) or layer < 0 or layer > self.K:
            raise ValueError(f"Layer index {layer} out of range.")
        if (layer, neuron) in self.stable_inactives_neurons:
            raise ValueError(
                f"Neuron {neuron} in layer {layer} is inactive and has no z variable."
            )

        matrix_idx = self.index_matrix_z(layer, front_of_matrix)
        group = self.layer_groups[matrix_idx]
        pos = group.index(layer)

        offset = 1 + sum(self._n_vars_in_layer(group[p]) for p in range(pos))

        if layer == self.K and self.LAST_LAYER:
            offset += (
                neuron
                - self.get_number_pruned_adversarial_targets_before_target(neuron)
                - self.get_number_pruned_neurons_on_layer(layer, neuron)
            )
        else:
            offset += neuron - self.get_number_pruned_neurons_on_layer(layer, neuron)

        return offset

    def ind_label_beta(self, class_label: int) -> int:
        """0-based position of beta_{class_label} in the list of active target classes."""
        if not self.BETAS:                                                                                                                                                                                               
          raise ValueError("Beta variables are not enabled.")                                                                                                                                                          
        if class_label == self.ytrue:                                                                                                                                                                                    
          raise ValueError("The true class label has no beta variable.")                                                                                                                                               
        if class_label not in self.ytargets:                                                                                                                                                                             
          raise ValueError(
              f"Class {class_label} is not an active target (pruned or invalid)."
          ) 
        pruned_before = self.get_number_pruned_adversarial_targets_before_target(class_label)
        if class_label < self.ytrue:
            return class_label - pruned_before
        else:
            return class_label - 1 - pruned_before

    def index_variable_beta(self, class_label: int) -> int:
        """
        Index of beta_{class_label} in its matrix.

        BETAS_Z=True  → embedded in the last z group, after zbar if ZBAR.
        BETAS_Z=False → dedicated matrix, 1-based position.
        """
        assert self.BETAS
        if self.BETAS_Z:
            base = self._offset_end_of_last_group()
            if self.ZBAR:
                base += 1
            return base + self.ind_label_beta(class_label)
        else:
            return 1 + self.ind_label_beta(class_label)

    def index_variable_zbar(self) -> int:
        """Index of zbar in the last z matrix (just after z variables, before betas)."""
        assert self.ZBAR
        assert self.BETAS_Z
        return self._offset_end_of_last_group()

    def _get_variable_index(self, var_type: str, is_first: bool = None, **kwargs) -> int:
        """
        Dispatcher: returns the variable index for a given type.

        Parameters
        ----------
        var_type : str
            'z', 'beta' or 'zbar'.
        is_first : bool or None
            None for linear variables, True/False for quadratic pairs.
        """
        suffix = "" if is_first is None else ("1" if is_first else "2")
        front_of_matrix = kwargs.get(f"front_of_matrix{suffix}", None)

        if var_type == "z":
            layer_key = f"layer{suffix}" if f"layer{suffix}" in kwargs else "layer"
            neuron_key = f"neuron{suffix}" if f"neuron{suffix}" in kwargs else "neuron"
            layer = kwargs.get(layer_key)
            neuron = kwargs.get(neuron_key)
            if layer is None or neuron is None:
                raise ValueError(
                    f"Layer and neuron required for z variable ({'first' if is_first else 'second'})"
                )
            if front_of_matrix is None:
                last_layer = self.layer_groups[-1][-1]
                front_of_matrix = (layer != last_layer)
            return self.index_variable_z(layer, neuron, front_of_matrix)

        elif var_type == "beta":
            class_label_key = (
                f"class_label{suffix}" if f"class_label{suffix}" in kwargs else "class_label"
            )
            class_label = kwargs.get(class_label_key)
            if class_label is None:
                raise ValueError(
                    f"Class label required for beta variable ({'first' if is_first else 'second'})"
                )
            return self.index_variable_beta(class_label)

        elif var_type == "zbar":
            return self.index_variable_zbar()

        else:
            raise ValueError(f"Unknown variable type: {var_type}")
