from typing import List, Union


def resolve_layer_groups(
    MATRIX_BY_LAYERS: Union[bool, List[List[int]]],
    K: int,
    LAST_LAYER: bool = False,
) -> List[List[int]]:
    """
    Résout MATRIX_BY_LAYERS en liste canonique de groupes de couches.

    Exemples :
      True,  K=5 → [[0,1],[1,2],[2,3],[3,4]]
      False, K=5 → [[0,1,2,3,4]]
      [[0,1],[1,2,3,4],[4,5]] → tel quel (avec K=5, LAST_LAYER=True)
    """
    last = K if LAST_LAYER else K - 1
    if isinstance(MATRIX_BY_LAYERS, bool):
        if MATRIX_BY_LAYERS:
            return [[k, k + 1] for k in range(last)]
        else:
            return [list(range(last + 1))]
    else:
        assert MATRIX_BY_LAYERS[0][0] == 0, "First group must start at layer 0"
        assert MATRIX_BY_LAYERS[-1][-1] == last, (
            f"Last group must end at layer {last}, got {MATRIX_BY_LAYERS[-1][-1]}"
        )
        for i in range(len(MATRIX_BY_LAYERS) - 1):
            assert MATRIX_BY_LAYERS[i][-1] == MATRIX_BY_LAYERS[i + 1][0], (
                f"Groups {MATRIX_BY_LAYERS[i]} and {MATRIX_BY_LAYERS[i + 1]} "
                f"must share exactly one boundary layer"
            )
        return MATRIX_BY_LAYERS


class Indexes_Matrixes_for_Mosek_Solver:
    """
    Class to handle the indexes of the matrices in the MOSEK solver.
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
        **kwargs,
    ):
        """
        Initialize the Indexes_Mosek_Solver class.

        Parameters
        ----------
        n: List[int]
            List of the number of neurons in each layer.
        K: int
            Number of layers.
        MATRIX_BY_LAYERS: Union[bool, List[List[int]]]
            If bool: True = standard pairwise chordal decomposition, False = single matrix.
            If List[List[int]]: explicit grouping of layers into matrices, e.g. [[0,1],[1,2,3,4],[4,5]].
            Consecutive groups must share exactly one boundary layer.
        LAST_LAYER: bool
            Whether the last layer (logits) is included as variables.
        BETAS: bool
            Whether to include beta variables.
        BETAS_Z: bool
            Whether beta variables are embedded in the last z-matrix.
        ZBAR: bool
            Whether to include zbar variables.
        """
        self.n = n
        self.K = K
        self.LAST_LAYER = LAST_LAYER
        self.BETAS = BETAS
        self.BETAS_Z = BETAS_Z
        self.ZBAR = ZBAR

        self.ytargets = kwargs.get("ytargets")
        self.stable_inactives_neurons = kwargs.get("stable_inactives_neurons")
        self.stable_actives_neurons = kwargs.get("stable_actives_neurons")

        self.layer_groups = resolve_layer_groups(MATRIX_BY_LAYERS, K, LAST_LAYER)
        self._layer_to_groups = self._build_layer_to_groups()  # Mapping layer → list of (group_idx, position_in_group)merc

        self.check_conformity()

        self.current_matrices_variables = []
        self.count_nb_matrices()

    def _build_layer_to_groups(self) -> dict:
        """
        Build a mapping: layer → list of (group_idx, position_in_group).
        A boundary layer shared between two groups appears twice.
        """
        mapping = {}
        for group_idx, group in enumerate(self.layer_groups):
            for pos, layer in enumerate(group):
                mapping.setdefault(layer, []).append((group_idx, pos))
        return mapping

    def count_nb_matrices(self):
        """
        Count the number of SDP matrices based on the layer grouping.
        """
        self.nb_matrices = len(self.layer_groups)
        if not self.BETAS_Z and self.BETAS:
            self.nb_matrices += 1  # separate matrix for beta variables

    def check_conformity(self):
        """
        Check the presence of unstable neurons in each layer.
        Check the number of layers and the number of neurons in each layer.
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
                print(f"Taille de la matrice pour le groupe {group_idx} {group} : {unstable_count}")
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
                "There are layers with only inactive neurons : a special treatment is needed : the output is fixed."
            )

    def is_in_matrix_with_betas(self, layer: int) -> bool:
        """
        Check if `layer` belongs to the last z-matrix (which carries beta variables when BETAS_Z=True).
        """
        if self.BETAS_Z:
            last_group = self.layer_groups[-1]
            return layer in last_group
        return False

    def index_matrix_z(self, layer: int, front_of_matrix: bool) -> int:
        """
        Get the index of the SDP matrix for variable z_{layer}.

        Parameters
        ----------
        layer: int
            The layer number.
        front_of_matrix: bool
            True  → the layer is NOT the last in its group (look for a group where it appears before the end).
            False → the layer IS the last in its group (look for the group where it is the last element).

        Returns
        -------
        int
            The index of the matrix.
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
        """
        Get the index of the matrix containing beta variables.
        """
        assert self.BETAS
        if self.BETAS_Z:
            # Betas are embedded in the last z-matrix
            return len(self.layer_groups) - 1
        else:
            # Betas have their own dedicated matrix, after all z-matrices
            return len(self.layer_groups)

    def index_matrix_zbar(self) -> int:
        """
        Get the index of the matrix containing the zbar variable.
        """
        assert self.ZBAR
        assert self.BETAS_Z
        # zbar is embedded in the last z-matrix alongside betas
        return len(self.layer_groups) - 1

    def _get_matrix_index(self, var_type: str, is_first: bool = None, **kwargs):
        """
        Helper method to get matrix index based on type and parameters.

        Parameters
        ----------
        var_type : str
            Type of the variable: 'z', 'beta', or 'zbar'.
        is_first : bool or None
            None for linear variables (no suffix), True/False for quadratic pairs.
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
                # Default: last layer of the network can only be back-of-matrix
                last_layer = self.layer_groups[-1][-1]
                front_of_matrix = (layer != last_layer)

            return self.index_matrix_z(layer, front_of_matrix)

        elif var_type == "beta":
            class_label_key = (
                f"class_label{suffix}" if f"class_label{suffix}" in kwargs else "class_label"
            )
            class_label = kwargs.get(class_label_key)
            if class_label is None:
                raise ValueError(
                    f"Class label required for beta variable ({'first' if is_first else 'second'})"
                )
            return self.index_matrix_beta()

        elif var_type == "zbar":
            return self.index_matrix_zbar()

        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    def get_shape_matrix(self, num_matrix: int):
        """
        Get the shape of the matrix based on the number of neurons in each layer.
        """
        return self.current_matrices_variables[num_matrix]["dim"]

    def get_name_matrix(self, num_matrix: int):
        """
        Get the name of the matrix based on the number of neurons in each layer.
        """
        return self.current_matrices_variables[num_matrix]["name"]
