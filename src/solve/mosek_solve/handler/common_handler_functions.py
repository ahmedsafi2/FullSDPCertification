from tools import (
    get_m_indexes_of_higher_values_in_list,
    get_project_path, 
    create_folder, 
    add_row_from_dict
)
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List
from solve.mosek_solve.run_benchmark import (
    compute_cuts_str,
    print_dual_variable_to_file_for_cb_solver,
    print_solution_to_file_for_cb_solver
)


logger_mosek = logging.getLogger("Mosek_logger")


def initialize_variables(self):
    """
    Add variables to the task.
    """
    logger_mosek.info("Initializing variables...")
    print("Initializing variables...")
    if self.BETAS_Z:
        logger_mosek.info("Model with betaz variables")
        if self.MATRIX_BY_LAYERS:
            logger_mosek.info("Model with matrices by layers")
            for k in range(self.K - 2):
                self.add_matrix_variable(
                    name=f"z_layers_{k}_{k+1}",
                    dim=1
                    + self.n[k]
                    + self.n[k + 1]
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(layer=k)
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(
                        layer=k + 1
                    ),
                )
            if self.LAST_LAYER:
                logger_mosek.info("Model with last layer in solution matrices")
                self.add_matrix_variable(
                    name=f"z_layers_{self.K-2}_{self.K-1}",
                    dim=1
                    + self.n[self.K - 2]
                    + self.n[self.K - 1]
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(
                        layer=self.K - 2
                    )
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(
                        layer=self.K - 1
                    ),
                )
                if self.ZBAR:
                    logger_mosek.info("Model with zbar")
                    self.add_matrix_variable(
                        name=f"z_layers_{self.K-1}_{self.K}_zbar_betas",
                        dim=1
                        + self.n[self.K - 1]
                        + self.n[self.K]
                        + 1
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_on_layer(
                            layer=self.K - 1
                        ),
                    )
                else:
                    logger_mosek.info("Model without zbar")
                    self.add_matrix_variable(
                        name=f"z_layers_{self.K-1}_{self.K}_betas",
                        dim=1
                        + self.n[self.K - 1]
                        + self.n[self.K]
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_on_layer(
                            layer=self.K - 1
                        ),
                    )

            else:
                if self.ZBAR:
                    logger_mosek.info("Model with zbar")
                    self.add_matrix_variable(
                        name=f"z_layers_{self.K-2}_{self.K-1}_zbar_betas",
                        dim=1
                        + self.n[self.K - 2]
                        + self.n[self.K - 1]
                        + 1
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_on_layer(
                            layer=self.K - 2
                        )
                        - self.indexes_variables.get_number_pruned_neurons_on_layer(
                            layer=self.K - 1
                        ),
                    )
                else:
                    logger_mosek.info("Model without zbar")
                    self.add_matrix_variable(
                        name=f"z_layers_{self.K-2}_{self.K-1}_betas",
                        dim=1
                        + self.n[self.K - 2]
                        + self.n[self.K - 1]
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_on_layer(
                            layer=self.K - 2
                        )
                        - self.indexes_variables.get_number_pruned_neurons_on_layer(
                            layer=self.K - 1
                        ),
                    )
        else:
            if self.LAST_LAYER:
                logger_mosek.info("Model with last layer in solution matrices")
                if self.ZBAR:
                    logger_mosek.info("Model with zbar")
                    self.add_matrix_variable(
                        name="z_all_layers_zbar_betas",
                        dim=1
                        + sum(self.n)
                        + 1
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_before_layer(
                            layer=self.K - 1
                        ),
                    )
                else:
                    logger_mosek.info("Model without zbar")
                    self.add_matrix_variable(
                        name="z_all_layers_betas",
                        dim=1
                        + sum(self.n)
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_before_layer(
                            layer=self.K - 1
                        ),
                    )
            else:
                if self.ZBAR:
                    logger_mosek.info("Model with zbar")
                    self.add_matrix_variable(
                        name="z_all_layers_until_penultimate_zbar_betas",
                        dim=1
                        + sum(self.n[: self.K])
                        + 1
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_before_layer(
                            layer=self.K - 1,
                        ),
                    )
                else:
                    logger_mosek.info("Model without zbar")
                    self.add_matrix_variable(
                        name="z_all_layers_until_penultimate_betas",
                        dim=1
                        + sum(self.n[: self.K])
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_before_layer(
                            layer=self.K - 1,
                        ),
                    )

    else:
        if self.MATRIX_BY_LAYERS:
            logger_mosek.info("Model with matrices by layers")
            for k in range(self.K - 1):
                self.add_matrix_variable(
                    name=f"z_layers_{k}_{k+1}",
                    dim=1
                    + self.n[k]
                    + self.n[k + 1]
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(layer=k)
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(
                        layer=k
                        + 1  # Peut-être qu'il y a une erreur sur le layer choisi ici ou plus haut
                    ),
                )
            if self.LAST_LAYER:
                self.add_matrix_variable(
                    name=f"z_layers_{self.K-1}_{self.K}",
                    dim=1
                    + self.n[self.K - 1]
                    + self.n[self.K]
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(
                        layer=self.K
                    ),
                )
        else:
            if self.LAST_LAYER:
                self.add_matrix_variable(
                    name="z_all_layers",
                    dim=1
                    + sum(self.n)
                    - self.indexes_variables.get_number_pruned_neurons_before_layer(
                        layer=self.K
                    ),
                )
            else:
                print("Adding z_all_layers until penultimate layer without betas")
                print("sum(self.n[: self.K]) : ", sum(self.n[: self.K]))
                print(
                    "self.indexes_variables.get_number_pruned_neurons_before_layer(self.K) : ",
                    self.indexes_variables.get_number_pruned_neurons_before_layer(
                        self.K
                    ),
                )
                self.add_matrix_variable(
                    name="z_all_layers",
                    dim=1
                    + sum(self.n[: self.K])
                    - self.indexes_variables.get_number_pruned_neurons_before_layer(
                        self.K
                    ),
                )
        if self.BETAS:
            self.add_vector_variable(name="betas", dim=self.n[self.K])
    
    assert len(self.indexes_matrices.current_matrices_variables) == self.indexes_matrices.nb_matrices, (                                                                                    
        f"Nombre de matrices incohérent : {len(self.indexes_matrices.current_matrices_variables)} créées "                                                                                  
        f"mais {self.indexes_matrices.nb_matrices} attendues "                                                                                                                              
        f"(BETAS_Z={self.BETAS_Z}, MATRIX_BY_LAYERS={self.MATRIX_BY_LAYERS}, "
        f"LAST_LAYER={self.LAST_LAYER}, BETAS={self.BETAS}, K={self.K})"                                                                                                                    
    ) 



def print_index_variables_matrices(self):
    """
    String representation of the class.
    """
    line = ""

    for layer in range(self.K + 1 if self.LAST_LAYER else self.K):
        line += f"\n Layer {layer} : \n"

        for j in range(self.n[layer]):
            line += f"      Neuron {j} : \n"

            if (layer, j) in self.stable_inactives_neurons:
                line += "           is inactive \n"
                continue
            if (layer < self.K - 1 and not self.LAST_LAYER) or (
                layer < self.K and self.LAST_LAYER
            ):

                ind_matrix_front = self.indexes_matrices._get_matrix_index(
                    "z", layer=layer, neuron=j, front_of_matrix=True
                )

                ind_col_front = self.indexes_variables._get_variable_index(
                    "z", layer=layer, neuron=j, front_of_matrix=True
                )
                line += f"          front : index = {ind_matrix_front}, i = {ind_col_front} \n"
            if layer > 0:
                ind_col_back = self.indexes_variables._get_variable_index(
                    "z", layer=layer, neuron=j, front_of_matrix=False
                )
                ind_matrix_back = self.indexes_matrices._get_matrix_index(
                    "z", layer=layer, neuron=j, front_of_matrix=False
                )
                line += f"          back  : index = {ind_matrix_back}, i = {ind_col_back} \n"

    if self.ZBAR:
        line += "\n  Zbar : \n"
        ind_matrix = self.indexes_matrices._get_matrix_index("zbar")
        ind_col = self.indexes_variables._get_variable_index("zbar")
        line += f"          index = {ind_matrix}, i = {ind_col} \n"

    if self.BETAS:
        line += "\n  Betas : \n"
        for class_label in self.ytargets:
            if class_label == self.ytrue:
                continue
            line += f"      Class {class_label} : \n"
            ind_matrix = self.indexes_matrices._get_matrix_index(
                "beta", class_label=class_label
            )
            ind_col = self.indexes_variables._get_variable_index(
                "beta", class_label=class_label
            )
            line += f"          index = {ind_matrix}, i = {ind_col} \n"

    print(line)
    return line


def num_matrices_variables(self):
    """
    Return the number of matrices.
    """
    return len(self.indexes_matrices.current_matrices_variables)

def print_num_variables(self):
    num_variables = 0
    for i in range(self.num_matrices_variables()):
        dim = self.indexes_matrices.current_matrices_variables[i]["dim"]
        num_variables += (dim+1) * (dim+1)
    print(f"CALLBACK num variables : {num_variables}")
    return num_variables


def save_matrix_png(self, mat, name_solution, cuts: List):
    """
    Save a 2D matrix as an image with values rounded to two decimal places.
    """
    cuts_str = compute_cuts_str(cuts)
    max_width = 200

    mat = np.array(mat)
    n_rows, n_cols = mat.shape
    model_dir = get_project_path(f"{self.folder_name}/{self.name}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if n_rows < max_width:

        fig, ax = plt.subplots()
        ax.axis("off")
        ax.set_title(self.name, fontsize=14, pad=20)

        table_data = [
            [f"{mat[i, j]:.2f}" for j in range(n_cols)] for i in range(n_rows)
        ]
        table = ax.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(n_cols)))

        plt.savefig(
            os.path.join(
                model_dir,
                f"{name_solution}_{cuts_str}_solution.png",
            ),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def save_matrix_csv(self, mat, name_solution, cuts: List):
    """
    Save a the solution matrix as a CSV file with values rounded to two decimal places.
    """

    cuts_str = compute_cuts_str(cuts)

    model_dir = get_project_path(f"{self.folder_name}/{self.name}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    mat = pd.DataFrame(np.round(mat, decimals=2))

    mat.to_csv(
        os.path.join(model_dir, f"{name_solution}_{cuts_str}.csv"),
        index=False,
    )
    plt.close()





class Matrices_Solutions:
    def __init__(self):
        self._data = {}

    def add_value(self, cuts, value):
        """
        Ajoute une value à une configuration existante

        Args:
            cuts: Liste ou ensemble des coupes actives
            value: value à ajouter
        """
        key = frozenset(cuts)
        if key not in self._data:
            print(
                f"Ajout de la configuration {key} non presente avec la value de dim {value.shape}"
            )
            self._data[key] = value

        else:
            raise ValueError(
                f"Configuration {key} déjà présente avec la value de dim {self._data[key].shape}"
            )

        return self

    def configurations_disponibles(self):
        """Retourne toutes les configurations de coupes enregistrées"""
        return [set(config) for config in self._data.keys()]

    def __getitem__(self, cuts):
        """Permet d'utiliser l'opérateur [] pour accéder aux configurations"""
        """
        Récupère les values pour une combinaison de coupes actives

        Args:
            cuts: Liste ou ensemble des coupes actives

        Returns:
            Liste de values associées ou liste vide si combinaison non trouvée
        """
        key = frozenset(cuts)
        if key not in self._data:
            print(f"Configuration {key} non trouvée")
            raise ValueError(
                f"Configuration {key} non trouvée dans les configurations disponibles"
            )
        return self._data.get(key)

    def __contains__(self, cuts):
        """Permet d'utiliser l'opérateur 'in' pour vérifier si une configuration existe"""
        return frozenset(cuts) in self._data


def get_matrices_variables(self, cuts: List):
    """
    Get the matrices variables of the optimization problem.
    """
    if self.current_matrices_variables is None:
        raise ValueError(
            "No matrices variables found. Please initialize the variables first."
        )
    matrices = []
    for ind_solution in range(len(self.current_matrices_variables)):
        name_solution = self.current_matrices_variables[ind_solution]["name"]
        mat = self.current_matrices_variables[ind_solution]["value"][cuts]
        matrices.append(mat)
    return matrices


def compute_solutions(self, cuts: List, print_sol: bool = False):
    """
    Get the solutions and dual variables of the optimization problem.
    """
    cuts_str = compute_cuts_str(cuts)
    if print_sol:
        file_cb = open(
            get_project_path(f"{self.folder_name}/{self.name}/results_{cuts_str}.txt"),
            "w",
        )
        file_cb.write("Primal Solutions \n")
    for ind_solution in range(len(self.indexes_matrices.current_matrices_variables)):

        name_solution = self.indexes_matrices.current_matrices_variables[ind_solution][
            "name"
        ]
        dim = self.indexes_matrices.current_matrices_variables[ind_solution]["dim"]

        sol = self.get_solution(
            ind_solution=ind_solution, name_solution=name_solution, dim=dim
        )

        self.indexes_matrices.current_matrices_variables[ind_solution][
            "value"
        ].add_value(cuts, sol)

        # if verbose:
        #     logger_mosek.debug(
        #         f"Solution for {name_solution} of dimension {dim}: {mat}"
        #     )

        self.save_matrix_png(sol, name_solution=name_solution, cuts=cuts)
        self.save_matrix_csv(sol, name_solution=name_solution, cuts=cuts)

        if print_sol:
            print_solution_to_file_for_cb_solver(
                sol,
                index_matrix=ind_solution,
                dim=dim,
                file_cb=file_cb,
            )

    if print_sol:
        file_cb.write("Dual Solutions \n")
        self.get_dual_variables()
        print_dual_variable_to_file_for_cb_solver(
            list_cstr=self.Constraints.list_cstr, file_cb=file_cb
        )
        file_cb.close()
