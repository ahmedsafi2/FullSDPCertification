# Indexation
from .indexes import Indexes_Mosek_Solver, resolve_layer_groups
from .indexes_matrices import Indexes_Matrixes_for_Mosek_Solver  # alias rétrocompat
from .indexes_variables import Indexes_Variables_for_Mosek_Solver  # alias rétrocompat

# Éléments de base (variables, contraintes, objectif)
from .variable_elements import (
    ElementsinConstraintsObjectives,
    Equivalent_Neurons_Index,
    Equivalent_Betas_Index,
    add_dict_linear_to_elements,
    add_dict_quad_to_elements,
)
from .variables_call import VariablesCall, LayersValues
from .constraints import CommonConstraints
from .objective import Objective

# Solveurs concrets
from .mosek_classic import MosekClassicHandler, ConstraintsClassic, ObjectiveClassic
from .mosek_fusion import MosekFusionHandler

__all__ = [
    # Indexation
    "Indexes_Mosek_Solver",
    "resolve_layer_groups",
    "Indexes_Matrixes_for_Mosek_Solver",  # alias rétrocompat
    "Indexes_Variables_for_Mosek_Solver",  # alias rétrocompat
    # Éléments
    "ElementsinConstraintsObjectives",
    "Equivalent_Neurons_Index",
    "Equivalent_Betas_Index",
    "add_dict_linear_to_elements",
    "add_dict_quad_to_elements",
    # Base
    "VariablesCall",
    "LayersValues",
    "CommonConstraints",
    "Objective",
    # Solveurs
    "MosekClassicHandler",
    "ConstraintsClassic",
    "ObjectiveClassic",
    "MosekFusionHandler",
]