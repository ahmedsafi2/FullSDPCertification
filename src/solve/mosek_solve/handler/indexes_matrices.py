# Rétrocompatibilité : l'implémentation est dans indexes.py
from .indexes import Indexes_Mosek_Solver, resolve_layer_groups

Indexes_Matrixes_for_Mosek_Solver = Indexes_Mosek_Solver

__all__ = [
    "Indexes_Matrixes_for_Mosek_Solver",
    "resolve_layer_groups",
]
