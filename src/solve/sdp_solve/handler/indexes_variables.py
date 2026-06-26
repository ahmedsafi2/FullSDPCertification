# Backward compatibility: implementation is in indexes.py
from .indexes import Indexes_Mosek_Solver

Indexes_Variables_for_Mosek_Solver = Indexes_Mosek_Solver

__all__ = [
    "Indexes_Variables_for_Mosek_Solver",
]
