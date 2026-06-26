import logging
from .sdp_generic_solver import *
from .get_variables import *
from .SDPmodels.Targeted_SDP import TargetedSDP
from .SDPmodels.Untargeted_SDP import UntargetedSDP
import os
from fastsdp_tools.utils import get_project_path
from solve.sdp_solve.handler.sdp_variable_mapper import NeuronLinearization 
from run_benchmark import concat_dataframes_with_missing_columns


logger_mosek = logging.getLogger("Mosek_logger")
logger_mosek.setLevel(logging.DEBUG)
logger_mosek.propagate = False
handler = logging.FileHandler(get_project_path("results/Mosek_logger.log"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger_mosek.addHandler(handler)


__all__ = ["SDPSolver", "TargetedSDP", "UntargetedSDP", "LayersValues", "concat_dataframes_with_missing_columns"]
