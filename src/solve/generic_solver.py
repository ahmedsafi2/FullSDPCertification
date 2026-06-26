import logging
from typing import List
import torch
import numpy as np
import yaml
from fastsdp_tools.yaml_config import FullCertificationConfig
from networks import ReLUNN
from pydantic import ValidationError
import datetime

from .getting_results import get_results_trivially_solved
from bounds import (
    compute_bounds_,
    check_stability_neurons,
    prune_adversarial_targets,
    compute_IBP,
    compute_bounds_data_crown
)
from fastsdp_tools import (
    add_functions_to_class,
    get_project_path,
    create_folder,
    round_list_depth_2,
    round_list_depth_3,
    change_to_zero_negative_values,
)

logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@add_functions_to_class(
    compute_bounds_,
    compute_IBP,
    check_stability_neurons,
    prune_adversarial_targets,
    get_results_trivially_solved,
    compute_bounds_data_crown,
)
class Solver:
    def __init__(
        self,
        network: ReLUNN,
        epsilon: float,
        x: List[float],
        ytrue: int,
        verbose: bool = False,
        L: List[List[float]] = None,
        U: List[List[float]] = None,
        use_inactive_neurons: bool = False,
        use_active_neurons: bool = False,
        **kwargs,
    ):
        self.network = network.cpu()  
        self.K = network.K
        self.n = network.n
        
        self.W = network.W
        self.b = network.b

        self.n = np.array(self.n)
        self.W = [np.array(self.W[k - 1]) for k in range(1, self.K + 1)]

        self.b = [np.array(self.b[k]) for k in range(self.K)]

        self.dataset = kwargs.get("dataset")
        self.epsilon = epsilon
        self.norm = kwargs.get("norm", "Linf")
        logger.debug("Norm used in Solver: ", self.norm)
        network_device = next(network.parameters()).device
        self.x = x.to(network_device)
        
        self.ytrue = ytrue
        logger.debug("ytrue = ", self.ytrue)
        with torch.no_grad():
            self.label_predicted = self.network.label(self.x.cpu())

        self.ytarget = kwargs.get("ytarget", None)

        if "Targeted" in self.__class__.__name__ and self.ytarget is not None:
            self.ytargets = [self.ytarget]
        else:
            self.ytargets = [j for j in range(self.n[self.K]) if j != self.ytrue]

        logger.debug("ytargets : ", self.ytargets)
        
        self.bounds_method = kwargs.get("bounds_method")
        self.keep_penultimate_actives = kwargs.get("keep_penultimate_actives", None)
        self.ultimate_layer_use_active_neurons = kwargs.get("ultimate_layer_use_active_neurons", self.K+1)
        logger.debug("COEFF : self ultimate_layer_use_active_neurons : ", self.ultimate_layer_use_active_neurons)

        if L is None or U is None:
            self.compute_bounds_(
                method=self.bounds_method,
            )
        else : 
            self.L = L
            self.U = U
            self.compute_bounds_time = 0

            
        self.L = [np.array(self.L[k]) for k in range(self.K + 1)]
        self.U = [np.array(self.U[k]) for k in range(self.K + 1)]


 

        raw_iiv = kwargs.get("INPUT_IN_VARIABLES", True)
        if isinstance(raw_iiv, bool):
            self.input_proportion = 1.0 if raw_iiv else 0.0
        else:
            self.input_proportion = float(raw_iiv)

        if self.input_proportion <= 0.0:
            self.INPUT_IN_VARIABLES = False
            self.kept_input_neurons = set()
            self.pruned_input_neurons = set(range(int(self.n[0])))
        elif self.input_proportion >= 1.0:
            self.INPUT_IN_VARIABLES = True
            self.kept_input_neurons = set(range(int(self.n[0])))
            self.pruned_input_neurons = set()
        else:
            self.INPUT_IN_VARIABLES = True
            col_norms = np.sum(np.abs(self.W[0]), axis=0)  # shape (n[0],)
            n_keep = max(1, int(np.ceil(self.input_proportion * self.n[0])))
            top_indices = np.argsort(col_norms)[-n_keep:]
            self.kept_input_neurons = set(top_indices.tolist())
            self.pruned_input_neurons = set(range(int(self.n[0]))) - self.kept_input_neurons

        self.use_inactive_neurons = use_inactive_neurons
        self.use_active_neurons = use_active_neurons
        self.check_stability_neurons(
            use_active_neurons=use_active_neurons,
            use_inactive_neurons=use_inactive_neurons,
        )

        
        logger.debug("COEFF ACTIVES  after checking stable active neurons:", self.stable_actives_neurons)

        logger.debug("COEFF  after checking stable inactive neurons:", self.stable_inactives_neurons)

        self.U_above_zero = change_to_zero_negative_values(
            self.U, dim=2
        )  
        self.L_above_zero = change_to_zero_negative_values(
            self.L, dim=2
        )  

        self.LAST_LAYER = kwargs.get("LAST_LAYER", False)
        self.prune_adversarial_targets()

        self.is_robust = None
        self.best_adversarial_examples = None
        self.verbose = verbose

        self.name = kwargs.get("certification_model_type")

        self.folder_name = kwargs.get("folder_name", None)

        if self.folder_name is None:
            self.folder_name = "results"

        create_folder(f"{self.folder_name}/{self.name}")

        self.benchmark_dataframe = None
        self.data_index = kwargs.get("data_index", 0)
        self.network_name = kwargs.get("network_name", "ReLUNN")
        self.dataset_name = kwargs.get("dataset_name")
        logger.debug("everything good in init")

        logger.debug("n : ", self.n)
        logger.debug("K ⁼ ", self.K)
        with open("weights_nn.txt", "w") as f:
            f.write(f"K = {self.K}, n = {self.n}")
            for k in range(1, self.K + 1):
                f.write(f"\n\n\n\n     Layer : {k}, taille : {self.n[k]}")
                for j in range(self.n[k]):

                    f.write(f"\n\n         Neuron : {j}")
                    f.write(f"\n            W = {self.W[k - 1][j]}")
                    f.write(f"\n            b = {self.b[k - 1][j]}")
                    f.write(f"\n            U = {self.U[k][j]}, L = {self.L[k][j]}")

        for k in range(1, self.K + 1):
            for j in range(self.n[k]):
                if self.U[k][j] <= 0:
                    logger.debug(f"Neuron {j} in layer {k} is stable inactive with U = {self.U[k][j]} and L = {self.L[k][j]}")
                elif self.L[k][j] >= 0:
                    logger.debug(f"Neuron {j} in layer {k} is stable active with U = {self.U[k][j]} and L = {self.L[k][j]}")
                else:
                    logger.debug(f"Neuron {j} in layer {k} is unstable with U = {self.U[k][j]} and L = {self.L[k][j]}")

        self.is_trivially_solved = self.ytargets == []
        logger.debug("is trivially solved: ", self.is_trivially_solved)

    @staticmethod
    def parse_yaml(yaml_file):
        with open(yaml_file, "r") as f:
            raw_config = yaml.safe_load(f)

        try:
            validated_config = FullCertificationConfig(**raw_config)
        except ValidationError as e:
            raise

        return dict(
            dataset=validated_config.data.name,
            network=ReLUNN.from_yaml(yaml_file),
            epsilon=validated_config.epsilon,
            x=validated_config.data.x,
            ytrue=validated_config.data.y,
            ytarget=validated_config.data.ytarget,
            bounds_method=validated_config.data.bounds_method,
            L=validated_config.data.L,
            U=validated_config.data.U,
        )

    @classmethod
    def from_yaml(cls, yaml_file, **kwargs):
        params = cls.parse_yaml(yaml_file)
        return cls(**params, **kwargs)
