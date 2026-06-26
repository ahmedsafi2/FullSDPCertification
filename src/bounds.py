import logging
import torch
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import time

from fastsdp_tools import round_list_depth_2

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def compute_bounds_data_crown(
    self,
    method="alpha-beta-crown",
):
    """
    Compute pre-activation bounds using alpha-beta-CROWN (modern API)
    """

    network = self.network.to(device).eval()

    if not torch.is_tensor(self.x):
        self.x = torch.tensor(self.x, dtype=torch.float32)

    x = self.x.view(1, -1).to(device)
    zeros = torch.zeros_like(x)

    bounded_model = BoundedModule(
        network,
        zeros,
        bound_opts={"conv_mode": "patches"}
    ).to(device)

    if self.norm == "Linf":
        ptb = PerturbationLpNorm(norm=np.inf, eps=self.epsilon)
    elif self.norm == "L2":
        ptb = PerturbationLpNorm(norm=2, eps=self.epsilon)
    else:
        raise ValueError("Norm not supported")

    bounded_x = BoundedTensor(x, ptb)

    use_crown = method.lower() in ["alpha-beta-crown", "beta-crown", "crown-optimized"]

    if use_crown:
        with torch.no_grad():
            lb_ibp, ub_ibp = bounded_model.compute_bounds(
                x=(bounded_x,),
                method="IBP",
                bound_lower=True,
                bound_upper=True,
            )
        
        
        lb, ub = bounded_model.compute_bounds(
            x=(bounded_x,),
            method="CROWN-Optimized",
            bound_lower=True,
            bound_upper=True,
        )
        
        
        if ub is None:
            ub = ub_ibp
        if lb is None:
            lb = lb_ibp
            
    else:
        with torch.no_grad():
            lb, ub = bounded_model.compute_bounds(
                x=(bounded_x,),
                method="IBP",
                bound_lower=True,
                bound_upper=True,
            )
 
    preact_bounds = {}
    
    
    k = 0
    L = []
    U = []
    for _, node in enumerate(bounded_model.nodes()):
        node_type = type(node).__name__

        required_types = ['Input', 'Linear']
        assert k>0 or 'Input' in node_type, f"Expected Input node at index 0, got {node_type}"
        if any(el in node_type for el in required_types):
            
            if hasattr(node, 'lower') and hasattr(node, 'upper'):
                lower = node.lower
                upper = node.upper
                
                if lower is not None and upper is not None:
                    lower_cpu = lower.squeeze().detach().cpu()
                    upper_cpu = upper.squeeze().detach().cpu()
                    # if k == 0:
                    #     lower_cpu = torch.clamp(lower_cpu, min=0)
                    preact_bounds[k] = (lower_cpu, upper_cpu)
                    
                    L.append(lower_cpu.tolist())
                    U.append(upper_cpu.tolist())
                
                elif lower is not None and upper is None:
                    layer_relu_before = node.inputs[0]
                    layer_linear_before = layer_relu_before.inputs[0]
                    layer_before_upper = layer_linear_before.upper
                    layer_before_lower = layer_linear_before.lower  
                    device_ = layer_before_upper.device
                    dtype = layer_before_upper.dtype
                    
                    W_pos = torch.clamp(torch.tensor(self.W[k-1], device=device_, dtype=dtype), min=0)
                    W_neg = torch.clamp(torch.tensor(self.W[k-1], device=device_, dtype=dtype), max=0)
                    bias = torch.tensor(self.b[k-1], device=device_, dtype=dtype).unsqueeze(dim=1)
                    estimated_upper = W_pos @ torch.transpose(layer_before_upper, 0, 1) + W_neg @ torch.transpose(layer_before_lower, 0, 1) + bias

                    preact_bounds[k] = (
                        lower.squeeze().detach().cpu(),
                        estimated_upper.squeeze().detach().cpu()
                    )
                    L.append(lower.squeeze().detach().cpu().tolist())  
                    U.append(estimated_upper.squeeze().detach().cpu().tolist())


        if 'Relu' in node_type or 'Input' in node_type:
            k += 1



    if preact_bounds:
        for k, (l, u) in preact_bounds.items():
            pass
    else:
        pass
        
    self.L = L
    self.U = U

    self.compute_bounds_time = None

    return

def compute_bounds_data(
    network, x, epsilon, n, K,
    method: str = "IBP",
    norm: str = "Linf",
    n_runs: int = 1,
):
    """
    Compute the  L and U

    Args:
        method (str): The method to compute the bounds (CROWN, IBP, Linear, etc.).
        n_runs (int): Number of independent alpha-CROWN runs. Best-of-N is kept:
                      L_final[k][j] = max over runs, U_final[k][j] = min over runs.
                      Ignored (forced to 1) for deterministic methods (IBP, etc.).
    """
    logger.debug(f"Computing bounds with method: {method}, n_runs={n_runs} ...")

    if method == "GREAT_BOUNDS":
        L = [[-np.inf] * n[k] for k in range(K + 1)]
        U = [[np.inf] * n[k] for k in range(K + 1)]
        L[0] = [max(L[0][j], 0) for j in range(len(L[0]))]
        return L, U

    if not torch.is_tensor(x):
        x = torch.Tensor(x)

    x = x.type(torch.float).view(-1).unsqueeze(0).to(device)

    network = network.to(device)
    network.eval()
    zeros = torch.zeros_like(x).to(device)

    if norm == "Linf":
        ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    elif norm == "L2":
        ptb = PerturbationLpNorm(norm=2, eps=epsilon)
    else:
        raise NotImplementedError(f"Norm {norm} not implemented.")
    bounded_image = BoundedTensor(x, ptb)

    is_crown = method == "alpha-CROWN"
    
    actual_n_runs = n_runs if is_crown else 1

    best_L = None
    best_U = None

    for run_idx in range(actual_n_runs):
        
        
        try:
            bounded_model = BoundedModule(
                network,
                zeros,
                bound_opts={"conv_mode": "patches"},
            )
        except Exception as e:
            raise Exception("Error creating BoundedModule:", e)
        bounded_model.eval()

        if is_crown:
            _, _ = bounded_model.compute_bounds(x=(bounded_image,), method=method)
        else:
            with torch.no_grad():
                _, _ = bounded_model.compute_bounds(x=(bounded_image,), method=method)

        intermediate_bounds = bounded_model.save_intermediate()

        intermediate_bounds_list = list(intermediate_bounds.keys())
        layers_name = {intermediate_bounds_list[0]: 0}
        for k in range(1, K + 1):
            layers_name[intermediate_bounds_list[1 + (k - 1) * 2]] = k
        layers_name[intermediate_bounds_list[-1]] = K

        L_run = [[-np.inf] * n[k] for k in range(K + 1)]
        U_run = [[np.inf] * n[k] for k in range(K + 1)]

        for layer_name, (min_tensor, max_tensor) in intermediate_bounds.items():
            if layer_name not in layers_name:
                logger.debug(f"Layer {layer_name} not found in layers_name mapping.")
                continue
            k = layers_name[layer_name]
            if k == 0:
                min_tensor = torch.clamp(min_tensor, min=0).view(-1)
                max_tensor = max_tensor.view(-1)
            L_run[k] = min_tensor.squeeze().detach().cpu().numpy().tolist()
            U_run[k] = max_tensor.squeeze().detach().cpu().numpy().tolist()

        if best_L is None:
            best_L, best_U = L_run, U_run
        else:
            
            best_L = [
                [max(best_L[k][j], L_run[k][j]) for j in range(n[k])]
                for k in range(K + 1)
            ]
            best_U = [
                [min(best_U[k][j], U_run[k][j]) for j in range(n[k])]
                for k in range(K + 1)
            ]

        if actual_n_runs > 1:
            total_L = sum(sum(best_L[k]) for k in range(K + 1))


    best_L = round_list_depth_2(best_L)
    best_U = round_list_depth_2(best_U)

    return best_L, best_U


def compute_bounds_(self, method: str = "IBP"):
    """
    Compute the  L and U

    Args:
        method (str): The method to compute the bounds (CROWN, IBP, Linear, etc.).
    """
    logger.debug("Computing bounds with norm: ", self.norm, " ...")
    start_compute_bd_time = time.time()
    if method == "IBP":
        self.compute_IBP()
    elif method == "beta-CROWN" or method == "alpha-CROWN":
        self.compute_bounds_data_crown(
                method="alpha-beta-CROWN",
            )
    else:
        n_runs = getattr(self, "bounds_n_runs", 1)
        L, U = compute_bounds_data(
            self.network, self.x, self.epsilon, self.n, self.K,
            method=method, norm=self.norm, n_runs=n_runs,
        )
        self.L = L
        self.U = U

    end_compute_bd_time = time.time()
    
    self.compute_bounds_time = end_compute_bd_time - start_compute_bd_time


def check_stability_neurons(
    self, use_active_neurons: bool = False, use_inactive_neurons: bool = False
):
    """
    Check the stability of neurons in the network.
    """
    logger.debug("Checking stability of neurons ...")
    
    self.stable_inactives_neurons = []
    self.stable_actives_neurons = []
    for k in range(1, self.K):
        
        for j in range(self.n[k]):

            if self.L[k][j] <= 0 and self.U[k][j] <= 0 and not use_inactive_neurons:
                self.stable_inactives_neurons.append((k, j))
            elif self.L[k][j] >= 0 and self.U[k][j] > 0 and not use_active_neurons:
                if (k==self.K - 1 and self.keep_penultimate_actives) :
                    continue
                if (k==1) and len(self.pruned_input_neurons) > 0:
                    continue  # expansion of stable actives at k=1 may reference pruned z_0 neurons
                if k > self.ultimate_layer_use_active_neurons: 
                    continue
                else : 
                    self.stable_actives_neurons.append((k, j))
    self.stable_active_neurons = set(self.stable_actives_neurons)
    self.stable_inactive_neurons = set(self.stable_inactives_neurons)
    logger.debug("stable active neurons : ", self.stable_active_neurons)
    logger.debug("stable inactive neurons : ", self.stable_inactive_neurons)


def prune_adversarial_targets(self):
    """
    Prune the adversarial targets based on the bounds : targets with upper bound lower than other target's lower bound is removed from the adversarial target.
    """
    for j in list(self.ytargets):

        if j == self.ytrue:
            continue
        if self.U[self.K][j] <= self.L[self.K][self.ytrue]:
            self.ytargets.remove(j)
        elif any(
            self.U[self.K][j] < self.L[self.K][j2] for j2 in self.ytargets if j2 != j
        ):
            self.ytargets.remove(j)
        else:
            continue



def compute_IBP(self):

    lb = self.x - self.epsilon
    ub = self.x + self.epsilon

    L, U = [lb.detach().cpu().tolist()], [ub.detach().cpu().tolist()]

    for W, b in zip(self.W, self.b):
        W = torch.tensor(W, device = self.x.device, dtype = self.x.dtype)
        b = torch.tensor(b, device = self.x.device, dtype = self.x.dtype)
        W_pos = torch.maximum(W, torch.zeros_like(W))
        W_neg = torch.minimum(W, torch.zeros_like(W))

        z_l = W_pos @ lb + W_neg @ ub + b
        z_u = W_pos @ ub + W_neg @ lb + b

        L.append(z_l.detach().cpu().tolist())
        U.append(z_u.detach().cpu().tolist())

        lb = torch.maximum(z_l, torch.zeros_like(z_l))
        ub= torch.maximum(z_u, torch.zeros_like(z_u))

    self.L = L
    self.U = U
    
