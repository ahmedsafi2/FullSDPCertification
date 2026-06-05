import logging
import torch
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import time

from fastsdp_tools import round_list_depth_2, change_to_zero_negative_values

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def compute_bounds_data_new(network, x, epsilon, n, K, method: str = "IBP", norm : str = "Linf"):
    """
    Compute the  L and U

    Args:
        method (str): The method to compute the bounds (CROWN, IBP, Linear, etc.).
    """
    logger.debug(f"Computing bounds with method: {method} ...")
    print("epsilon : ", epsilon)
    L = [[-np.inf] * n[k] for k in range(K + 1)]
    U = [[np.inf] * n[k] for k in range(K + 1)]

    if method == "GREAT_BOUNDS":
        L[0] = [max(L[0][j], 0) for j in range(len(L[0]))]
        return

    if not torch.is_tensor(x):
        x = torch.Tensor(x)

    x = x.type(torch.float).view(-1).unsqueeze(0).to(device)
    print("x device : ", x.device)
    print("x shape : ", x.shape)

    network = network.to(device)
    print("network device : ", next(network.parameters()).device)
    network.eval()
    print("network is none : ", network is None)

    zeros = torch.zeros_like(x).to(device)
    print("zeros device : ", zeros.device)

    logger.debug("creating BoundedModule ...")
    try:

        print("About to create BoundedModule on device:", device)
        print("network device before BoundedModule:", next(network.parameters()).device)
        print("zeros device before BoundedModule:", zeros.device)
        bounded_model = BoundedModule(
            network,
            zeros,
            bound_opts={"conv_mode": "patches"},
        )
        print("created BoundedModule")
    except Exception as e:
        raise Exception("Error creating BoundedModule:", e)


    bounded_model.eval()
    logger.debug("bounded_model device : ", next(bounded_model.parameters()).device)

    if norm == "Linf":
        logger.debug("Using Linf norm for perturbation.")
        ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    elif norm == "L2":
        logger.debug("pertubation L2 used")
        ptb = PerturbationLpNorm(norm=2, eps=epsilon)
        #ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon**2)  # comparer les deux versions
    else:
        raise NotImplementedError(f"Norm {norm} not implemented.")
    bounded_image = BoundedTensor(x, ptb)
    
    use_grad = method.lower() in ["alpha-crown", "beta-crown", "CROWN-Optimized"]
    if use_grad:
        lb, ub, aux = bounded_model.compute_bounds(x=(bounded_image,), method=method, return_A = True)
        intermediate_bounds = aux["intermediate_bounds"]
    else:
        with torch.no_grad():
            lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method=method)
            intermediate_bounds = bounded_model.save_intermediate()   # Version save_intermediate depreciee
    print("Intermediate bounds : ", intermediate_bounds)

    for name, (lb, ub) in intermediate_bounds.items():
        print(name)


    intermediate_bounds_list = list(intermediate_bounds.keys())

    logger.debug("Intermediate bounds list : ", intermediate_bounds_list)

    layers_name = {}
    layers_name[intermediate_bounds_list[0]] = 0

    logger.debug("Preparing to create bounds...")
    print('Intermediate_bounds_list : ', intermediate_bounds_list )
    for k in range(1, K + 1):
        print(f"Adding layer for k = {k}, num_layer = {1 + (k - 1) * 2}")
        layers_name[intermediate_bounds_list[1 + (k - 1) * 2]] = k    ### !!!!  Before *3 because of the dropout layer  !!!!
    logger.debug("Layers name mapping : ", layers_name)

    logger.debug("Intermediate bounds list final values : ", intermediate_bounds_list[-1])
    layers_name[intermediate_bounds_list[-1]] = K

    for layer_name, (min_tensor, max_tensor) in intermediate_bounds.items():

        if layer_name not in layers_name:
            print(f"Layer {layer_name} not found in layers_name mapping.")
            print(f"  Min: {min_tensor.squeeze().shape}")
            print(f"  Max: {max_tensor.squeeze().shape} \n")
            continue
        print(f"{layer_name}:")
        print(f"  Min: {min_tensor.squeeze().shape}")
        print(f"  Max: {max_tensor.squeeze().shape} \n")
        if layers_name[layer_name] == 0:
            # For the first layer, we set the lower bound to 0
            min_tensor = torch.clamp(min_tensor, min=0).view(-1)
            max_tensor = max_tensor.view(-1)


        L[layers_name[layer_name]] = (
            min_tensor.squeeze().detach().cpu().numpy().tolist()
        )
        U[layers_name[layer_name]] = (
            max_tensor.squeeze().detach().cpu().numpy().tolist()
        )

    L = round_list_depth_2(L)
    U = round_list_depth_2(U)


    return L, U

def compute_bounds_data(network, x, epsilon, n, K, method: str = "IBP", norm : str = "Linf"):
    """
    Compute the  L and U

    Args:
        method (str): The method to compute the bounds (CROWN, IBP, Linear, etc.).
    """
    logger.debug(f"Computing bounds with method: {method} ...")
    print("epsilon : ", epsilon)
    L = [[-np.inf] * n[k] for k in range(K + 1)]
    U = [[np.inf] * n[k] for k in range(K + 1)]

    if method == "GREAT_BOUNDS":
        L[0] = [max(L[0][j], 0) for j in range(len(L[0]))]
        return

    if not torch.is_tensor(x):
        x = torch.Tensor(x)

    x = x.type(torch.float).view(-1).unsqueeze(0).to(device)
    print("x device : ", x.device)
    print("x shape : ", x.shape)

    network = network.to(device)
    print("network device : ", next(network.parameters()).device)
    network.eval()
    print("network is none : ", network is None)

    zeros = torch.zeros_like(x).to(device)
    print("zeros device : ", zeros.device)

    logger.debug("creating BoundedModule ...")
    try:

        # # Vérif optional : assure que tout est bien sur cuda
        # for param in network.parameters():
        #     print("param :  ", param)
        #     print("device :  ", device)
        #     print("param device :  ", param.device)
        #     assert param.device == device
        # print('parameters are on the right device')
        # for buf in network.buffers():
        #     assert buf.device == device
        # print('buffers are on the right device')
        print("About to create BoundedModule on device:", device)
        print("network device before BoundedModule:", next(network.parameters()).device)
        print("zeros device before BoundedModule:", zeros.device)
        bounded_model = BoundedModule(
            network,
            zeros,
            bound_opts={"conv_mode": "patches"},
        )
        print("created BoundedModule")
    except Exception as e:
        raise Exception("Error creating BoundedModule:", e)
 

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device)

    # # S'assurer que le modèle est bien déplacé
    # network.to(device)

    # # Vérifier que tous les paramètres ET buffers sont bien sur le même device
    # for name, param in network.named_parameters():
    #     print(f"STUDY : Parameter {name} is on device: {param.device}")

    # for name, layer in network.layers.items():
    #     print("STUDY : Layer : ", name)
    # # Créer l'entrée zéro sur le bon device
    # zeros = torch.zeros_like(x, device=device)

    # # Et maintenant seulement :
    # bounded_model = BoundedModule(
    #     network,
    #     zeros,
    #     bound_opts={"conv_mode": "patches"},
    # )

    bounded_model.eval()
    logger.debug("bounded_model device : ", next(bounded_model.parameters()).device)

    if norm == "Linf":
        logger.debug("Using Linf norm for perturbation.")
        ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    elif norm == "L2":
        logger.debug("pertubation L2 used")
        ptb = PerturbationLpNorm(norm=2, eps=epsilon)
        #ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon**2)  # comparer les deux versions
    else:
        raise NotImplementedError(f"Norm {norm} not implemented.")
    bounded_image = BoundedTensor(x, ptb)
    if method == "alpha-CROWN":
        lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method=method)
    else:
        with torch.no_grad():
            lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method=method)
    intermediate_bounds = bounded_model.save_intermediate()
    print("Intermediate bounds : ", intermediate_bounds)

    intermediate_bounds_list = list(intermediate_bounds.keys())

    logger.debug("Intermediate bounds list : ", intermediate_bounds_list)

    for layer_name, (min_tensor, max_tensor) in intermediate_bounds.items():
        logger.debug(f"{layer_name}")
        logger.debug(f"shape = {max_tensor.squeeze().detach().cpu().numpy().shape}")
        
        # if self.data_modele == "blob":
        #     print(f"  Min: {min_tensor.squeeze().cpu().numpy()}")
        #     print(f"  Max: {max_tensor.squeeze().cpu().numpy()}")
        # else:
        # print(f"  Min SHAPE: {min_tensor.squeeze().cpu().numpy().shape}")
        logger.debug("Min min : ", min_tensor.min())
        # print("Min max : ", min_tensor.max())
        # print(f"  Max SHAPE: {max_tensor.squeeze().cpu().numpy().shape}")
        # print("Max min : ", max_tensor.min())
        logger.debug("Max max : ", max_tensor.max())

    layers_name = {}
    layers_name[intermediate_bounds_list[0]] = 0

    logger.debug("Preparing to create bounds...")
    print('Intermediate_bounds_list : ', intermediate_bounds_list )
    for k in range(1, K + 1):
        print(f"Adding layer for k = {k}, num_layer = {1 + (k - 1) * 2}")
        layers_name[intermediate_bounds_list[1 + (k - 1) * 2]] = k    ### !!!!  Before *3 because of the dropout layer  !!!!
    logger.debug("Layers name mapping : ", layers_name)

    logger.debug("Intermediate bounds list final values : ", intermediate_bounds_list[-1])
    layers_name[intermediate_bounds_list[-1]] = K

    for layer_name, (min_tensor, max_tensor) in intermediate_bounds.items():

        if layer_name not in layers_name:
            print(f"Layer {layer_name} not found in layers_name mapping.")
            print(f"  Min: {min_tensor.squeeze().shape}")
            print(f"  Max: {max_tensor.squeeze().shape} \n")
            continue
        print(f"{layer_name}:")
        print(f"  Min: {min_tensor.squeeze().shape}")
        print(f"  Max: {max_tensor.squeeze().shape} \n")
        if layers_name[layer_name] == 0:
            # For the first layer, we set the lower bound to 0
            min_tensor = torch.clamp(min_tensor, min=0).view(-1)
            max_tensor = max_tensor.view(-1)

       
        L[layers_name[layer_name]] = (
            min_tensor.squeeze().detach().cpu().numpy().tolist()
        )
        U[layers_name[layer_name]] = (
            max_tensor.squeeze().detach().cpu().numpy().tolist()
        )

    L = round_list_depth_2(L)
    U = round_list_depth_2(U)

    # for k in range(len(L)):
    #     min_layer_diff = 1e10
    #     max_layer_diff = -1e10
    #     min_layer_diff_ecart_relatif = 1e10
    #     min_layer = min(L[k])
    #     max_layer = max(U[k])
    #     for j in range(len(L[k])):
    #         if L[k][j] > U[k][j]:
    #             print(
    #                 f"STUDY : Warning: Inconsistent bounds at layer {k}, neuron {j}: L={L[k][j]} > U={U[k][j]}. Adjusting L to U."
    #             )
    #         else :
    #             if U[k][j] - L[k][j] < min_layer_diff:
    #                 min_layer_diff = U[k][j] - L[k][j]
    #             if U[k][j] - L[k][j] > max_layer_diff:
    #                 max_layer_diff = U[k][j] - L[k][j]
    #             if 2*(U[k][j] - L[k][j]) / (abs(U[k][j]) + abs(L[k][j])) < min_layer_diff_ecart_relatif:
    #                 min_layer_diff_ecart_relatif = 2*(U[k][j] - L[k][j]) / (abs(U[k][j]) + abs(L[k][j]))

    #     print("STUDY : Bounds differences at layer ", k, " : min=", min_layer, ";  max=", max_layer, " : min_diff=", min_layer_diff, ";  max_diff=", max_layer_diff, ";  rel_min=", min_layer_diff_ecart_relatif)

    return L, U


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
        print("OULA bornes alpha claude")
        self.compute_bounds_data_crown(
                method="alpha-beta-CROWN",
            )
    else :
        L, U = compute_bounds_data(
            self.network, self.x, self.epsilon, self.n, self.K, method=method, norm=self.norm
        )
        self.L = L
        self.U = U

    end_compute_bd_time = time.time()
    print("compute bounds time créé dans : ", self.__class__.__name__)
    self.compute_bounds_time = end_compute_bd_time - start_compute_bd_time


def check_stability_neurons(
    self, use_active_neurons: bool = False, use_inactive_neurons: bool = False
):
    """
    Check the stability of neurons in the network.
    """
    logger.debug("Checking stability of neurons ...")
    logger.debug("stable use_active_neurons: ", use_active_neurons)
    logger.debug("stable use_inactive_neurons: ", use_inactive_neurons)
    self.stable_inactives_neurons = []
    self.stable_actives_neurons = []
    # Check if the neurons are stable
    for k in range(1, self.K):
        
        for j in range(self.n[k]):
            # print(
            #     "STUDY : Layer ",
            #     k,
            #     " Neuron ",
            #     j,
            #     " L=",
            #     self.L[k][j],
            #     " U=",
            #     self.U[k][j],
            # )
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
    print(
        "STUDY : Nb Stable neurons : ",
        len(self.stable_active_neurons) + len(self.stable_inactive_neurons),
    )
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
            # print("STUDY : Adversarial target selected : ", j)
            continue


# def compute_ibp(self):
#     "Compute bounds L and U with IBP."
#     L = [[(self.x[i].item() - self.epsilon) for i in range(self.n[0])]]
#     U = [[(self.x[i].item() + self.epsilon) for i in range(self.n[0])]]
#     for k in range(1,self.K+1):
#         print("IBP de la couche ", k)
#         lb_layer = []
#         ub_layer = []
#         for j in range(self.n[k]):
#             print(f"Neuron j = {j}")
#             lb= self.b[k-1][j]
#             ub = self.b[k-1][j]
#             for i in range(self.n[k-1]):
#                 print(f"Neurone precdent i = {i}")
#                 print("len(self.W_k-1) : ", len(self.W[k-1]))
#                 print("len(self.W_k-1[0]) : ", len(self.W[k-1][0]))

#                 w_ij = self.W[k-1][j][i]
#                 print("Poids wij = ", w_ij)
#                 print(f"L = ", L)
#                 if w_ij >= 0:
#                     print("Poids positif")
#                     print(f"L_{k-1} : ", L[k-1])
#                     lb += w_ij * L[k-1][i]
#                     ub += w_ij * U[k-1][i]
#                     print("Rajouté")
#                 else:
#                     print("Poids négatif")
#                     print(f"L_{k-1} : ", L[k-1])
#                     lb += w_ij * U[k-1][i]
#                     ub += w_ij * L[k-1][i]
#                     print("Rajouté")
#             lb_layer.append(lb)
#             ub_layer.append(ub)

#         L.append(lb_layer)
#         U.append(ub_layer)



def compute_IBP(self):
    """
    Calcule les bornes pré-activation (l, u) pour chaque couche
    via IBP en norme infinie (L∞).
    
    Retourne :
        bounds = [(l1, u1), (l2, u2), ..., (lL, uL)]
    """
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
    
    for k in range(self.K+1):
        print(f"Layer {k}, L = {L[k]}, U = {U[k]}")
