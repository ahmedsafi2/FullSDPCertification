import torch.nn as nn
import torch
import sys
import os
import yaml
from fastsdp_tools import get_project_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.jit.ignore(drop=True)
class ReLUNN(nn.Module):
    def __init__(self, K, n, W=None, b=None, dropout_prob: float = 0, name="ReLUNN"):

        super(ReLUNN, self).__init__()
        self.name = name
        self.K = K
        self.W = W
        self.b = b
        self.n = n


        self.layers = nn.ModuleDict()
        self.penultimate_layer = "Layer_" + str(K - 1) + "_Linear"
        for k in range(1, K + 1):
            layer_lin = nn.Linear(in_features=self.n[k - 1], out_features=self.n[k])
            layer_lin_name = f"Layer_{k}_Linear"
            if W is not None:
                layer_lin.weight = nn.Parameter(
                    torch.tensor(W[k - 1], dtype=torch.float32).clone().detach(),
                    requires_grad=False,
                )
                layer_lin.bias = nn.Parameter(
                    torch.tensor(b[k - 1], dtype=torch.float32).clone().detach(),
                    requires_grad=False,
                )
            self.layers[layer_lin_name] = layer_lin

            if k < K:
                layer_relu = nn.ReLU(inplace=True)
                layer_relu_name = f"Layer_{k}_ReLU"
                self.layers[layer_relu_name] = layer_relu
                if dropout_prob > 0:
                    self.layers[f"Layer_{k}_Dropout"] = nn.Dropout(p=dropout_prob)
                
        if W is None:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def extract_weights(self):

        W = []
        b = []

        for k in range(1, self.K + 1):
            layer_name = f"Layer_{k}_Linear"
            layer = self.layers[layer_name]
            W.append(layer.weight.data.tolist())
            b.append(layer.bias.data.tolist())

        self.W = W
        self.b = b
        return W, b

    @classmethod
    def from_yaml(cls, yaml_file):

        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
            K = config["network"]["K"]
            n = config["network"]["n"]
            

            path = get_project_path(config["network"]["path"].replace("\\", "/"))

            parametres = torch.load(path)


            W = []
            b = []
            n = []

            for k in range(1, K + 1):
                weight_k = f"layers.Layer_{k}_Linear.weight"
                bias_k = f"layers.Layer_{k}_Linear.bias"
                in_features = parametres[weight_k].shape[1]
                out_features = parametres[weight_k].shape[0]

                n.append(in_features)
                W.append(parametres[weight_k].tolist())
                b.append(parametres[bias_k].tolist())

            out_features = parametres[weight_k].shape[0]
            n.append(out_features)
        return cls(K, n, W, b)
    
    @classmethod
    def from_pth(cls, pth_path, dropout_prob=0, bb_beta_crown:bool = False):
      
        parametres = torch.load(pth_path, map_location="cpu")
        if bb_beta_crown and 'state_dict' in parametres:
            parametres = parametres['state_dict'][0]

        W, b, n = [], [], []

        k = 1
        for key in parametres.keys():
            if key.endswith(".weight"):
                weight = parametres[key]
                bias = parametres[key.replace(".weight", ".bias")]

                out_features, in_features = weight.shape

                if not n:
                    n.append(in_features)
                n.append(out_features)

                W.append(weight.tolist())
                b.append(bias.tolist())

        K = len(W)
        return cls(K=K, n=n, W=W, b=b, dropout_prob=dropout_prob)


    def forward(self, x, return_last_hidden=False):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor.
            return_last_hidden (bool): If True, return the output of the last hidden layer.
        """

        if x.dim() >= 4:
            x = x.view(x.size(0), -1)

        for layer_name, layer in self.layers.items():
            x = layer(x)
            if layer_name == self.penultimate_layer and return_last_hidden:
                return x

        return x

    def return_values_each_layer(self, x):

        if x.dim() >= 4:
            x = x.view(x.size(0), -1)

        values_layer = []
        for layer_name, layer in self.layers.items():
            x = layer(x)
            if "Linear" in layer_name:
                values_layer.append(x.flatten().detach().cpu().tolist())
        return values_layer

    def label(self, x):
        res = self.forward(x)
        label = torch.argmax(res)
        return label.item()

    def __str__(self, *args, **kwds):
        description = ""
        for layer_name, layer in self.layers.items():
            if isinstance(layer, nn.ReLU):
                description += f"    {layer_name} ReLU : {layer}\n"
            elif isinstance(layer, nn.Dropout):
                description += f"    {layer_name} Dropout : {layer}\n"
            else:
                description += f"    {layer_name} Linear : {layer}\n"
                description += f"    Weights : {layer.weight}\n"
                description += f"    Bias : {layer.bias}\n"
            description += "\n"
        return description




if __name__ == "__main__":
    yaml_file = "config/moon.yaml"
    net = ReLUNN.from_yaml(yaml_file)
