from typing import List
from fastsdp_tools import summing_values_two_dicts, change_to_zero_negative_values
from collections import Counter


class NeuronLinearization :
    """
    A class to handle the values of neurons (in particular stable active neurons) accros layers
    """

    def __init__(
        self,
        K: int,
        n: List[int],
        W: list,
        b: list,
        stable_inactives_neurons: List[tuple] = [],
        stable_actives_neurons: List[tuple] = [],
        L: List[List[float]] = None,
        U: List[List[float]] = None,
        **kwargs,
    ):
        """
        Initialize the NeuronLinearization  class.
        """
        self.n = n
        self.W = W
        self.b = b
        self.K = K
        self.stable_inactives_neurons = stable_inactives_neurons
        self.stable_actives_neurons = stable_actives_neurons
        self.LAST_LAYER = kwargs.get("LAST_LAYER", False)
        self.keep_actives_penultimate = kwargs.get("keep_penultimate_actives", None)
        assert (
            self.keep_actives_penultimate is not None
        ), "keep_penultimate_actives must be specified."

        self.equivalent_values_layers = {
            (layer, neuron): {"neurons_weight": {}, "constant": 0}
            for layer in range(K + 1)
            for neuron in range(n[layer])
        }
        for k in range(K + 1):
            for j in range(n[k]):
                self.add_equivalent_values(k, j)

    def add_equivalent_values(self, layer: int, neuron: int):
        
        if ( ((layer, neuron) in self.stable_actives_neurons
            and (not self.keep_actives_penultimate or layer < self.K - 1)) or (layer == self.K and not self.LAST_LAYER)):
         
            self.equivalent_values_layers[(layer, neuron)]["constant"] += self.b[
                layer - 1
            ][neuron]
            
            for i in range(self.n[layer - 1]):

                self.equivalent_values_layers[(layer, neuron)]["neurons_weight"] = (
                    summing_values_two_dicts(
                        self.equivalent_values_layers[(layer, neuron)][
                            "neurons_weight"
                        ],
                        {
                            (layer2, neuron2): (value * self.W[layer - 1][neuron][i])
                            for (
                                layer2,
                                neuron2,
                            ), value in self.equivalent_values_layers[(layer - 1, i)][
                                "neurons_weight"
                            ].items()
                        },
                    )
                )
                self.equivalent_values_layers[(layer, neuron)]["constant"] += (
                    self.equivalent_values_layers[(layer - 1, i)]["constant"]
                    * self.W[layer - 1][neuron][i]
                )


           

            coordinates = [
                (layer, neuron)
                for (layer, neuron), value in self.equivalent_values_layers[
                    (layer, neuron)
                ]["neurons_weight"].items()
            ]
            counts = Counter(coordinates)

        elif (layer, neuron) in self.stable_inactives_neurons:
            pass
        elif layer == self.K and self.LAST_LAYER:
            self.equivalent_values_layers[(layer, neuron)]["neurons_weight"] = {
                (layer, neuron): 1
            }

        else:
            self.equivalent_values_layers[(layer, neuron)]["neurons_weight"] = {
                (layer, neuron): 1
            }

    def get_equivalent_values(self, layer: int, neuron: int):
        """
        Get the equivalent values for a given layer and neuron.
        """
        if layer < 0 or layer > self.K:
            raise ValueError(f"Layer {layer} is out of bounds (0 to {self.K}).")
        if neuron < 0 or neuron >= self.n[layer]:
            raise ValueError(
                f"Neuron {neuron} in layer {layer} is out of bounds (0 to {self.n[layer] - 1})."
            )
        return (
            self.equivalent_values_layers[(layer, neuron)]["neurons_weight"],
            self.equivalent_values_layers[(layer, neuron)]["constant"],
        )

    def is_unstable(self, layer: int, neuron: int) -> bool:
        """
        Check if the neuron is a stable active neuron.
        """
        return (layer, neuron) not in (
            self.stable_actives_neurons + self.stable_inactives_neurons
        ) and (layer is not None and neuron is not None)

    def is_stable_active(self, layer: int, neuron: int) -> bool:
        """
        Check if the neuron is a stable active neuron.
        """
        return (layer, neuron) in self.stable_actives_neurons and (
            layer is not None and neuron is not None
        )

    def computing_bounds_based_on_stable_neurons(
        self,
        L: List[List[float]] = None,
        U: List[List[float]] = None,
    ):
        """
        Compute the bounds based on stable neurons.
        """
        for k in range(self.K + 1):
            for j in range(self.n[k]):
                if (k, j) in self.stable_actives_neurons:
                    upper_bounds = (
                        sum(
                            value * U[layer][neuron]
                            for (layer, neuron), value in self.equivalent_values_layers[k, j][
                                "neurons_weight"
                            ].items()
                            if value > 0
                        )
                        + sum(
                            value * L[layer][neuron]
                            for (layer, neuron), value in self.equivalent_values_layers[k, j][
                                "neurons_weight"
                            ].items()
                            if value < 0
                        )
                        + self.equivalent_values_layers[k, j]["constant"]
                    )
                    if upper_bounds < U[k][j]:
                        U[k][j] = upper_bounds
                    else:
                        pass
                    # print(f"Upper bound for layer {k}, neuron {j}: {self.upper_bounds[(k, j)]} and U = {U[k][j]}")
                    lower_bounds = (
                        sum(
                            value * L[layer][neuron]
                            for (layer, neuron), value in self.equivalent_values_layers[k, j][
                                "neurons_weight"
                            ].items()
                            if value > 0
                        )
                        + sum(
                            value * U[layer][neuron]
                            for (layer, neuron), value in self.equivalent_values_layers[k, j][
                                "neurons_weight"
                            ].items()
                            if value < 0
                        )
                        + self.equivalent_values_layers[k, j]["constant"]
                    )
                    if lower_bounds > L[k][j]:
                        L[k][j] = lower_bounds
                    else:
                        pass
                    
        return L, U
