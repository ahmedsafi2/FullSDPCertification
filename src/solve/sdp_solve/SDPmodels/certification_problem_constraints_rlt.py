import numpy as np

from .certification_problem_constraints_bounds import McCormick_inter_layers
from fastsdp_tools import get_m_indexes_of_higher_values_in_list


# ********************************* RLT (Lan, 2022) Constraints *********************************
def add_RLT_constraints(self, p: float = 0.5):
    """
    Add the RLT constraints to the task.
    """
    nb_rlt = 0
    
    print("Adding RLT constraint")
    start_k = 2 if not self.INPUT_IN_VARIABLES else 1
    for k in range(start_k, self.K + 1 if self.LAST_LAYER else self.K):
        nb_cstr = int(p * self.n[k - 1])
        
        indexes_pruned = [
            j
            for j in range(self.n[k - 1])
            if (k - 1, j) in self.stable_inactives_neurons
            or (k - 1, j) in self.stable_actives_neurons
            or (k == 1 and j in self.pruned_input_neurons)
        ]
        if k == self.K and self.LAST_LAYER:
            neurons_next = list(set([self.ytrue]).union(self.ytargets))
        else:
            neurons_next = [j for j in range(self.n[k]) 
                            if (k, j) not in self.stable_actives_neurons 
                            and (k, j) not in self.stable_inactives_neurons]
        
        study_ablation =  True
        for neuron_next in neurons_next:
            if (k, neuron_next) in self.stable_inactives_neurons:
                
                continue
            if (k, neuron_next) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or k != self.K - 1
            ):
                
                continue
            neurons_with_great_weights = get_m_indexes_of_higher_values_in_list(
                np.abs(self.W[k - 1][neuron_next]), nb_cstr, indexes_pruned
            )
            for neuron_prev in neurons_with_great_weights:

                assert (k - 1, neuron_prev) not in self.stable_inactives_neurons and (
                    k - 1,
                    neuron_prev,
                ) not in self.stable_actives_neurons

                self.McCormick_inter_layers(k, neuron_prev, neuron_next)
                nb_rlt+=1
    print(f"Number of RLT constraints = {nb_rlt}")    