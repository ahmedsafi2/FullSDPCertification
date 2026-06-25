def add_z_quad_active_neuron_heuristic(
                self,
                layer_prev : int,
                neuron_prev : int,
                layer_next : int, 
                neuron_next : int,
                front_of_matrix_prev : bool,
                front_of_matrix_next : bool,
                weight : float,
                bound_sense : str,
                mccormick_type : str,
            ):
    self.handler.Constraints.add_z_quad_active_neuron(
                layer_prev=layer_prev,
                neuron_prev=neuron_prev,
                layer_next=layer_next,
                neuron_next=neuron_next,
                front_of_matrix_prev=True,
                front_of_matrix_next=False,
                weight=weight,
                bound_sense=bound_sense,
                mccormick_type=mccormick_type,
            )
    


def quadratic_constraint_heuristic(
            self,
            k : int,
            j : int,
            heuristic_choice : str            
):
    assert heuristic_choice in ['RANDOM','4_DETERMINISTIC_CONSTRAINTS']
    if heuristic_choice == 'RANDOM':
        self.quadratic_constraint_heuristic_RANDOM(k,j,nb_constraints_random = 8)
    elif heuristic_choice == '4_DETERMINISTIC_CONSTRAINTS':
        self.quadratic_constraint_heuristic_4_DETERMINISTIC_CONSTRAINTS(k,j)
    # A COMPLETER AVEC DE MEILLEURES HEURISTIQUES
    
                




def quadratic_constraint_heuristic_4_DETERMINISTIC_CONSTRAINTS(
            self,
            k : int,
            j : int):
                    
    self.ReLU_constraint_stable_active_relaxation(
        k, j, bound_sense="upper", mccormick_type="one_variable"
    )
    self.ReLU_constraint_stable_active_relaxation(
        k, j, bound_sense="lower", mccormick_type="one_variable"
    )
    self.ReLU_constraint_stable_active_relaxation(
        k, j, bound_sense="upper", mccormick_type="composed"
    )
    self.ReLU_constraint_stable_active_relaxation(
        k, j, bound_sense="lower", mccormick_type="composed"
    )


def quadratic_constraint_heuristic_RANDOM(
            self,
            k : int,
            j : int,
            nb_constraints_random : int            
):
                
    for i in range(nb_constraints_random):
        self.ReLU_constraint_stable_active_relaxation(
            k, j, bound_sense="upper", mccormick_type="random", name = f"random_{i}"
        )
        self.ReLU_constraint_stable_active_relaxation(
            k, j, bound_sense="lower", mccormick_type="random", name = f"random_{i}"
        )