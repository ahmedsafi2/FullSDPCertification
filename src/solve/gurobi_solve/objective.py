import gurobipy as gp
from gurobipy import GRB
from typing import List


def add_objective_diff(self):
    """Objectif zKyrtue - sum(zKj) for j != ytrue"""
    if self.LAST_LAYER:
        raise NotImplementedError(
            "Objective zKyrtue - sum(zKj) for j != ytrue is not implemented for the last layer."
        )
    else:
        self.m.setObjective(
            (
                gp.quicksum(
                    self.W[self.K - 1][self.ytrue][i] * self.z[self.K - 1, i]
                    for i in range(self.n[self.K - 1]) if (self.K - 1, i) not in self.stable_inactives_neurons
                )
                + self.b[self.K - 1][self.ytrue]
                - gp.quicksum(
                    gp.quicksum(
                        self.W[self.K - 1][j][i] * self.z[self.K - 1, i]
                        for i in range(self.n[self.K - 1]) if (self.K - 1, i) not in self.stable_inactives_neurons
                    )
                    + self.b[self.K - 1][j]
                    for j in self.ytargets
                    if j != self.ytrue
                )
            ),
            GRB.MINIMIZE,
        )


def add_objective_Lan(self):
    """Objectif Advytarget"""
    if self.LAST_LAYER:
        raise NotImplementedError(
            "Objective Advytarget is not implemented for the last layer."
        )
    else:
        self.m.setObjective(
            gp.quicksum(
                self.W[self.K - 1][self.ytrue][i] * self.z[self.K - 1, i]
                for i in range(self.n[self.K - 1]) if (self.K - 1, i) not in self.stable_inactives_neurons
            )
            - (
                gp.quicksum(
                    self.W[self.K - 1][self.ytarget][i] * self.z[self.K - 1, i]
                    for i in range(self.n[self.K - 1]) if (self.K - 1, i) not in self.stable_inactives_neurons
                )
            ),
            GRB.MINIMIZE,
        )
        self.constant = (
            self.b[self.K - 1][self.ytrue] - self.b[self.K - 1][self.ytarget]
        )


def add_objective_Md(self):
    """Objectif Md"""
    if self.LAST_LAYER:
        raise NotImplementedError("Objective Md is not implemented for the last layer.")
    else:
        print("Stable inactive neurons in objective:", self.stable_inactives_neurons)
        self.m.setObjective(
            (
                gp.quicksum(
                    self.W[self.K - 1][self.ytrue][i] * self.z[self.K - 1, i]
                    for i in range(self.n[self.K - 1]) if (self.K - 1, i) not in self.stable_inactives_neurons 
                )
                - gp.quicksum(
                    (
                        gp.quicksum(
                            self.W[self.K - 1][j][i]
                            * self.z[self.K - 1, i]
                            * self.beta[j]
                            for i in range(self.n[self.K - 1])
                            if (self.K - 1, i) not in self.stable_inactives_neurons
                        )
                        + self.b[self.K - 1][j] * self.beta[j]
                    )
                    for j in self.ytargets
                    if j != self.ytrue
                )
            ),
            GRB.MINIMIZE,
        )

        self.constant = self.b[self.K - 1][self.ytrue]


def add_objective_zbar(self):
    """Objective zKytrue - zbar"""
    if self.LAST_LAYER:
        raise NotImplementedError(
            "Objective zKytrue - zbar is not implemented for the last layer."
        )
    else:
        self.m.setObjective(
            (
                gp.quicksum(
                    self.W[self.K - 1][self.ytrue][i] * self.z[self.K - 1, i]
                    for i in range(self.n[self.K - 1])
                )
                - self.zbar
            ),
            GRB.MINIMIZE,
        )
        self.constant = self.b[self.K - 1][self.ytrue]



def add_objective_bound_layer(self):
    obj_sense = GRB.MINIMIZE if self.bound_obj == "lower" else GRB.MAXIMIZE
    self.m.setObjective(
            (
                gp.quicksum(
                    self.W[self.layer_obj - 1][self.neuron_obj][i] * self.z[self.layer_obj - 1, i]
                    for i in range(self.n[self.layer_obj - 1])
                )
            ),
            obj_sense,
        )
    self.constant = self.b[self.layer_obj - 1][self.neuron_obj]