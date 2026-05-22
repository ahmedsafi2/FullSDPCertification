"""
Statistics on W_1 (first layer weight matrix) for 9x100 and 9x200 networks.

Two angles:
  - Raw coefficients W_1[j][i]  (shape: n_1 × n_0)
  - Column norms  col_norm[i] = Σ_j |W_1[j][i]|  (shape: n_0)
    → used as selection criterion for partial INPUT_IN_VARIABLES

Usage:
    conda activate certif
    python tests/stats_W1_input_neurons.py
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in (SRC_PATH, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from networks.network import ReLUNN

MODELS = {
    "9x100": "data/models/mnist_adv_9x100.pt",
    "9x200": "data/models/mnist_adv_9x200.pt",
    "6x100": "data/models/mnist_adv_6x100.pt",
    "6x200": "data/models/mnist_adv_6x200.pt",
}

PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def print_stats(label: str, values: np.ndarray) -> None:
    print(f"\n  [{label}]  (n={values.size})")
    print(f"    min    = {values.min():.6f}")
    print(f"    max    = {values.max():.6f}")
    print(f"    mean   = {values.mean():.6f}")
    print(f"    median = {np.median(values):.6f}")
    print(f"    std    = {values.std():.6f}")
    pcts = np.percentile(values, PERCENTILES)
    for p, v in zip(PERCENTILES, pcts):
        print(f"    p{p:02d}    = {v:.6f}")


for name, rel_path in MODELS.items():
    path = os.path.join(PROJECT_ROOT, rel_path)
    net = ReLUNN.from_pth(path)
    W1 = np.array(net.W[0])   # shape (n_1, n_0)
    n1, n0 = W1.shape

    print(f"\n{'='*60}")
    print(f"Network: {name}   W_1 shape: {n1} × {n0}")
    print(f"{'='*60}")

    # --- Raw coefficients ---
    print_stats("raw coefficients  W_1[j][i]", W1.ravel())
    print_stats("absolute values  |W_1[j][i]|", np.abs(W1).ravel())

    # --- Column norms (selection criterion for partial INPUT_IN_VARIABLES) ---
    col_norms = np.sum(np.abs(W1), axis=0)   # shape (n_0,)
    print_stats("column norms  Σ_j |W_1[j][i]|  (per input neuron)", col_norms)

    # --- Extra: how many neurons would be kept for various proportions? ---
    print("\n  [neurons kept vs. proportion p]")
    for p in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]:
        n_keep = max(1, int(np.ceil(p * n0)))
        threshold = np.sort(col_norms)[-n_keep]
        print(f"    p={p:.2f}  →  {n_keep:4d}/{n0} neurons kept  "
              f"(col_norm threshold ≥ {threshold:.4f})")
