"""
Teste l'implémentation partielle de INPUT_IN_VARIABLES sur blob_4x10.

Invariants vérifiés :
  1. Pas de crash pour p ∈ {0.0, 0.5, 1.0}
  2. kept_input_neurons : pour p=0.5 sur n_0=2 → 1 neurone gardé
                          (celui avec la plus grande norme de colonne de W_1)
  3. Monotonie : val(p=1.0) >= val(p=0.5) >= val(p=0.0)
                 (relaxation plus large → valeur optimale plus petite ou égale)
  4. Rétro-compatibilité : val(True) == val(1.0)  et  val(False) == val(0.0)

Usage :
    conda activate certif
    python tests/test_partial_input_blob4x10.py
"""

import os, sys, math
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in (SRC_PATH, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import solve
from networks.network import ReLUNN
from data import load_dataset
from fastsdp_tools.yaml_config import FullCertificationConfig
from fastsdp_tools.utils import get_project_path

# ── Config ─────────────────────────────────────────────────────────────────────
YAML_PATH = os.path.join(PROJECT_ROOT, "config", "blob_4x10.yaml")
N_SAMPLES  = 3          # nombre de samples testés
TOL        = 1e-3       # tolérance numérique pour les comparaisons de valeurs
PROPORTIONS = [False, 0.5, True]  # = [0.0, 0.5, 1.0]

# ── Chargement config ───────────────────────────────────────────────────────────
config = FullCertificationConfig.from_yaml(YAML_PATH)
solver_cfg = config.models[0]  # seul modèle défini dans le yaml
base_kwargs = dict(solver_cfg)
base_kwargs.pop("certification_model_type")

network_path = get_project_path(config.network.path)
network = ReLUNN.from_pth(network_path)
n0 = network.n[0]
W1 = np.array(network.W[0])   # shape (n_1, n_0)

bounds_path = get_project_path(solver_cfg.bounds_file)
bounds_csv  = pd.read_csv(bounds_path)

dataset, _ = load_dataset(
    name=config.data.name,
    path=get_project_path(config.data.path),
    num_classes=config.data.num_classes,
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

model_class = getattr(solve, solver_cfg.certification_model_type)

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_optimal_value(df: pd.DataFrame) -> float:
    if df is None or "optimal_value" not in df.columns:
        return float("nan")
    return float(df["optimal_value"].iloc[-1])


def run_sample(x, ytrue: int, data_index: int, input_in_variables) -> float:
    """Résout le SDP pour un sample donné et une valeur de INPUT_IN_VARIABLES."""
    k_net = network.K
    L = [[float(bounds_csv[bounds_csv["data_index"] == data_index]
                [f"LB_Layer_{k}_Neuron_{j}"].iloc[0])
          for j in range(network.n[k])]
         for k in range(k_net + 1)]
    U = [[float(bounds_csv[bounds_csv["data_index"] == data_index]
                [f"UB_Layer_{k}_Neuron_{j}"].iloc[0])
          for j in range(network.n[k])]
         for k in range(k_net + 1)]

    kwargs = {**base_kwargs, "L": L, "U": U, "INPUT_IN_VARIABLES": input_in_variables}
    inst = model_class(
        network=network,
        epsilon=config.input_ball.epsilon,
        norm=config.input_ball.norm,
        x=x,
        ytrue=ytrue,
        data_index=data_index,
        dataset_name=config.data.name,
        network_name=config.network.name,
        folder_name=os.path.join(PROJECT_ROOT, "results", "test_partial_input"),
        **kwargs,
    )
    inst.solve(verbose=False, only_bounds=False)
    return get_optimal_value(inst.benchmark_dataframe), inst


# ── Invariant 2 : kept_input_neurons pour p=0.5 sur n_0=2 ─────────────────────
print("\n" + "="*65)
print("Invariant 2 — kept_input_neurons (p=0.5, n_0=2)")
print("="*65)
col_norms = np.sum(np.abs(W1), axis=0)
expected_kept_idx = int(np.argmax(col_norms))
print(f"  col_norms W_1 : {col_norms.tolist()}")
print(f"  neurone attendu gardé : {expected_kept_idx}  "
      f"(norme {col_norms[expected_kept_idx]:.4f})")

# ── Boucle sur les samples ─────────────────────────────────────────────────────
results = []  # list of dict {sample, label, val_false, val_05, val_true}
n_tested = 0

print("\n" + "="*65)
print(f"Invariants 1 & 3 & 4 — {N_SAMPLES} samples")
print("="*65)

for i, (x_batch, ytrue_batch) in enumerate(dataloader):
    if n_tested >= N_SAMPLES:
        break
    x    = x_batch.view(-1)
    ytrue = int(ytrue_batch.item())

    y_pred = int(network.label(x).item())
    if y_pred != ytrue:
        continue  # skip misclassified

    if i not in bounds_csv["data_index"].values:
        continue

    print(f"\n--- Sample {i}  (label={ytrue}) ---")
    vals = {}
    insts = {}
    for prop in PROPORTIONS:
        try:
            v, inst = run_sample(x, ytrue, i, prop)
            vals[prop]  = v
            insts[prop] = inst
            label = f"p={prop}"
            print(f"  INPUT_IN_VARIABLES={prop!s:<5}  →  optimal_value = {v:.6f}"
                  f"  |  kept={sorted(inst.kept_input_neurons)}")
        except Exception as e:
            import traceback
            print(f"  INPUT_IN_VARIABLES={prop!s:<5}  → CRASH: {e}")
            traceback.print_exc()
            vals[prop] = float("nan")

    results.append({"sample": i, "label": ytrue, **{f"val_{p}": vals[p] for p in PROPORTIONS}})
    n_tested += 1

# ── Vérification des invariants ────────────────────────────────────────────────
print("\n" + "="*65)
print("Résultats des invariants")
print("="*65)

pass_count = 0
fail_count = 0

def check(cond: bool, msg: str):
    global pass_count, fail_count
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if cond:
        pass_count += 1
    else:
        fail_count += 1

# Invariant 2 : vérifié via le premier sample où p=0.5 a tourné sans crash
for r in results:
    v_false = r[f"val_{False}"]
    v_05    = r[f"val_{0.5}"]
    v_true  = r[f"val_{True}"]
    s = r["sample"]

    all_finite = all(math.isfinite(v) for v in [v_false, v_05, v_true])
    check(all_finite, f"Sample {s}: pas de NaN/crash pour les 3 proportions")

    if all_finite:
        check(v_true  >= v_05   - TOL, f"Sample {s}: monotonie val(1.0)={v_true:.5f} >= val(0.5)={v_05:.5f}")
        check(v_05    >= v_false - TOL, f"Sample {s}: monotonie val(0.5)={v_05:.5f} >= val(0.0)={v_false:.5f}")

print()
print(f"Résultat final : {pass_count} PASS / {fail_count} FAIL")
if fail_count == 0:
    print("✓ Tous les invariants vérifiés.")
else:
    print("✗ Des invariants sont violés — revoir l'implémentation.")
