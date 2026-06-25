# FastSDPCertification

A library for certifying the adversarial robustness of ReLU neural networks using Semi-Definite Programming (SDP) relaxations, solved with MOSEK (or Gurobi for LP/QP variants).

## Overview

Given a neural network and an input perturbation ball, FastSDPCertification computes a certified lower bound on the minimum adversarial perturbation. It formulates the certification problem as an SDP relaxation and solves it with cutting-plane techniques to tighten the bound.

**Supported architectures:** fully-connected ReLU networks (MLP).  
**Supported norms:** `Linf`, `L2`, `L1`.  
**Solvers:** MOSEK (classic API or Fusion API), with optional CVXPY backend.

## Installation

```bash
pip install -e .
pip install -r requirements.txt
```

A valid MOSEK license is required (`opt/mosek.lic`). Gurobi is optional for LP/QP models.

## Usage

Experiments are driven by YAML configuration files located in `config/`. Launch a benchmark run by pointing to a config file:

```python
from solve.sdp_solve.mosek_generic_solver import run_from_yaml

run_from_yaml("config/mnist-8x1024-0.3.yaml")
```

---

## YAML Configuration Reference

A configuration file has four top-level sections: `input_ball`, `data`, `network`, and `models`. An optional `divide_run` and `conic_solver` section are also available.

### `input_ball`

Defines the perturbation ball around the input.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `norm` | `"Linf"` \| `"L2"` \| `"L1"` | Yes | Norm of the perturbation ball |
| `epsilon` | float | Yes | Radius of the perturbation ball |

```yaml
input_ball:
  norm: "Linf"
  epsilon: 0.03137
```

---

### `data`

Two modes are available depending on whether you certify a dataset or a single example.

#### Dataset mode (multiple samples)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Dataset identifier (e.g. `"mnist"`, `"cifar10"`, `"blob"`) |
| `path` | str | Yes | Path to the `.pth` dataset file |
| `num_classes` | int | Yes | Number of classes |
| `num_samples` | int | Yes | Number of samples to certify |

```yaml
data:
  name: "mnist"
  path: "data/datasets/mnist_subset_10_per_class.pth"
  num_classes: 10
  num_samples: 100
```

#### Single-example mode

Used for certifying one specific input (e.g. in benchmarks on a single data point).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Dataset name |
| `y` | int | Yes | True label of the example |
| `x` | list[float] \| str | Yes | Input tensor (flat list) or path to a `.pth` file |
| `ytarget` | int | No | If set, only certify against this specific adversarial class |

```yaml
data:
  name: "moon"
  y: 1
  x:
    - 0.0
    - 0.0
  ytarget: 0
```

---

### `network`

Describes the ReLU network to certify.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Network identifier |
| `path` | str | Yes | Path to the `.pt` or `.pth` model file |
| `K` | int | Yes | Number of layers (excluding input) |
| `n` | list[int] | Yes | Sizes of all layers, from input to output (length `K+1`) |
| `dropout` | float | No | Dropout probability used at training time (default: `0`) |

```yaml
network:
  name: "6x100"
  path: "data/models/mnist_adv_6x100.pt"
  K: 7
  n: [784, 100, 100, 100, 100, 100, 100, 10]
```

---

### `models`

A list of solver configurations to run. Each entry is either an SDP model (MOSEK) or a QP/LP model (Gurobi).

#### SDP models (`TargetedSDP`, `UntargetedSDP`, `MzbarSDP`)

These are the core SDP relaxations solved with MOSEK.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `certification_model_type` | `"TargetedSDP"` \| `"UntargetedSDP"` \| `"MzbarSDP"` | — | SDP formulation to use |
| `cuts` | list[str] | `[]` | List of cutting planes to add (see below) |
| `RLT_props` | list[float] | `[0.0]` | Fraction of RLT constraints to add per neuron, one run per value |
| `all_combinations_cuts` | bool | `false` | If `true`, run all subsets of `cuts` independently |
| `MATRIX_BY_LAYERS` | bool \| list[list[int]] | `true` | SDP block structure (see below) |
| `LAST_LAYER` | bool | `false` | Include the logit layer in the SDP variables |
| `solver` | `"mosek_classic"` \| `"mosek_fusion"` | `"mosek_classic"` | MOSEK API backend |
| `use_fusion` | bool | `false` | Deprecated — use `solver: "mosek_fusion"` instead |
| `cp_solver` | str | `"MOSEK"` | CVXPY solver backend (`"MOSEK"`, `"SCS"`, `"CLARABEL"`, etc.) |
| `cp_solver_kwargs` | dict | `null` | Extra keyword arguments passed to `cp.Problem.solve()` |
| `use_callback` | bool | `false` | Enable MOSEK integrality callback |
| `use_active_neurons` | bool | `false` | Add stable-active neurons as explicit SDP variables |
| `ultimate_layer_use_active_neurons` | int | `100000` | Controls active neurons in the penultimate layer when `use_active_neurons` is true: `0` = none, `1` = penultimate only, `2` = all |
| `use_inactive_neurons` | bool | `false` | Add stable-inactive neurons as explicit SDP variables |
| `keep_penultimate_actives` | bool | `false` | Keep penultimate-layer active neurons even when `use_active_neurons` is `false`. Incompatible with `use_active_neurons: true` |
| `bounds_method` | `"IBP"` \| `"alpha-CROWN"` \| `"GREAT_BOUNDS"` \| `"from_file"` | `"alpha-CROWN"` | Method to compute pre-activation bounds |
| `bounds_n_runs` | int | `1` | Number of independent alpha-CROWN runs; the best bounds (max L, min U) are kept. Ignored for non-CROWN methods |
| `bounds_file` | str | `null` | Path to a precomputed bounds CSV (required when `bounds_method: "from_file"`) |
| `L` | list[float] | `null` | Per-layer lower bound constants (used with `bounds_method: "GREAT_BOUNDS"`) |
| `U` | list[float] | `null` | Per-layer upper bound constants (used with `bounds_method: "GREAT_BOUNDS"`) |
| `INPUT_IN_VARIABLES` | bool \| float | `true` | Whether to include the input layer in the SDP variables. `false`/`0.0` removes it entirely; a value `0 < p < 1` keeps the top `p × n₀` input neurons ranked by L1 column norm of W₁ |
| `write_model` | bool | `false` | Export the SDP model to a file for inspection |
| `solver_time_limit` | int | `7200` | MOSEK time limit in seconds. Set to `null` for no limit |

##### Available cuts

Cuts are additional linear or semidefinite constraints that tighten the SDP relaxation.

| Cut name | Description |
|----------|-------------|
| `"RLT"` | Reformulation-Linearization Technique: bilinear constraints between pre- and post-activation variables. The density is controlled by `RLT_props` |
| `"triangularization"` | Triangular RLT constraints between pairs of neurons across consecutive layers |
| `"McCormick_beta_z"` | McCormick envelope constraints between `beta` (activation indicator) and `z` (pre-activation) variables |
| `"beta_logits_comparaison"` | Constraints comparing beta variables against logit differences |

##### `MATRIX_BY_LAYERS` — SDP block structure

Controls how the SDP matrix variable is decomposed into blocks:

| Value | Meaning |
|-------|---------|
| `false` | Single global SDP matrix of size `(1 + Σnₖ) × (1 + Σnₖ)` |
| `true` | One block per consecutive pair of layers: block `k` covers layers `k` and `k+1` |
| `[[0, 1], [1, 2, 3]]` | Custom grouping: each inner list is a group of layer indices. Adjacent groups must share exactly one boundary index |

```yaml
models:
  - certification_model_type: "TargetedSDP"
    cuts:
      - "RLT"
      - "triangularization"
    RLT_props:
      - 0.1
      - 0.2
    all_combinations_cuts: false
    MATRIX_BY_LAYERS: true
    LAST_LAYER: false
    solver: "mosek_classic"
    use_callback: false
    use_active_neurons: true
    use_inactive_neurons: false
    keep_penultimate_actives: false
    bounds_method: "alpha-CROWN"
    bounds_n_runs: 1
    INPUT_IN_VARIABLES: true
    solver_time_limit: 7200
```

#### LP/QP models (`LanQuad`, `MdQuad`, `MzbarQuad`, `ClassicLP`, `LPBoundLayer`)

Quadratic or linear relaxations solved with Gurobi.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `certification_model_type` | `"LanQuad"` \| `"MdQuad"` \| `"MzbarQuad"` \| `"ClassicLP"` \| `"LPBoundLayer"` | — | Model type |
| `LAST_LAYER` | bool | `false` | Include the logit layer |
| `use_active_neurons` | bool | `false` | Use stable-active neurons as variables |
| `use_inactive_neurons` | bool | `false` | Use stable-inactive neurons as variables |
| `bounds_method` | str | `"IBP"` | Bounds computation method |

---

### `divide_run`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `divide_run` | int | `1` | Split the dataset into this many chunks for parallel processing |

---

### `conic_solver` (optional)

Configuration for an external conic bundle solver (warm-starting / post-processing).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | str | — | Output file for the conic bundle solver |
| `McCormick` | `"none"` \| `"diagonal"` | `"none"` | McCormick relaxation variant |

---

## Full example

```yaml
input_ball:
  norm: "Linf"
  epsilon: 0.026

data:
  name: "mnist"
  path: "data/datasets/mnist_subset_10_per_class.pth"
  num_classes: 10
  num_samples: 100

network:
  name: "6x100"
  path: "data/models/mnist_adv_6x100.pt"
  K: 7
  n: [784, 100, 100, 100, 100, 100, 100, 10]

divide_run: 1

models:
  # Fast SDP with layer-wise blocks and RLT + triangularization cuts
  - certification_model_type: "TargetedSDP"
    cuts:
      - "RLT"
      - "triangularization"
    RLT_props:
      - 0.1
    all_combinations_cuts: false
    MATRIX_BY_LAYERS: true
    LAST_LAYER: false
    solver: "mosek_classic"
    use_callback: false
    use_active_neurons: true
    use_inactive_neurons: false
    keep_penultimate_actives: false
    bounds_method: "alpha-CROWN"
    bounds_n_runs: 1
    solver_time_limit: 3600

  # Richer SDP with McCormick cuts and custom block decomposition
  - certification_model_type: "UntargetedSDP"
    cuts:
      - "RLT"
      - "triangularization"
      - "McCormick_beta_z"
      - "beta_logits_comparaison"
    RLT_props:
      - 1.0
    MATRIX_BY_LAYERS:
      - [0, 1, 2]
      - [2, 3, 4]
      - [4, 5, 6]
    LAST_LAYER: false
    use_active_neurons: false
    keep_penultimate_actives: true
    bounds_method: "alpha-CROWN"
    solver_time_limit: 7200
```

## Project structure

```
FastSDPCertification/
├── config/                  # YAML experiment configs
├── src/
│   ├── fastsdp_tools/
│   │   └── yaml_config.py   # Pydantic config models (validation)
│   ├── networks/            # ReLU network definition and training
│   ├── solve/
│   │   ├── generic_solver.py
│   │   └── sdp_solve/
│   │       ├── mosek_generic_solver.py
│   │       ├── handler/     # MOSEK classic and fusion handlers
│   │       └── SDPmodels/   # SDP constraint builders
│   ├── bounds.py            # IBP / GREAT_BOUNDS
│   ├── bounds_crown.py      # alpha-CROWN bounds
│   └── data/                # Dataset loading utilities
├── data/
│   ├── datasets/            # Pre-processed .pth dataset files
│   └── models/              # Pre-trained .pt network weights
├── results/                 # Output logs and benchmark CSVs
└── opt/                     # License files (mosek.lic, gurobi.lic)
```

## Dependencies

- Python ≥ 3.7
- PyTorch
- MOSEK (≥ 10) with a valid license
- `mosek` Python package
- `pydantic` ≥ 2
- `numpy`, `pandas`
- `torchvision`
- Gurobi (optional, for LP/QP models)
- CVXPY (optional, for `cp_solver` backend)
- `auto_LiRPA` (for `alpha-CROWN` bounds)
