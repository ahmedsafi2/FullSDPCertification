# FastSDPCertification — Contexte du projet

## Objectif
Librairie de **certification de robustesse adversariale** de réseaux de neurones profonds (DNNs) à activations ReLU, via des relaxations **Semi-Définies Positives (SDP)**.

L'objectif est de vérifier formellement qu'un DNN est ε-robuste sur un dataset : pour toute entrée x et toute perturbation dans la boule B_ε(x), le réseau prédit toujours la bonne classe y.

Le point clé du projet : introduire une formulation **untargeted** (SDPU) qui certifie toutes les classes cibles simultanément en **une seule résolution SDP**, contrairement aux approches classiques qui nécessitent une résolution par classe cible.

---

## Concepts mathématiques fondamentaux

### Problème de certification
- **Robustesse ciblée** : vérifier que min_{z0 ∈ B_ε(x)} z^y_K - z^j_K ≥ 0 pour une classe cible j donnée.
- **Robustesse complète** : vérifier min_{j ∈ J̄_K} min_{z0 ∈ B_ε(x)} z^y_K - z^j_K ≥ 0, i.e. pour toutes les classes.
- Un **lower bound non-négatif** suffit à certifier la robustesse (vérification incomplète).

### Notations clés
- `K` : nombre de couches du réseau
- `k`: nom d'une couche du réseau
- `z_k` : vecteur post-activations de la couche k
- `W_k`, `b_k` : poids et biais de la couche k
- `L_k`, `U_k` : bornes inf/sup sur le vecteur de pré-activation de la couche k (calculées par α-β-CROWN)
- `J̄_K` : ensemble des classes cibles possibles (toutes sauf la vraie classe y)
- `l_k = (ReLU(U_k) - ReLU(L_k)) / (U_k - L_k)` : indicateur de stabilité des neurones
- `l_k = 1` : neurone **stable actif** (L_k ≥ 0)
- `l_k = 0` : neurone **stable inactif** (U_k ≤ 0)
- `0 < l_k < 1` : neurone **instable**


### Formulations quadratiques
- **(QPT^j)** : formulation quadratique ciblée pour une classe j — non convexe, NP-difficile
- **(QPU)** : nouvelle formulation quadratique untargeted avec variables binaires β_j ∈ {0,1}
  - β_j = 1 si la classe j est la pire attaque adversariale
  - Contrainte Σ β_j = 1
  - **Théorème clé** : v(QPU) = min_{j ∈ J̄_K} v(QPT^j)

### Relaxations SDP
- **(SDPT^j)** : relaxation SDP de (QPT^j), avec décomposition chordale en matrices Pk par paires de couches consécutives
- **(SDPU)** : relaxation SDP de (QPU), avec matrice additionnelle P_{K-2} = [1 z_{K-2} z_{K-1} β]^T pour linéariser les produits β_j * z_K

### Familles de contraintes et emplacement dans le code 
1. **Contraintes ReLU linéarisées** linéarisation de z_{k+1} = ReLU(W_{k+1}z_k + b_{k+1}) ()
2. **Contraintes de borne** diag(Pk[z_k z_k^T]) - (L_k+U_k)⊙Pk[z_k] + U_k⊙L_k ≤ 0
3. **Contrainte triangulaire** z_k ≤ A_k * z_{k-1} + B_k (tightening de la borne sup)
4. **Contrainte de cohérence** Pk[(1 z_{k+1})(1 z_{k+1})^T] = P_{k+1}[(1 z_{k+1})(1 z_{k+1})^T]
5. **Coupes RLT** Reformulation-Linearization Technique, sélectionnées par heuristique (top-p% des poids)
6. **Coupes McCormick** Pour les produits β_j * z_k dans la formulation untargeted
7. **Coupes spécifiques QPU** β_{j1}*β_{j2} = 0, et inégalités couplant les logits adversariaux

---

## Fichiers yaml dans le dossier config
Les tests sont faits à partir de fichiers yaml dans le dossier config. Ils sont lus grâce à la librairie pydantic dans le fichier src/tools/yaml_config.
Exemple de fichier yaml :

data :
  name : "mnist"
  path : "data/datasets/mnist_subset_10_per_class.pth"
  num_classes : 10
  num_samples : 100

input_ball : 
  norm : "Linf"
  epsilon : 0.026  # 0.026

network :
  name : "6x100"
  path : "data/models/mnist_adv_6x100.pt"
  K : 7
  n : 
    - 784
    - 100
    - 100
    - 100
    - 100
    - 100
    - 100
    - 10

models :  # Models SDP ou LP ou QP pour résoudre le problème de certification
  - certification_model_name : "MdSDP"   # Base du modèle : MdSDP pour SDP untargeted, SDP qui certifie sur toutes les classes en même temps. Choisir LanSDP pour un modèle SDP qui certifie sur une classe une par une
    cuts : # Ensemble de coupes rajoutées dans le modèle
    - "RLT"  
    - "triangularization" 
    - "Tij"
    - "beta_logits_comparaison_1"
    - "beta_logits_comparaison_2" -->
    - "McC_betaz_logits"
    - "Tij_before_penultimate_layer"
    RLT_props : # Proportion de coupes RLT à rajouter selon l'heuristique définie dans le fichier certification_problem_constraints_rlt.py
    - 1.
    all_combinations_cuts : false # Ancien paramètre, n'est peut-être plus à jour désormais, qui voulait tester toutes les combinaisons de coupes pour chaque modèle
    MATRIX_BY_LAYERS: true  # Si true, fait de la décomposition chordale : passage d'une matrice P à des sous-matrices P_k
    LAST_LAYER: false # Si true, considère les derniers élements du réseau sur la couche K (les logits) comme étant des variables
    use_fusion: false  # Whether to use the fusion API for MOSEK
    use_callback : false # Whether to use the callback for MOSEK
    use_active_neurons: true # Whether to use active neurons in the certification problem as variables
    #ultimate_layer_use_active_neurons: 5 # Whether to use active neurons in the ultimate layer in the certification problem as variables
    use_inactive_neurons : false # Whether to use inactive neurons in the certification problem as variables
    keep_penultimate_actives : true # Whether to keep the penultimate layer active neurons in the certification problem
    bounds_file: "/share/homes/boyerma/FastSDPCertification/results/benchmark/6x100-0.026/Bornes/bornes_sdpu.csv"  # Folder where precomputed bounds are stored
    bounds_method: "from_file"   #"alpha-CROWN"  # Method to compute bounds, options: "IBP", "GREAT_BOUNDS", "GREAT_BOUNDS_LIN"

### Run sur les SDP classiques
| `SDPT-IP` | SDP ciblé sans décomposition chordale (Raghunathan et al. 2018) |
Paramètres configurants SDP-IP :
    certification_model_name : "LanSDP"  
    # cuts :  # Pas de coupes rajoutées dans SDP-IP
    # RLT_props : 
    MATRIX_BY_LAYERS: false  # Pas de décomposition chordale pour SDP-IP
    LAST_LAYER: false # Les variables vont jusqu'à l'avant dernière couche
    use_active_neurons: true # Les neurones actifs sont considérés comme des variables, pas prunés
    use_inactive_neurons : true # Les neurones inactifs sont considérés comme des variables, pas prunés
    keep_penultimate_actives : true # Whether to keep the penultimate layer active neurons in the certification problem

| `SDPT,layer` | SDP ciblé avec décomposition chordale + neurones inactifs + triangulaire | (Batten et al, 2021)
Paramètres configurants SDP-Layer :
    certification_model_name : "LanSDP"  
    cuts : 
      - "triangularization" 
    # RLT_props : # pas de coupes RLT pour SDP-Layer
    MATRIX_BY_LAYERS: true  # Utilisation de la décomposition chordale pour SDP-Layer
    LAST_LAYER: false # Les variables vont jusqu'à l'avant dernière couche
    use_active_neurons: true # Les neurones actifs sont considérés comme des variables, pas prunés
    use_inactive_neurons : false # Les neurones inactifs ne sont ps considérés comme des variables, ils sont prunés
    keep_penultimate_actives : true # Whether to keep the penultimate layer active neurons in the certification problem

| `SDPT` | SDPT,layer + 10% de coupes RLT |
Paramètres configurants SDP-t :
    certification_model_name : "LanSDP"  
    cuts : 
      - "triangularization" 
      - "RLT" # Utilisation des coupes RLT dans SDPt
    RLT_props : 0.1 # Choix classique, peut varier
    MATRIX_BY_LAYERS: true  # Utilisation de la décomposition chordale pour SDPt
    LAST_LAYER: false # Les variables vont jusqu'à l'avant dernière couche
    use_active_neurons: true # Les neurones actifs sont considérés comme des variables, pas prunés
    use_inactive_neurons : false # Les neurones inactifs ne sont ps considérés comme des variables, ils sont prunés
    keep_penultimate_actives : true # Whether to keep the penultimate layer active neurons in the certification problem

| `SDPU` | Notre méthode : SDP untargeted + pruning actifs + 100% RLT |
Paramètres configurants SDPu :
  - certification_model_name : "MdSDP"   # Modèle SDP non ciblé
    cuts : # Ensemble de coupes rajoutées dans le modèle
    - "RLT"  # Coupe déjà existantes dans les modèles SDP ciblés
    - "triangularization" # Coupe déjà existantes dans les modèles SDP ciblés
    - "Tij"  # McCormick sur les produits β_j * z_k 
    - "beta_logits_comparaison_1" # Coupes spécifiques au modèle non ciblé liant des logits et des β_j de deux classes différentes
    - "beta_logits_comparaison_2" # Coupes spécifiques au modèle non ciblé liant des logits et des β_j de deux classes différentes
    - "McC_betaz_logits" # McCormick sur les produits β_j * z_K sur la dernière couche 
    - "Tij_before_penultimate_layer" # McCormick sur les produits β_j * z_k 
    RLT_props : # Proportion de coupes RLT à rajouter selon l'heuristique définie dans le fichier certification_problem_constraints_rlt.py
    - 1.
    all_combinations_cuts : false # Ancien paramètre, n'est peut-être plus à jour désormais, qui voulait tester toutes les combinaisons de coupes pour chaque modèle
    MATRIX_BY_LAYERS: true  # Si true, fait de la décomposition chordale : passage d'une matrice P à des sous-matrices P_k
    LAST_LAYER: false # Si true, considère les derniers élements du réseau sur la couche K (les logits) comme étant des variables
    use_fusion: false  # Whether to use the fusion API for MOSEK
    use_callback : false # Whether to use the callback for MOSEK
    use_active_neurons: true # Whether to use active neurons in the certification problem as variables
    #ultimate_layer_use_active_neurons: 5 # Whether to use active neurons in the ultimate layer in the certification problem as variables
    use_inactive_neurons : false # Whether to use inactive neurons in the certification problem as variables
    keep_penultimate_actives : true # Whether to keep the penultimate layer active neurons in the certification problem



## Architecture de la librairie

FastSDPCertification/src/
├── CLAUDE.md
│
├── certification_problem.py        # Classe principale : définit le problème de certification
├── bounds.py                       # Calcul des bornes L_k, U_k (méthode de base)
├── bounds_crown.py                 # Calcul des bornes via α-β-CROWN
├── bounds_crown_claude.py          # Variante / expérimentation des bornes CROWN
│
├── adversarial_attacks/            # Attaques adversariales (pour évaluation empirique)
│   ├── pgd.py                      # Attaque PGD (Projected Gradient Descent)
│   ├── lp.py                       # Attaque LP
│   ├── lp_multiprocessing.py       # Versions parallélisées de l'attaque LP
│   ├── lp_multiprocessing2.py
│   ├── lp_multiprocessing3.py
│   ├── crown_ibp.py                # Attaque via CROWN-IBP
│   └── sdp.py                      # Attaque via SDP
│
├── networks/                       # Définition et entraînement des réseaux
│   ├── network.py                  # Classe de base du réseau (MLP ReLU)
│   ├── mlp_sdp_crown.py            # MLP adapté pour SDP + CROWN
│   ├── mlp_bb_beta_crown.py        # MLP adapté pour Branch&Bound + β-CROWN
│   ├── train.py                    # Entraînement standard
│   ├── train_cifar.py              # Entraînement sur CIFAR
│   ├── adv_train.py                # Entraînement adversarial (PGD)
│   └── robust_evaluate.py          # Évaluation de robustesse empirique
│
├── data/                           # Chargement et génération des datasets
│   ├── load.py                     # Chargement général des datasets
│   ├── generate_mnist.py
│   ├── generate_cifar10.py
│   ├── generate_cifar100.py
│   ├── generate_subsets_cifar100_by_labels.py
│   ├── generate_blob_moon.py       # Datasets synthétiques (debug/test)
│   └── analyse.py                  # Analyse des datasets
│
├── solve/                          # Cœur de la résolution SDP
│   ├── generic_solver.py           # Interface générique (MOSEK / Gurobi / CB)
│   ├── getting_results.py          # Extraction et agrégation des résultats
│   ├── benchmark_mosek.py          # Benchmarks MOSEK
│   ├── benchmark_cb.py             # Benchmarks Conic Bundle
│   │
│   ├── mosek_solve/                # ★ Module principal : résolution via MOSEK
│   │   ├── mosek_generic_solver.py # Point d'entrée MOSEK
│   │   ├── get_variables.py        # Extraction des variables de la solution
│   │   ├── run_benchmark.py
│   │   │
│   │   ├── SDPmodels/              # ★★ Construction du problème SDP (fichiers critiques)
│   │   │   ├── certification_problem_objective.py          # Objectif : min W^y P[z_{K-1}] - Σ W^j P[β_j z_{K-1}]
│   │   │   ├── certification_problem_constraints_relu.py   # Contraintes ReLU (14)-(15) + pruning actifs 
│   │   │   ├── certification_problem_constraints_bounds.py # Contraintes de borne (16)
│   │   │   ├── certification_problem_constraints_beta.py   # Contraintes β (23)-(30) : untargeted
│   │   │   ├── certification_problem_constraints_rlt.py    # Coupes RLT (42)-(46)
│   │   │   ├── certification_problem_constraints_sdp.py    # Contrainte SDP Pk ⪰ 0 
│   │   │   ├── certification_problem_constraints_division_by_layers.py  # Décomposition chordale
│   │   │   ├── Lan_SDP.py          # Implémentation de SDPT (Lan et al. 2022) (Cette classe Lan_SDP peut permettre d'implémenter SDP_Layer ou SDP-IP en faisant changer les paramètres)
│   │   │   ├── Md.py               # Matrice Md (structure des variables)
│   │   │   ├── Mzbar.py            # Matrice Mzbar
│   │   │   ├── krelu.py            # Contraintes k-ReLU
│   │   │   └── SDP_attack.py       # Attaque SDP
│   │   │
│   │   └── handler/                # Gestion bas niveau MOSEK (API classic vs Fusion)
│   │       ├── constraints.py
│   │       ├── indexes_matrices.py  # ★★ Indexation des matrices Pk dans MOSEK
│   │       ├── indexes_variables.py # ★★ Indexation des variables
│   │       ├── objective.py
│   │       ├── variables_call.py # ★★★ Fichier très critique 
│   │       ├── variable_elements.py # ★★★ Fichier très critique
│   │       ├── common_handler_functions.py
│   │       ├── mosek_classic/      # API MOSEK classique (bas niveau)
│   │       │   ├── handler_classic.py
│   │       │   ├── constraints_classic.py
│   │       │   ├── objective_classic.py
│   │       │   ├── callback_classic.py
│   │       │   └── results_classic.py
│   │       └── mosek_fusion/       # API MOSEK Fusion (haut niveau)
│   │           ├── handler_fusion.py
│   │           ├── constraints_fusion.py
│   │           ├── objective_fusion.py
│   │           ├── callback_fusion.py
│   │           └── results_fusion.py
│   │
│   └── gurobi_solve/               # Résolution alternative via Gurobi (LP/QP)
│       ├── gurobi_generic_solver.py
│       ├── constraints.py
│       ├── variables.py
│       ├── objective.py
│       ├── analyser.py
│       ├── callback.py
│       ├── lpmodels/               # Modèles LP (bornes par couche, attaque LP)
│       │   ├── LP_attack.py
│       │   └── LP_layer_bound.py
│       └── quadmodels/             # Modèles quadratiques
│           ├── Lan_quad.py
│           ├── Md_quad.py
│           └── Mzbar_quad.py
│
├── conic_bundle/                   # Résolution alternative via Conic Bundle (non important ici)
│
└── tools/                          # Utilitaires généraux
    ├── benchmark.py                # Orchestration des benchmarks
    ├── utils.py                    # Fonctions utilitaires
    └── yaml_config.py              # Chargement de la configuration YAML


## Fichiers les plus critiques (★★)
Les bugs sont très probablement dans ces fichiers :

solve/mosek_solve/SDPmodels/certification_problem_constraints_relu.py — pruning des neurones actifs + McCormick cross-layer
solve/mosek_solve/SDPmodels/certification_problem_constraints_beta.py — contraintes untargeted (β)
solve/mosek_solve/handler/variables_call.py — Implémentation de la classe VariablesCall qui est centrale (les classes Constraints et Objectives dérivent de VariablesCall)
solve/mosek_solve/handler/variables_elements.py — Implémentation efficace (rapidité) des tuples (layer, neuron) et (index_matrix, index_row, index_col) avec la librairie numba. Ce fichier est très critique : les clés des dictionnaires doivent bien être uniques, entre autres.
solve/mosek_solve/handler/indexes_matrices.py — indexation des matrices Pk dans MOSEK
solve/mosek_solve/handler/indexes_variables.py — indexation des variables
bounds_crown.py — calcul des bornes L_k, U_k
certification_problem.py — orchestration globale


> **Note pour Claude** : explorer la structure réelle avec `find . -name "*.py" | head -50` au début de chaque session.



## Bugs connus / comportements suspects

> **À remplir** : décrivez ici les symptômes observés, les erreurs obtenues, les fonctions suspectes.

Exemples à documenter :
- [ ] Pourquoi beaucoup de valeurs optimales tendent vers 0 quand le problème est suffisamment grand?
- [ ] Pruning des neurones actifs qui produit des contraintes incorrectes ?
- [ ] Problème dans l'implémentation des éléments "elements" dans le dictionnaire des contraintes ?
- [ ] Erreurs dans l'implémentation des contraintes ?

---

## Dépendances

- **MOSEK** (≥ 9.x) : solveur SDP, API Python
- **CVXPY** ou interface directe MOSEK : construction du problème
- **NumPy** : manipulation des matrices
- **α-β-CROWN** : propagation des bornes L_k, U_k
- **PyTorch** : chargement des réseaux de neurones
- **Datasets** : MNIST, CIFAR-100, FashionMNIST, EMNIST, KMNIST


## Réseaux de test (pour reproduire les expériences)

| Réseau | Architecture | Dataset | ε |
|---|---|---|---|
| 9x100 | 784-9×100-10 | MNIST | 0.026 |
| 9x200 | 784-9×200-10 | MNIST | 0.015 |
| FCNNA | 3072-2×20-100 | CIFAR-100 | 8/255 |
| 6x100-{5,20,50,67} | 784-6×100-{5..67} | Multi-dataset | 0.05 |

Entraînés avec PGD adversarial training (Adam, lr=0.001, batch=128, 200 epochs).


## Points mathématiques délicats à garder en tête

1. **Pruning des neurones stables actifs** : pour chaque neurone stable actif `a` de la couche `k`, on exprime `z^a_k` comme combinaison linéaire des neurones instables de toutes les couches `l ≤ k-1`. Erreur facile : oublier la récursion sur les couches intermédiaires.

2. **McCormick cross-layer** : les produits `z^j_{k+1} * z^u_l` pour `l ≤ k-1` ne sont pas dans les matrices Pk (décomposition chordale = seulement couches consécutives). Il faut les borner avec McCormick en utilisant `U^j_{k+1}` et `[L^u_l, U^u_l]` (avec `L^u_l = 0` pour `l ≥ 1`).

3. **Pruning des classes dominées** (: avant de lancer SDPU, on peut éliminer les classes j telles qu'il existe une classe j̃ avec `U^j_K ≤ L^{j̃}_K`. L'intersection `∩_{j ∈ J̄_K} [L^j_K, U^j_K] ≠ ∅` doit être vérifiée.

4. **Budget RLT** : le nombre max de coupes RLT est plafonné à N_max. Si le nombre théorique dépasse ce seuil, on ajuste p' = p * N_max / N.

5. **Contrainte de cohérence relaxée** : on ne garde que les `n_k` contraintes linéaires `Pk[z_{k+1}] = P_{k+1}[z_{k+1}]`, pas les `n_k(n_k+3)/2` contraintes quadratiques complètes.

---

## Commandes utiles

```bash
# Explorer la structure
find . -name "*.py" | sort

# Lancer une certification sur MNIST 9x100
python src/certification_problem.py mnist-9x100 "Test-certification"

# Vérifier les dépendances
python -c "import mosek; print(mosek.version())"
```
