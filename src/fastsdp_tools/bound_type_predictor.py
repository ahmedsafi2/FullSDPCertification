"""Prédiction du type de relaxation McCormick ('composed' vs 'one_variable')
pour un produit croisé z_{l,u} * z_{k,j}.

Par défaut (method="tree"), la décision utilise un arbre "gelé" pur Python
(_predict_gain_frozen_tree ci-dessous, seuils issus du notebook
combo_visualisation, section 11) : aucune dépendance, aucun fichier à
déployer, disponible immédiatement. L'inférence est donc quasi-instantanée,
ce qui permet de l'appeler "au fur et à mesure" pendant la construction des
contraintes, dès que L[layer][neuron] / U[layer][neuron] sont connus (voir
variables_call.py::add_z_quad_active_neuron, mccormick_type="auto").

D'autres méthodes ("rbf", "tree_ml", ...) peuvent être exportées en .joblib
(notebook, section 11) dans MODELS_DIR et sélectionnées via method=... ; elles
sont chargées une seule fois (cache module-level) puis réutilisées.

⚠️ L'arbre gelé (et les modèles .joblib) sont spécifiques au réseau/epsilon/
cuts sur lesquels ils ont été entraînés (ex: blob_nn_4x10, epsilon=1). Il faut
les ré-entraîner (notebook, section 11) pour toute autre config. Ceci
n'affecte que la vitesse/qualité de la relaxation, jamais sa validité : les
deux choix restent des relaxations McCormick correctes.
"""

from __future__ import annotations

import os
from pathlib import Path

# Dossier contenant les .joblib générés par la section 11 du notebook
# combo_visualisation. Surchargeable via la variable d'environnement
# FASTSDP_BOUND_TYPE_MODELS_DIR (utile sur cluster / Jean Zay).
MODELS_DIR = Path(os.environ.get("FASTSDP_BOUND_TYPE_MODELS_DIR", "models"))

PROD_FEATURES = ["LB_neuron1", "UB_neuron1", "LB_neuron2", "UB_neuron2", "l", "k"]

_MODEL_CACHE: dict = {}
_UNAVAILABLE_WARNED = False


def _predict_gain_frozen_tree(LB_neuron1, UB_neuron1, LB_neuron2, UB_neuron2, l, k):
    """Arbre "gelé" (seuils recopiés depuis le notebook combo_visualisation,
    section 11 — réseau blob_nn_4x10, epsilon=1). Pur Python, sans I/O ni
    dépendance externe : c'est ce qui rend method="tree" utilisable
    immédiatement, sans avoir à exporter/déployer de .joblib.

    ⚠️ Spécifique au réseau/epsilon/cuts sur lesquels il a été entraîné :
    régénérez ces seuils (notebook, section 11) pour toute autre config.
    Comme la décision ne fait que choisir entre deux relaxations McCormick
    valides (elle n'affecte pas la correction de la certification, seulement
    sa qualité/vitesse), un arbre légèrement désajusté reste sans danger.
    """
    if LB_neuron2 <= 10.321605:
        if LB_neuron2 <= 9.981143:
            if LB_neuron2 <= -27.166220:
                return -0.040124
            else:
                return 0.003577
        else:
            if LB_neuron1 <= -8.505470:
                return 0.229491
            else:
                return 0.161073
    else:
        if LB_neuron2 <= 10.927463:
            if LB_neuron1 <= -8.394440:
                return -0.124494
            else:
                return -0.064336
        else:
            if UB_neuron2 <= 87.908302:
                return -0.032395
            else:
                return -0.000543


def _load_model(name: str):
    if name not in _MODEL_CACHE:
        import joblib  # import paresseux : pas de dépendance dure si method="auto" n'est jamais utilisé

        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable: {path}. "
                f"Générez-le via la section 11 du notebook combo_visualisation, "
                f"ou pointez FASTSDP_BOUND_TYPE_MODELS_DIR vers le bon dossier."
            )
        _MODEL_CACHE[name] = joblib.load(path)
    return _MODEL_CACHE[name]


def _features_row(l: int, u_idx: int, k: int, j_idx: int,
                   LB_neuron1: float, UB_neuron1: float,
                   LB_neuron2: float, UB_neuron2: float):
    import pandas as pd

    return pd.DataFrame([{
        "LB_neuron1": LB_neuron1, "UB_neuron1": UB_neuron1,
        "LB_neuron2": LB_neuron2, "UB_neuron2": UB_neuron2,
        "l": l, "k": k,
    }])[PROD_FEATURES]


def predict_bound_type(
    l: int, u_idx: int, k: int, j_idx: int,
    LB_neuron1: float, UB_neuron1: float, LB_neuron2: float, UB_neuron2: float,
    method: str = "tree",
    gain_threshold: float = 0.0,
    fallback: str = "one_variable",
) -> str:
    """Décide 'composed' ou 'one_variable' pour un produit croisé, dès que ses
    bornes LB/UB sont connues.

    Args:
        l, k: indices de couche des 2 neurones du produit.
        u_idx, j_idx: indices de neurone (non utilisés par le modèle actuel,
            gardés pour signature/logging cohérents avec le reste du pipeline).
        LB_neuron1, UB_neuron1, LB_neuron2, UB_neuron2: bornes pré-activation
            des 2 neurones (ex: self.L[l][u_idx], self.U[l][u_idx], ...).
        method: "tree" (défaut, arbre gelé pur Python, aucun fichier requis)
            ou le nom d'un modèle exporté en .joblib dans MODELS_DIR (ex:
            "rbf", ou "tree_ml" pour un arbre sklearn ré-entraîné).
        gain_threshold: seuil de gain prédit au-delà duquel 'composed' est
            recommandé.
        fallback: valeur retournée si le modèle .joblib demandé est
            indisponible — pour ne jamais faire planter un run de
            production à cause d'un modèle non déployé. Sans effet pour
            method="tree" (aucun fichier requis).

    Returns:
        "composed" ou "one_variable"
    """
    if method == "tree":
        predicted_gain = _predict_gain_frozen_tree(
            LB_neuron1, UB_neuron1, LB_neuron2, UB_neuron2, l, k
        )
        return "composed" if predicted_gain > gain_threshold else "one_variable"

    global _UNAVAILABLE_WARNED
    try:
        reg = _load_model(f"bound_type_{method}_reg")
    except FileNotFoundError as e:
        if not _UNAVAILABLE_WARNED:
            import logging
            logging.getLogger("Mosek_logger").warning(
                f"{e} — repli sur '{fallback}' pour tous les produits."
            )
            _UNAVAILABLE_WARNED = True
        return fallback

    x = _features_row(l, u_idx, k, j_idx, LB_neuron1, UB_neuron1, LB_neuron2, UB_neuron2)
    predicted_gain = reg.predict(x)[0]
    return "composed" if predicted_gain > gain_threshold else "one_variable"
