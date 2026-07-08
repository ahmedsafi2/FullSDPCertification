#!/usr/bin/env python3
"""
Génère des fichiers de configuration YAML pour un produit cartésien de paramètres.

Ce script crée des fichiers YAML dans le dossier `all_product_yamls/`.
Chaque fichier correspond à une combinaison unique de paramètres définis
dans la grille `PARAM_GRID`.

Le nom de fichier est généré à partir des paramètres, par exemple :
  blob_4x10_eps0p01_timeout100_iter10.yaml
"""

import copy
import itertools
import yaml
from pathlib import Path

# Chemin racine du projet (remonte de deux niveaux depuis scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Dossier de sortie pour les fichiers YAML générés
DEFAULT_YAML_DIR = PROJECT_ROOT / "all_product_yamls"

# Configuration de base pour le réseau 'blob_4x10'.
# Les valeurs ici seront utilisées si elles ne sont pas surchargées par la grille.
BASE_CONFIG = {
    'general': {
        'enable_incomplete_verification': True,
        'csv_name': None,  # Pourrait être paramétré pour avoir un CSV par run
    },
    'model': {
        # Le nom du modèle est fixé ici, mais pourrait être un paramètre.
        'name': 'blob_4x10',
    },
    'data': {
        'dataset': 'blob',
        'start': 0,
        'end': 100,  # Nombre d'images à tester
    },
    'specification': {
        'epsilon': 0.01,  # Valeur par défaut, sera surchargée
        'norm': 'inf',
    },
    'solver': {
        'batch_size': 2048,
        'beta-crown': {
            'iteration': 20,  # Valeur par défaut, sera surchargée
        },
    },
    'bab': {
        'timeout': 100,  # Valeur par défaut, sera surchargée
    }
}

# Grille de paramètres à combiner.
# Chaque clé est un chemin dans le dictionnaire de configuration (ex: 'specification.epsilon').
# La valeur est une liste des valeurs à tester.
PARAM_GRID = {
    'specification.epsilon': [0.01, 0.025, 0.05],
    'bab.timeout': [100, 300],
    'solver.beta-crown.iteration': [10, 20],
}


def set_nested_value(d: dict, key_path: str, value):
    """Modifie une valeur dans un dictionnaire imbriqué en utilisant un chemin de clé."""
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def main():
    """Fonction principale du script."""
    output_dir = DEFAULT_YAML_DIR
    output_dir.mkdir(exist_ok=True)
    print(f"Génération des fichiers YAML dans : {output_dir}")

    # Nettoyer le dossier de sortie avant de générer de nouveaux fichiers
    print("Nettoyage du dossier de sortie...")
    count_deleted = 0
    for old_file in output_dir.glob('*.yaml'):
        old_file.unlink()
        count_deleted += 1
    if count_deleted > 0:
        print(f"{count_deleted} ancien(s) fichier(s) YAML supprimé(s).")

    # Créer le produit cartésien des paramètres
    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Génération de {len(param_combinations)} combinaisons de paramètres...")

    # Boucler sur chaque combinaison et générer un fichier YAML
    for i, params in enumerate(param_combinations):
        config = copy.deepcopy(BASE_CONFIG)
        filename_parts = [BASE_CONFIG['model']['name']]

        for key, value in params.items():
            set_nested_value(config, key, value)
            short_key = key.split('.')[-1].replace('epsilon', 'eps').replace('iteration', 'iter')
            value_str = str(value).replace('.', 'p')
            filename_parts.append(f'{short_key}{value_str}')

        filename = '_'.join(filename_parts) + '.yaml'
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nTerminé. {len(param_combinations)} fichiers YAML ont été générés dans {output_dir}.")


if __name__ == "__main__":
    main()