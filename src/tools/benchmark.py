import os
from .utils import get_project_path


def create_folder(folder_path):
    os.makedirs(get_project_path(folder_path), exist_ok=True)


def create_folder_benchmark(folder_name):
    """Création du dossier de résultats"""
    folder_dir = f"results/benchmark/{folder_name}"
    create_folder(folder_dir)


def create_subfolder_benchmark(folder_name, subfolder_name):
    """Création du sous-dossier de résultats"""
    folder_dir = f"results/benchmark/{folder_name}/{subfolder_name}"
    create_folder(folder_dir)

