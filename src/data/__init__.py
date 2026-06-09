from .load import load_dataset
from .analyse import analyze_class_distribution
import torch
from torch.utils.data import TensorDataset


# Ajouter TensorDataset à la liste des objets sécurisés
# torch.serialization.add_safe_globals([TensorDataset])

__all__ = ["load_dataset", "analyze_class_distribution"]
