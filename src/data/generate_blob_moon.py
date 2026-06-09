import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import os

if not os.path.exists("data"):
    os.makedirs("data")


def generate_blob(n_samples=1000, centers=3, n_features=2, save: bool = False):
    """
    Generate a synthetic blob dataset with feature values scaled between 0 and 5.
    """
    X_blob, y_blob = make_blobs(
        n_samples=n_samples, centers=centers, n_features=n_features, random_state=42
    )

    # Scale features to range [0, 5]
    scaler = MinMaxScaler(feature_range=(0, 5))
    X_blob_scaled = scaler.fit_transform(X_blob)

    X_blob_tensor = torch.FloatTensor(X_blob_scaled)
    y_blob_tensor = torch.LongTensor(y_blob)

    dataset = TensorDataset(X_blob_tensor, y_blob_tensor)

    if save:
        print("Saving blob dataset ...")
        torch.save(
            {"dataset": dataset},
            "data/datasets/blob_dataset.pth",
        )
    else:
        return dataset


def generate_moon(n_samples=1000, noise=0.1, save: bool = False):
    """
    Generate a synthetic moon dataset with feature values scaled between 0 and 5.
    """
    X_moon, y_moon = make_moons(n_samples=n_samples, noise=noise, random_state=42)

    # Scale features to range [0, 5]
    scaler = MinMaxScaler(feature_range=(0, 5))
    X_moon_scaled = scaler.fit_transform(X_moon)

    X_moon_tensor = torch.FloatTensor(X_moon_scaled)
    y_moon_tensor = torch.LongTensor(y_moon)

    dataset = TensorDataset(X_moon_tensor, y_moon_tensor)

    if save:
        print("Saving moon dataset ...")
        torch.save(
            {"dataset": dataset},
            "data/datasets/moon_dataset.pth",
        )
    else:
        return dataset


if __name__ == "__main__":
    generate_blob(save=True)
    generate_moon(save=True)
