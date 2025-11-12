"""Defines the base SnapshotDataset for PyTorch."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class SnapshotDataset(Dataset):
    def __init__(self, snaps: List[dict], edge_index: np.ndarray, node_features: np.ndarray,
                 input_mean: np.ndarray, input_std: np.ndarray,
                 target_mean: np.ndarray, target_std: np.ndarray,
                 target_cols: List[int]):
        self.snaps = snaps
        self.edge_index = torch.from_numpy(edge_index).long()
        self.node_features = torch.from_numpy(node_features).float()
        self.in_mean = input_mean; self.in_std = input_std
        self.t_mean = target_mean; self.t_std = target_std
        self.target_cols = target_cols

    def __len__(self):
        return len(self.snaps)

    def _split_cols(self, edge_attr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        F_e = edge_attr.shape[1]
        mask = np.zeros(F_e, dtype=bool)
        mask[self.target_cols] = True
        X = edge_attr[:, ~mask]
        y = edge_attr[:, mask]
        return X.astype(np.float32), y.astype(np.float32)

    def __getitem__(self, idx):
        snap = self.snaps[idx]
        edge_attr = snap['edge_attr'] if isinstance(snap, dict) else snap[1]
        X, y = self._split_cols(edge_attr)
        Xn = ((X - self.in_mean) / self.in_std) if X.shape[1] > 0 else X
        Yn = ((y - self.t_mean) / self.t_std)
        return self.node_features, self.edge_index, torch.from_numpy(Xn), torch.from_numpy(Yn)