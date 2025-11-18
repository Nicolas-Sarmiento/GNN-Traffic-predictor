"""Defines the base SnapshotDataset for PyTorch."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, List, Optional, Tuple

class SnapshotDataset(Dataset):
    def __init__(self, snaps: List[dict], edge_index: np.ndarray, node_features: np.ndarray,
                 input_mean: np.ndarray, input_std: np.ndarray,
                 target_mean: np.ndarray, target_std: np.ndarray,
                 target_cols: List[int],
                 edge_attr_getter: Optional[Callable[[dict], np.ndarray]] = None):
        self.snaps = snaps
        self.edge_index = torch.from_numpy(edge_index).long()
        self.node_features = torch.from_numpy(node_features).float()
        self.in_mean = input_mean; self.in_std = input_std
        self.t_mean = target_mean; self.t_std = target_std
        self.target_cols = target_cols
        self.edge_attr_getter = edge_attr_getter

    def __len__(self):
        return len(self.snaps)

    def _extract_edge_attr(self, snap: dict) -> np.ndarray:
        if self.edge_attr_getter is not None:
            return self.edge_attr_getter(snap)
        if isinstance(snap, dict) and 'edge_attr' in snap:
            candidate = snap['edge_attr']
        elif isinstance(snap, (list, tuple)) and len(snap) > 1:
            candidate = snap[1]
        else:
            raise KeyError("Snapshot missing 'edge_attr' entry; supply an edge_attr_getter")
        if not isinstance(candidate, np.ndarray):
            raise TypeError("edge_attr must be a numpy array")
        return candidate

    def _split_cols(self, edge_attr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        F_e = edge_attr.shape[1]
        mask = np.zeros(F_e, dtype=bool)
        mask[self.target_cols] = True
        X = edge_attr[:, ~mask]
        y = edge_attr[:, mask]
        return X.astype(np.float32), y.astype(np.float32)

    def __getitem__(self, idx):
        snap = self.snaps[idx]
        edge_attr = self._extract_edge_attr(snap)
        X, y = self._split_cols(edge_attr)
        Xn = ((X - self.in_mean) / self.in_std) if X.shape[1] > 0 else X
        Yn = ((y - self.t_mean) / self.t_std)
        return self.node_features, self.edge_index, torch.from_numpy(Xn), torch.from_numpy(Yn)