"""Functions for model evaluation and result processing."""

import math
import numpy as np
import torch
from typing import List, Tuple, Dict

from model import EdgeGCN
from dataset import SnapshotDataset 

def compute_stats_train(train_snaps: List[dict], target_cols: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Note: This function was present in the original script but not used by main()."""
    X_all = []
    Y_all = []
    for snap in train_snaps:
        edge_attr = snap['edge_attr'] if isinstance(snap, dict) else snap[1]
        F_e = edge_attr.shape[1]
        mask = np.zeros(F_e, dtype=bool); mask[target_cols] = True
        X = edge_attr[:, ~mask]; Y = edge_attr[:, mask]
        X_all.append(X); Y_all.append(Y)
    X_concat = np.concatenate(X_all, axis=0) if X_all and X_all[0].shape[1] > 0 else np.zeros((len(Y_all), 0), dtype=np.float32)
    Y_concat = np.concatenate(Y_all, axis=0)
    eps = 1e-6
    in_mean = X_concat.mean(0) if X_concat.shape[1] > 0 else np.array([], dtype=np.float32)
    in_std  = X_concat.std(0) if X_concat.shape[1] > 0 else np.array([], dtype=np.float32)
    if in_std.size: in_std[in_std < eps] = 1.0
    t_mean = Y_concat.mean(0); t_std = Y_concat.std(0); t_std[t_std < eps] = 1.0
    return in_mean.astype(np.float32), in_std.astype(np.float32), t_mean.astype(np.float32), t_std.astype(np.float32)


def denormalize(Yn: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return Yn * std + mean


def evaluate_denorm(model: EdgeGCN, ds: SnapshotDataset, device: torch.device,
                     t_mean: np.ndarray, t_std: np.ndarray) -> Dict[str, float]:
    model.eval()
    mae_list = []; mse_list = []; rmse_list = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, eidx, xf, y_norm = ds[i]
            x = x.to(device); eidx = eidx.to(device); xf = xf.to(device); y_norm = y_norm.to(device)
            yhat_norm = model(x, eidx, xf).cpu().numpy()
            y_true = denormalize(y_norm.cpu().numpy(), t_mean, t_std)
            y_pred = denormalize(yhat_norm, t_mean, t_std)
            diff = y_pred - y_true
            mae = np.mean(np.abs(diff))
            mse = np.mean(diff**2)
            rmse = math.sqrt(mse)
            mae_list.append(mae); mse_list.append(mse); rmse_list.append(rmse)
    return {
        'MAE': float(np.mean(mae_list)),
        'MSE': float(np.mean(mse_list)),
        'RMSE': float(np.mean(rmse_list)),
    }