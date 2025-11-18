"""Visualization function to plot edge predictions on a map."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from model import EdgeGCN
from dataset import SnapshotDataset
from metrics import denormalize

def visualize_snapshot(model: EdgeGCN, ds: SnapshotDataset, positions: np.ndarray, device: torch.device,
                       t_mean: np.ndarray, t_std: np.ndarray, title: str = 'Predicción target[0]') -> None:
    model.eval()
    with torch.no_grad():
        x, eidx, xf, y_norm = ds[0]
        x = x.to(device); eidx = eidx.to(device); xf = xf.to(device)
        yhat_norm = model(x, eidx, xf).cpu().numpy()
        yhat = denormalize(yhat_norm, t_mean, t_std)
    edge_index_np = eidx.cpu().numpy()
    preds = yhat[:, 0]  
    segs = []
    for u, v in zip(edge_index_np[0], edge_index_np[1]):
        p1 = positions[u]; p2 = positions[v]
        segs.append([p1, p2])
    segs = np.array(segs, dtype=np.float32)
    vmin, vmax = np.percentile(preds, [2, 98])
    if vmax <= vmin: vmax = vmin + 1e-6
    cmap = plt.get_cmap('viridis')
    norm_vals = np.clip((preds - vmin) / (vmax - vmin + 1e-9), 0, 1)
    colors = cmap(norm_vals)
    lc = LineCollection(segs, colors=colors, linewidths=1.0)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel('lon'); ax.set_ylabel('lat')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label('valor')
    plt.show()