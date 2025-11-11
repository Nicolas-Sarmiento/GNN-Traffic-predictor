"""Train an EdgeGCN on the original road graph using data in gnn_dataset.

Features:
- Loads edge_index.npy, node_features.npy, snapshots (train/val) from gnn_dataset.
- Auto-detects snapshot pickle files (train_snapshots.pkl / val_snapshots.pkl) or falls back to temporal_snapshots.pkl + train/val index CSVs.
- Builds positions.npy (lon/lat) if missing, using node_features if plausible or data/nodes.geojson.
- Allows selecting target columns by name or index (configure TARGET_COLS or TARGET_NAME_SUBSTRINGS).
- Normalizes inputs & targets using train-only statistics (mean/std) with zero std protection.
- EdgeGCN architecture: node GCN layers + per-edge MLP combining (h_u, h_v, edge_features).
- Computes per-epoch train/val loss; every N epochs prints MAE/MSE/RMSE des-normalized.
- Early stopping by validation MSE.
- Saves artefacts: best_model.pt, model_ts.pt (TorchScript), scalers.json, positions.npy.
- Produces a matplotlib visualization coloring edges by predicted target[0] for first validation snapshot.

Run:
    python train_edge_gcn.py --epochs 80 --target-cols 0 1 --val-metrics-interval 5

Optional arguments show full list with -h.
"""
from __future__ import annotations
import os
import json
import math
import argparse
import pickle
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
try:
    from torch_geometric.nn import GCNConv
except ImportError as e:
    raise SystemExit("PyTorch Geometric no instalado. Instala antes de ejecutar este script.")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# Configuration / Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train EdgeGCN on gnn_dataset")
    p.add_argument('--epochs', type=int, default=200, help='Max training epochs')
    p.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--weight-decay', type=float, default=1e-5, help='Adam weight decay')
    p.add_argument('--hidden', type=int, default=96, help='Hidden size for GCN + MLP')
    p.add_argument('--gcn-layers', type=int, default=2, help='Number of GCNConv layers')
    p.add_argument('--dropout', type=float, default=0.2, help='Dropout in MLP')
    p.add_argument('--val-metrics-interval', type=int, default=5, help='Interval epochs to compute full val metrics')
    p.add_argument('--target-cols', type=int, nargs='*', default=None, help='Explicit target column indices inside edge_attr (if not using name substrings)')
    p.add_argument('--target-name-substrings', nargs='*', default=None, help='Substrings to match in feature_names for selecting targets')
    p.add_argument('--feature-names-json', default=None, help='Optional JSON file with list of feature names corresponding to edge_attr columns. If relative path, resolved inside gnn_dataset.')
    p.add_argument('--edge-attr-key', default=None, help='Explicit key inside snapshot dict to use as edge attribute matrix (overrides auto-detect).')
    p.add_argument('--save-prefix', default='edge_gcn', help='Prefix for saved artifacts')
    p.add_argument('--device', default=None, help='Force device (cpu/cuda) else auto')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------

def load_snapshots(dataset_dir: str) -> Tuple[List[dict], List[dict]]:
    train_pkl = os.path.join(dataset_dir, 'train_snapshots.pkl')
    val_pkl   = os.path.join(dataset_dir, 'val_snapshots.pkl')
    if os.path.exists(train_pkl) and os.path.exists(val_pkl):
        train_snaps = pickle.load(open(train_pkl, 'rb'))
        val_snaps = pickle.load(open(val_pkl, 'rb'))
        print(f"Loaded train/val snapshots pickles: {len(train_snaps)} / {len(val_snaps)}")
        return train_snaps, val_snaps
    # Fallback: temporal_snapshots + CSV indices
    temporal_pkl = os.path.join(dataset_dir, 'temporal_snapshots.pkl')
    if not os.path.exists(temporal_pkl):
        raise FileNotFoundError("Neither train_snapshots.pkl/val_snapshots.pkl nor temporal_snapshots.pkl found.")
    all_snaps = pickle.load(open(temporal_pkl, 'rb'))
    train_csv = os.path.join(dataset_dir, 'train_index.csv')
    val_csv   = os.path.join(dataset_dir, 'val_index.csv')
    if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
        raise FileNotFoundError("train_index.csv / val_index.csv missing for split.")
    train_idx = np.loadtxt(train_csv, delimiter=',', dtype=int)
    val_idx   = np.loadtxt(val_csv, delimiter=',', dtype=int)
    if train_idx.ndim == 0: train_idx = np.array([train_idx])
    if val_idx.ndim == 0: val_idx = np.array([val_idx])
    train_snaps = [all_snaps[i] for i in train_idx]
    val_snaps   = [all_snaps[i] for i in val_idx]
    print(f"Loaded temporal_snapshots.pkl and split into train={len(train_snaps)} val={len(val_snaps)}")
    return train_snaps, val_snaps

# ---------------------------------------------------------------------------
# Positions Handling
# ---------------------------------------------------------------------------

def build_positions(dataset_dir: str, node_features: np.ndarray) -> np.ndarray:
    pos_path = os.path.join(dataset_dir, 'positions.npy')
    if os.path.exists(pos_path):
        return np.load(pos_path)
    # Try to interpret node_features as lon/lat
    if node_features.shape[1] == 2:
        lon = node_features[:,0]; lat = node_features[:,1]
        if np.all((lon >= -180) & (lon <= 180)) and np.all((lat >= -90) & (lat <= 90)):
            np.save(pos_path, node_features.astype(np.float32))
            print("positions.npy inferred from node_features")
            return node_features.astype(np.float32)
    # Else read nodes.geojson + graph_structure
    graph_struct = os.path.join(dataset_dir, 'graph_structure.json')
    # Prefer nodes.geojson inside gnn_dataset; fallback to ../data/nodes.geojson
    nodes_geojson = os.path.join(dataset_dir, 'nodes.geojson')
    if not os.path.exists(nodes_geojson):
        repo_root = os.path.abspath(os.path.join(dataset_dir, os.pardir))
        candidates = [
            os.path.join(repo_root, 'data', 'nodes.geojson'),
        ]
        nodes_geojson = next((p for p in candidates if os.path.exists(p)), nodes_geojson)
    if not (os.path.exists(graph_struct) and os.path.exists(nodes_geojson)):
        raise FileNotFoundError("Cannot build positions: node_features not lon/lat and missing graph_structure.json or nodes.geojson (expected inside gnn_dataset or ../data)")
    gs = json.load(open(graph_struct, 'r', encoding='utf-8'))
    node_ids = gs.get('node_ids')
    if not isinstance(node_ids, list) or len(node_ids) != node_features.shape[0]:
        raise ValueError("graph_structure.json node_ids list missing or length mismatch with node_features")
    id_to_idx = {str(nid): i for i, nid in enumerate(node_ids)}
    gj = json.load(open(nodes_geojson, 'r', encoding='utf-8'))
    geo_map = {}
    for feat in gj.get('features', []):
        props = feat.get('properties', {})
        nid = props.get('node_id', props.get('osmid'))
        coords = feat.get('geometry', {}).get('coordinates')
        if nid is not None and coords and len(coords) == 2:
            geo_map[str(nid)] = (float(coords[0]), float(coords[1]))
    positions = np.full((node_features.shape[0], 2), np.nan, dtype=np.float32)
    hits = 0
    for nid_str, idx in id_to_idx.items():
        if nid_str in geo_map:
            positions[idx] = geo_map[nid_str]; hits += 1
    if np.isnan(positions[:,0]).any():
        missing = np.isnan(positions[:,0]).sum()
        raise RuntimeError(f"Missing coordinates for {missing} nodes while building positions.npy")
    np.save(pos_path, positions)
    print(f"positions.npy built from nodes.geojson with {hits} nodes")
    return positions

# ---------------------------------------------------------------------------
# Feature / Target Selection
# ---------------------------------------------------------------------------

def resolve_target_cols(edge_attr: np.ndarray, args, feature_names: List[str] | None) -> List[int]:
    if args.target_cols is not None and len(args.target_cols) > 0:
        return args.target_cols
    if args.target_name_substrings and feature_names:
        substrings = [s.lower() for s in args.target_name_substrings]
        sel = []
        for i, name in enumerate(feature_names):
            nlow = name.lower()
            if any(sub in nlow for sub in substrings):
                sel.append(i)
        if not sel:
            raise ValueError("No feature names matched target_name_substrings")
        return sel
    # Fallback: first two columns as targets
    if edge_attr.shape[1] < 2:
        raise ValueError("Edge attributes have fewer than 2 columns; specify --target-cols explicitly")
    return [0, 1]

# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

class EdgeGCN(nn.Module):
    def __init__(self, in_node: int, in_edge: int, hidden: int, gcn_layers: int, dropout: float, out_dim: int):
        super().__init__()
        self.gcn_in = GCNConv(in_node, hidden)
        self.gcn_layers = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(gcn_layers - 1)])
        mlp_in = hidden * 2 + in_edge
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn_in(x, edge_index))
        for conv in self.gcn_layers:
            h = F.relu(conv(h, edge_index))
        u, v = edge_index[0], edge_index[1]
        hu, hv = h[u], h[v]
        if edge_feat.numel() == 0:
            z = torch.cat([hu, hv], dim=-1)
        else:
            z = torch.cat([hu, hv, edge_feat], dim=-1)
        return self.mlp(z)

# ---------------------------------------------------------------------------
# Metrics / Helpers
# ---------------------------------------------------------------------------

def compute_stats_train(train_snaps: List[dict], target_cols: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_snapshot(model: EdgeGCN, ds: SnapshotDataset, positions: np.ndarray, device: torch.device,
                       t_mean: np.ndarray, t_std: np.ndarray, title: str = 'Predicción target[0]') -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    model.eval()
    with torch.no_grad():
        x, eidx, xf, y_norm = ds[0]
        x = x.to(device); eidx = eidx.to(device); xf = xf.to(device)
        yhat_norm = model(x, eidx, xf).cpu().numpy()
        yhat = denormalize(yhat_norm, t_mean, t_std)
    edge_index_np = eidx.cpu().numpy()
    preds = yhat[:, 0]  # first target component
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

# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Resolve dataset_dir flexibly: support script placed at repo root OR inside gnn_dataset OR running from gnn_dataset cwd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    # 1) script_dir/gnn_dataset
    candidates.append(os.path.join(script_dir, 'gnn_dataset'))
    # 2) script_dir (if script is inside gnn_dataset)
    candidates.append(script_dir)
    # 3) current working directory (if running from inside gnn_dataset)
    candidates.append(os.getcwd())
    dataset_dir = None
    for c in candidates:
        if os.path.isfile(os.path.join(c, 'edge_index.npy')) and os.path.isfile(os.path.join(c, 'node_features.npy')):
            dataset_dir = c
            break
    if dataset_dir is None:
        raise FileNotFoundError("Could not locate gnn_dataset: expected edge_index.npy and node_features.npy in script folder, its gnn_dataset subfolder, or current working directory.")
    edge_index = np.load(os.path.join(dataset_dir, 'edge_index.npy'))  # (2,E)
    node_features = np.load(os.path.join(dataset_dir, 'node_features.npy'))  # (N,F_node)
    train_snaps, val_snaps = load_snapshots(dataset_dir)

    # ------------------------------------------------------------------
    # Robust extraction of edge attribute matrices from snapshot objects
    # ------------------------------------------------------------------
    expected_edges = edge_index.shape[1]

    def extract_edge_attr(snap):
        """Return (edge_attr_array, key_used) raising error if not found.
        Accepts dict or tuple/list; picks first 2D ndarray with first dim == E or second dim == E (transpose if needed).
        """
        # If user provided key explicitly
        if args.edge_attr_key:
            if isinstance(snap, dict) and args.edge_attr_key in snap:
                arr = snap[args.edge_attr_key]
            else:
                raise KeyError(f"edge-attr-key '{args.edge_attr_key}' not found in snapshot dict")
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                raise ValueError(f"edge-attr-key '{args.edge_attr_key}' is not a 2D numpy array")
            if arr.shape[0] == expected_edges:
                return arr, args.edge_attr_key
            if arr.shape[1] == expected_edges:
                return arr.T, args.edge_attr_key + "(transposed)"
            raise ValueError(f"edge-attr-key '{args.edge_attr_key}' shape {arr.shape} does not align with E={expected_edges}")

        # Auto-detect
        candidates = []
        if isinstance(snap, dict):
            for k, v in snap.items():
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    if v.shape[0] == expected_edges or v.shape[1] == expected_edges:
                        candidates.append((k, v))
        elif isinstance(snap, (list, tuple)):
            for idx, v in enumerate(snap):
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    if v.shape[0] == expected_edges or v.shape[1] == expected_edges:
                        candidates.append((f"pos_{idx}", v))
        else:
            raise TypeError("Snapshot type not supported for edge attribute extraction")

        if not candidates:
            raise KeyError("No candidate 2D arrays matching edge count found in first snapshot. Provide --edge-attr-key explicitly.")
        # Prefer array with first dimension == expected_edges
        for k, v in candidates:
            if v.shape[0] == expected_edges:
                return v, k
        # Else transpose the first candidate whose second dimension matches expected_edges
        k, v = candidates[0]
        if v.shape[1] == expected_edges:
            return v.T, k + "(transposed)"
        raise RuntimeError("Failed to align any candidate array to edge count")

    first_edge_attr, used_key = extract_edge_attr(train_snaps[0])
    print(f"Detected edge attribute key: {used_key} shape={first_edge_attr.shape}")

    # Detect feature names if provided
    feature_names = None
    # If feature names provided, load them now (after detecting first_edge_attr)
    if args.feature_names_json:
        # Resolve relative path inside dataset_dir
        feat_path = args.feature_names_json
        if not os.path.isabs(feat_path):
            feat_path = os.path.join(dataset_dir, feat_path)
        if os.path.exists(feat_path):
            with open(feat_path, 'r', encoding='utf-8') as f:
                feature_names = json.load(f)
            if not isinstance(feature_names, list):
                raise ValueError("feature_names_json must hold a JSON list")
        else:
            raise FileNotFoundError(f"Feature names file not found: {feat_path}")

    target_cols = resolve_target_cols(first_edge_attr, args, feature_names)
    print(f"Target column indices: {target_cols}")

    # Recompute train stats using robust extraction per snapshot
    def get_edge_attr_matrix(snap):
        arr, _ = extract_edge_attr(snap)
        return arr
    X_all = []
    Y_all = []
    for snap in train_snaps:
        edge_attr_mat = get_edge_attr_matrix(snap)
        F_e = edge_attr_mat.shape[1]
        mask = np.zeros(F_e, dtype=bool); mask[target_cols] = True
        X_part = edge_attr_mat[:, ~mask]
        Y_part = edge_attr_mat[:, mask]
        X_all.append(X_part); Y_all.append(Y_part)
    X_concat = np.concatenate(X_all, axis=0) if X_all and X_all[0].shape[1] > 0 else np.zeros((len(Y_all), 0), dtype=np.float32)
    Y_concat = np.concatenate(Y_all, axis=0)
    eps = 1e-6
    in_mean = X_concat.mean(0) if X_concat.shape[1] > 0 else np.array([], dtype=np.float32)
    in_std  = X_concat.std(0) if X_concat.shape[1] > 0 else np.array([], dtype=np.float32)
    if in_std.size: in_std[in_std < eps] = 1.0
    t_mean = Y_concat.mean(0); t_std = Y_concat.std(0); t_std[t_std < eps] = 1.0
    print("Input feature dim (edge):", (first_edge_attr.shape[1] - len(target_cols)))
    print("Target dim:", len(target_cols))

    # Save scalers
    scalers_path = os.path.join(dataset_dir, f'{args.save_prefix}_scalers.json')
    with open(scalers_path, 'w', encoding='utf-8') as f:
        json.dump({
            'target_cols': target_cols,
            'in_mean': in_mean.tolist(),
            'in_std': in_std.tolist(),
            't_mean': t_mean.tolist(),
            't_std': t_std.tolist(),
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved scalers to {scalers_path}")

    # Build datasets
    # Wrap SnapshotDataset but override internal split extractor with new logic
    class RobustSnapshotDataset(SnapshotDataset):
        def __getitem__(self, idx):
            snap = self.snaps[idx]
            edge_attr_mat, _ = extract_edge_attr(snap)
            F_e = edge_attr_mat.shape[1]
            mask = np.zeros(F_e, dtype=bool); mask[self.target_cols] = True
            X = edge_attr_mat[:, ~mask]
            y = edge_attr_mat[:, mask]
            Xn = ((X - self.in_mean) / self.in_std) if X.shape[1] > 0 else X
            Yn = ((y - self.t_mean) / self.t_std)
            return self.node_features, self.edge_index, torch.from_numpy(Xn), torch.from_numpy(Yn)

    train_ds = RobustSnapshotDataset(train_snaps, edge_index, node_features, in_mean, in_std, t_mean, t_std, target_cols)
    val_ds = RobustSnapshotDataset(val_snaps, edge_index, node_features, in_mean, in_std, t_mean, t_std, target_cols)

    model = EdgeGCN(node_features.shape[1], (first_edge_attr.shape[1] - len(target_cols)),
                    hidden=args.hidden, gcn_layers=args.gcn_layers, dropout=args.dropout,
                    out_dim=len(target_cols)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float('inf'); best_state = None; bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = 0.0
        for i in range(len(train_ds)):
            x, eidx, xf, y = train_ds[i]
            x=x.to(device); eidx=eidx.to(device); xf=xf.to(device); y=y.to(device)
            optim.zero_grad()
            pred = model(x, eidx, xf)
            loss = loss_fn(pred, y)
            loss.backward(); optim.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_ds))

        # Validation loss
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for i in range(len(val_ds)):
                x, eidx, xf, y = val_ds[i]
                pred = model(x.to(device), eidx.to(device), xf.to(device))
                val_loss += loss_fn(pred.cpu(), y).item()
        val_loss /= max(1, len(val_ds))

        metrics_str = ''
        if epoch % args.val_metrics_interval == 0 or epoch == 1:
            val_metrics = evaluate_denorm(model, val_ds, device, t_mean, t_std)
            metrics_str = f" | val_MAE={val_metrics['MAE']:.3f} val_RMSE={val_metrics['RMSE']:.3f}" 
        print(f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}{metrics_str}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    best_model_path = os.path.join(dataset_dir, f'{args.save_prefix}_best_model.pt')
    torch.save(model.state_dict(), best_model_path)
    print(f"Saved best model to {best_model_path}")

    # TorchScript export (use full node feature matrix to avoid index OOB)
    model.eval()
    sample_edges = min(edge_index.shape[1], 200)
    # Important: use the FULL node matrix so edge indices always fall within bounds
    ex_x = torch.from_numpy(node_features).float().to(device)
    ex_e = torch.from_numpy(edge_index[:, :sample_edges]).long().to(device)
    # Build dummy edge feats aligned with sample (use zeros if no edge features input)
    edge_feat_dim = first_edge_attr.shape[1] - len(target_cols)
    if edge_feat_dim > 0:
        ex_xf = torch.zeros((sample_edges, edge_feat_dim), dtype=torch.float32).to(device)
    else:
        ex_xf = torch.zeros((sample_edges, 0), dtype=torch.float32).to(device)
    try:
        scripted = torch.jit.trace(model, (ex_x, ex_e, ex_xf))
        ts_path = os.path.join(dataset_dir, f'{args.save_prefix}_model_ts.pt')
        scripted.save(ts_path)
        print(f"Saved TorchScript model to {ts_path}")
    except Exception as e:
        print(f"Warning: TorchScript trace failed: {e}. Skipping export.")

    # Build positions for visualization
    positions = build_positions(dataset_dir, node_features)

    # Final validation metrics (denorm)
    final_val_metrics = evaluate_denorm(model, val_ds, device, t_mean, t_std)
    print("Final validation metrics (denormalized):", final_val_metrics)

    # Visualization
    visualize_snapshot(model, val_ds, positions, device, t_mean, t_std, title='Predicción target[0] (val snapshot 0)')

if __name__ == '__main__':
    main()
