"""Main training script for the EdgeGCN model.

Orchestrates data loading, model definition, training, and evaluation
by importing components from other modules.
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

# Local module imports
from config import parse_args
from model import EdgeGCN
from data_utils import load_snapshots, build_positions, resolve_target_cols
from dataset import SnapshotDataset
from metrics import denormalize, evaluate_denorm
from visualization import visualize_snapshot

# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    
    candidates.append(os.path.join(script_dir, 'gnn_dataset'))
    candidates.append(script_dir)
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

    expected_edges = edge_index.shape[1]

    def extract_edge_attr(snap):
        """Return (edge_attr_array, key_used) raising error if not found.
        Accepts dict or tuple/list; picks first 2D ndarray with first dim == E or second dim == E (transpose if needed).
        """

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

        for k, v in candidates:
            if v.shape[0] == expected_edges:
                return v, k

        k, v = candidates[0]
        if v.shape[1] == expected_edges:
            return v.T, k + "(transposed)"
        raise RuntimeError("Failed to align any candidate array to edge count")

    first_edge_attr, used_key = extract_edge_attr(train_snaps[0])
    print(f"Detected edge attribute key: {used_key} shape={first_edge_attr.shape}")

    feature_names = None

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

    model.eval()
    sample_edges = min(edge_index.shape[1], 200)
    ex_x = torch.from_numpy(node_features).float().to(device)
    ex_e = torch.from_numpy(edge_index[:, :sample_edges]).long().to(device)
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

    positions = build_positions(dataset_dir, node_features)

    final_val_metrics = evaluate_denorm(model, val_ds, device, t_mean, t_std)
    print("Final validation metrics (denormalized):", final_val_metrics)

    visualize_snapshot(model, val_ds, positions, device, t_mean, t_std, title='Predicción target[0] (val snapshot 0)')

if __name__ == '__main__':
    main()