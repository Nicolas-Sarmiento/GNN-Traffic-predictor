"""Training script for the EdgeGCN model.

This module now focuses strictly on data preparation and training. Model
evaluation, metrics visualization, and reporting are handled by the new
`evaluate.py` entrypoint.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from config import parse_args
from data_utils import load_snapshots, resolve_target_cols
from dataset import SnapshotDataset
from metrics import evaluate_denorm
from model import EdgeGCN
from pipeline_utils import (
    build_edge_attr_extractor,
    compute_scalers,
    find_edge_attr_in_snapshot,
    load_feature_names,
    locate_dataset_dir,
    save_scaler_bundle,
)
from reporting import ensure_reports_dir, plot_history_curves, plot_metric_summary

# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_roots = [
        os.path.join(script_dir, 'gnn_dataset'),
        script_dir,
        os.getcwd(),
    ]
    dataset_dir = locate_dataset_dir(args.dataset_dir, search_roots)
    edge_index = np.load(os.path.join(dataset_dir, 'edge_index.npy'))  # (2,E)
    node_features = np.load(os.path.join(dataset_dir, 'node_features.npy'))  # (N,F_node)
    train_snaps, val_snaps = load_snapshots(dataset_dir)

    expected_edges = edge_index.shape[1]
    extract_edge_attr = build_edge_attr_extractor(expected_edges, args.edge_attr_key)
    first_edge_attr, used_key = find_edge_attr_in_snapshot(train_snaps[0], expected_edges, args.edge_attr_key)
    print(f"Detected edge attribute source: {used_key} shape={first_edge_attr.shape}")

    feature_names = load_feature_names(dataset_dir, args.feature_names_json)

    target_cols = resolve_target_cols(first_edge_attr, args, feature_names)
    print(f"Target column indices: {target_cols}")

    scalers = compute_scalers(train_snaps, target_cols, extract_edge_attr)
    print("Input feature dim (edge):", (first_edge_attr.shape[1] - len(target_cols)))
    print("Target dim:", len(target_cols))

    scalers_path = os.path.join(dataset_dir, f'{args.save_prefix}_scalers.json')
    metadata: Dict[str, List[int] | Dict[str, float | int | None] | str | None] = {
        'target_cols': target_cols,
        'edge_attr_key': args.edge_attr_key,
        'detected_edge_attr_key': used_key,
        'model_config': {
            'hidden': args.hidden,
            'gcn_layers': args.gcn_layers,
            'dropout': args.dropout,
        },
    }
    save_scaler_bundle(scalers_path, scalers, metadata)
    print(f"Saved scalers to {scalers_path}")

    train_ds = SnapshotDataset(
        train_snaps,
        edge_index,
        node_features,
        scalers['in_mean'],
        scalers['in_std'],
        scalers['t_mean'],
        scalers['t_std'],
        target_cols,
        edge_attr_getter=extract_edge_attr,
    )
    val_ds = SnapshotDataset(
        val_snaps,
        edge_index,
        node_features,
        scalers['in_mean'],
        scalers['in_std'],
        scalers['t_mean'],
        scalers['t_std'],
        target_cols,
        edge_attr_getter=extract_edge_attr,
    )

    edge_feature_dim = first_edge_attr.shape[1] - len(target_cols)
    model = EdgeGCN(node_features.shape[1], edge_feature_dim,
                    hidden=args.hidden, gcn_layers=args.gcn_layers, dropout=args.dropout,
                    out_dim=len(target_cols)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for i in range(len(train_ds)):
            x, eidx, xf, y = train_ds[i]
            x = x.to(device); eidx = eidx.to(device); xf = xf.to(device); y = y.to(device)
            optim.zero_grad()
            pred = model(x, eidx, xf)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_ds))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(len(val_ds)):
                x, eidx, xf, y = val_ds[i]
                pred = model(x.to(device), eidx.to(device), xf.to(device))
                val_loss += loss_fn(pred.cpu(), y).item()
        val_loss /= max(1, len(val_ds))

        val_metrics = evaluate_denorm(model, val_ds, device, scalers['t_mean'], scalers['t_std'])
        metrics_str = ''
        if epoch == 1 or epoch % max(1, args.val_metrics_interval) == 0:
            metrics_str = (
                f" | val_MAE={val_metrics['MAE']:.3f}"
                f" val_RMSE={val_metrics['RMSE']:.3f}"
                f" val_R2={val_metrics['R2']:.3f}"
            )
        print(f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}{metrics_str}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mae': val_metrics['MAE'],
            'val_mse': val_metrics['MSE'],
            'val_rmse': val_metrics['RMSE'],
            'val_r2': val_metrics['R2'],
        })

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    best_model_path = os.path.join(dataset_dir, f'{args.save_prefix}_best_model.pt')
    torch.save(model.state_dict(), best_model_path)
    print(f"Saved best model to {best_model_path}")

    history_path = os.path.join(dataset_dir, f'{args.save_prefix}_history.json')
    with open(history_path, 'w', encoding='utf-8') as handle:
        json_history = [{k: float(v) if isinstance(v, (int, float)) else v for k, v in entry.items()} for entry in history]
        json.dump(json_history, handle, ensure_ascii=False, indent=2)
    print(f"Saved training history to {history_path}")

    final_metrics = evaluate_denorm(model, val_ds, device, scalers['t_mean'], scalers['t_std'])
    print("Final validation metrics (denormalized):", final_metrics)

    reports_dir = ensure_reports_dir(dataset_dir)
    plot_history_curves(history, reports_dir, args.save_prefix, show=args.show_plots)
    summary_path = os.path.join(reports_dir, f"{args.save_prefix}_training_metrics.png")
    plot_metric_summary(final_metrics, summary_path, accuracy_key='R2', show=args.show_plots, title='Final validation metrics')
    print(f"Saved training plots to {reports_dir}")

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

    print("Training complete. Run evaluate.py to generate evaluation metrics and plots.")

if __name__ == '__main__':
    main()