"""Evaluation and visualization entrypoint for trained EdgeGCN models."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch

from data_utils import load_snapshots
from dataset import SnapshotDataset
from metrics import denormalize, evaluate_denorm
from model import EdgeGCN
from pipeline_utils import (
    build_edge_attr_extractor,
    find_edge_attr_in_snapshot,
    load_scaler_bundle,
    locate_dataset_dir,
)
from reporting import (
    ensure_reports_dir,
    plot_history_curves,
    plot_metric_summary,
    plot_prediction_scatter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained EdgeGCN model and generate accuracy plots")
    parser.add_argument('--dataset-dir', default=None, help='Folder containing edge_index.npy/node_features.npy')
    parser.add_argument('--save-prefix', default='edge_gcn', help='Prefix used when saving training artifacts')
    parser.add_argument('--model-path', default=None, help='Explicit path to the .pt checkpoint (default: <dataset-dir>/<prefix>_best_model.pt)')
    parser.add_argument('--scalers-path', default=None, help='Explicit path to the scalers JSON (default: <dataset-dir>/<prefix>_scalers.json)')
    parser.add_argument('--history-path', default=None, help='Optional training history JSON for plotting loss curves')
    parser.add_argument('--split', choices=['train', 'val'], default='val', help='Dataset split to evaluate when no snapshot pickle is provided')
    parser.add_argument('--snapshot-pkl', default=None, help='Optional snapshot pickle to override the split selection')
    parser.add_argument('--edge-attr-key', default=None, help='Override automatic edge attribute detection when needed')
    parser.add_argument('--device', default=None, help='Force evaluation device (cpu/cuda) else auto-detect')
    parser.add_argument('--output-dir', default=None, help='Directory to store plots and metrics (default: <dataset-dir>/reports)')
    parser.add_argument('--feature-names-json', default=None, help='Unused placeholder for compatibility (ignored but accepted)')
    parser.add_argument('--hidden', type=int, default=None, help='Override hidden size if metadata missing')
    parser.add_argument('--gcn-layers', type=int, default=None, help='Override GCN layers if metadata missing')
    parser.add_argument('--dropout', type=float, default=None, help='Override dropout if metadata missing')
    parser.add_argument('--prediction-target-idx', type=int, default=0, help='Target column index for scatter plots')
    parser.add_argument('--scatter-sample', type=int, default=4000, help='Max points sampled for scatter plot')
    parser.add_argument('--show-plots', action='store_true', help='Display plots interactively in addition to saving them')
    return parser.parse_args()


def locate_and_load_artifacts(args: argparse.Namespace) -> Tuple[str, Dict[str, object], np.ndarray, np.ndarray]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_roots = [
        os.path.join(script_dir, 'gnn_dataset'),
        script_dir,
        os.getcwd(),
    ]
    dataset_dir = locate_dataset_dir(args.dataset_dir, search_roots)
    scalers_path = args.scalers_path or os.path.join(dataset_dir, f"{args.save_prefix}_scalers.json")
    bundle = load_scaler_bundle(scalers_path)
    edge_index = np.load(os.path.join(dataset_dir, 'edge_index.npy'))
    node_features = np.load(os.path.join(dataset_dir, 'node_features.npy'))
    return dataset_dir, bundle, edge_index, node_features


def resolve_model_config(bundle: Dict[str, object], args: argparse.Namespace) -> Dict[str, float | int]:
    metadata = bundle.get('model_config', {}) if isinstance(bundle.get('model_config'), dict) else {}
    hidden = args.hidden if args.hidden is not None else metadata.get('hidden', 96)
    gcn_layers = args.gcn_layers if args.gcn_layers is not None else metadata.get('gcn_layers', 2)
    dropout = args.dropout if args.dropout is not None else metadata.get('dropout', 0.2)
    return {'hidden': int(hidden), 'gcn_layers': int(gcn_layers), 'dropout': float(dropout)}


def load_snapshots_for_eval(dataset_dir: str, args: argparse.Namespace) -> List[dict]:
    if args.snapshot_pkl:
        with open(args.snapshot_pkl, 'rb') as handle:
            return pickle.load(handle)
    train_snaps, val_snaps = load_snapshots(dataset_dir)
    return train_snaps if args.split == 'train' else val_snaps


def collect_predictions(model: EdgeGCN, ds: SnapshotDataset, device: torch.device, t_mean: np.ndarray, t_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            x, eidx, xf, y_norm = ds[i]
            x = x.to(device); eidx = eidx.to(device); xf = xf.to(device)
            preds = model(x, eidx, xf).cpu().numpy()
            y_true = denormalize(y_norm.numpy(), t_mean, t_std)
            y_pred = denormalize(preds, t_mean, t_std)
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
    if not y_true_all:
        return np.zeros((0, t_mean.shape[0])), np.zeros((0, t_mean.shape[0]))
    return np.concatenate(y_true_all, axis=0), np.concatenate(y_pred_all, axis=0)


def save_metrics_json(metrics: Dict[str, float], output_dir: str, save_prefix: str) -> str:
    path = os.path.join(output_dir, f"{save_prefix}_metrics.json")
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    args = parse_args()
    dataset_dir, bundle, edge_index, node_features = locate_and_load_artifacts(args)
    scalers = {
        'in_mean': bundle['in_mean'],
        'in_std': bundle['in_std'],
        't_mean': bundle['t_mean'],
        't_std': bundle['t_std'],
    }
    target_cols = bundle['target_cols']
    model_cfg = resolve_model_config(bundle, args)
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    snapshots = load_snapshots_for_eval(dataset_dir, args)
    if not snapshots:
        raise RuntimeError('Snapshot list is empty; cannot evaluate.')

    expected_edges = edge_index.shape[1]
    extract_edge_attr = build_edge_attr_extractor(expected_edges, args.edge_attr_key or bundle.get('edge_attr_key'))
    first_edge_attr, used_key = find_edge_attr_in_snapshot(snapshots[0], expected_edges, args.edge_attr_key or bundle.get('edge_attr_key'))
    print(f"Using edge attribute source: {used_key} shape={first_edge_attr.shape}")

    dataset = SnapshotDataset(
        snapshots,
        edge_index,
        node_features,
        scalers['in_mean'],
        scalers['in_std'],
        scalers['t_mean'],
        scalers['t_std'],
        target_cols,
        edge_attr_getter=extract_edge_attr,
    )

    edge_feat_dim = first_edge_attr.shape[1] - len(target_cols)
    model = EdgeGCN(
        node_features.shape[1],
        edge_feat_dim,
        hidden=model_cfg['hidden'],
        gcn_layers=model_cfg['gcn_layers'],
        dropout=model_cfg['dropout'],
        out_dim=len(target_cols),
    ).to(device)
    model_path = args.model_path or os.path.join(dataset_dir, f"{args.save_prefix}_best_model.pt")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded model weights from {model_path}")

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = ensure_reports_dir(dataset_dir)

    metrics = evaluate_denorm(model, dataset, device, scalers['t_mean'], scalers['t_std'])
    metrics_path = save_metrics_json(metrics, output_dir, args.save_prefix)
    print(f"Aggregate metrics saved to {metrics_path}")

    summary_plot = os.path.join(output_dir, f"{args.save_prefix}_metric_summary.png")
    plot_metric_summary(metrics, summary_plot, show=args.show_plots, title='Evaluation metrics')
    print(f"Metric summary plot saved to {summary_plot}")

    y_true, y_pred = collect_predictions(model, dataset, device, scalers['t_mean'], scalers['t_std'])
    scatter_path = os.path.join(output_dir, f"{args.save_prefix}_scatter_target{args.prediction_target_idx}.png")
    plot_prediction_scatter(
        y_true,
        y_pred,
        scatter_path,
        target_idx=args.prediction_target_idx,
        sample_size=args.scatter_sample,
        show=args.show_plots,
    )
    print(f"Prediction scatter saved to {scatter_path}")

    history_path = args.history_path or os.path.join(dataset_dir, f"{args.save_prefix}_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as handle:
            history_data = json.load(handle)
        history_outputs = plot_history_curves(history_data, output_dir, args.save_prefix, show=args.show_plots)
        for path in history_outputs:
            print(f"History plot saved to {path}")
    else:
        print(f"History file not found at {history_path}; skipping training curve plots.")


if __name__ == '__main__':
     main()
