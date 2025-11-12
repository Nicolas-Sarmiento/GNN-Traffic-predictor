"""Configuration and argument parsing for EdgeGCN training."""

import argparse

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