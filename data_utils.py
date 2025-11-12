"""Utility functions for loading and processing dataset files."""

import os
import json
import pickle
import numpy as np
from typing import List, Tuple, Dict, Optional 

def load_snapshots(dataset_dir: str) -> Tuple[List[dict], List[dict]]:
    train_pkl = os.path.join(dataset_dir, 'train_snapshots.pkl')
    val_pkl   = os.path.join(dataset_dir, 'val_snapshots.pkl')
    if os.path.exists(train_pkl) and os.path.exists(val_pkl):
        train_snaps = pickle.load(open(train_pkl, 'rb'))
        val_snaps = pickle.load(open(val_pkl, 'rb'))
        print(f"Loaded train/val snapshots pickles: {len(train_snaps)} / {len(val_snaps)}")
        return train_snaps, val_snaps
    
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

def build_positions(dataset_dir: str, node_features: np.ndarray) -> np.ndarray:
    pos_path = os.path.join(dataset_dir, 'positions.npy')
    if os.path.exists(pos_path):
        return np.load(pos_path)

    if node_features.shape[1] == 2:
        lon = node_features[:,0]; lat = node_features[:,1]
        if np.all((lon >= -180) & (lon <= 180)) and np.all((lat >= -90) & (lat <= 90)):
            np.save(pos_path, node_features.astype(np.float32))
            print("positions.npy inferred from node_features")
            return node_features.astype(np.float32)

    graph_struct = os.path.join(dataset_dir, 'graph_structure.json')
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

def resolve_target_cols(edge_attr: np.ndarray, args, feature_names: Optional[List[str]]) -> List[int]: 
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
        
    if edge_attr.shape[1] < 2:
        raise ValueError("Edge attributes have fewer than 2 columns; specify --target-cols explicitly")
    return [0, 1]