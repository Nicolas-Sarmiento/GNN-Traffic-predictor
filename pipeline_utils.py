"""Shared helpers for the EdgeGCN training/evaluation pipeline."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

Snapshot = Any


def locate_dataset_dir(preferred: Optional[str], search_roots: Optional[Iterable[str]] = None) -> str:
    """Return the first directory containing the expected numpy artifacts."""
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    if search_roots:
        for root in search_roots:
            if root and root not in candidates:
                candidates.append(root)
    for candidate in candidates:
        if not candidate:
            continue
        if not os.path.isdir(candidate):
            continue
        edge_idx = os.path.join(candidate, 'edge_index.npy')
        node_feat = os.path.join(candidate, 'node_features.npy')
        if os.path.isfile(edge_idx) and os.path.isfile(node_feat):
            return candidate
    raise FileNotFoundError(
        "Could not locate dataset directory. Provide --dataset-dir explicitly or place edge_index.npy/node_features.npy in a searchable folder."
    )


def load_feature_names(dataset_dir: str, feature_names_json: Optional[str]) -> Optional[List[str]]:
    if not feature_names_json:
        return None
    feat_path = feature_names_json
    if not os.path.isabs(feat_path):
        feat_path = os.path.join(dataset_dir, feat_path)
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature names file not found: {feat_path}")
    with open(feat_path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("feature_names_json must contain a JSON list")
    return data


def _candidate_edge_arrays(snap: Snapshot) -> List[Tuple[str, np.ndarray]]:
    candidates: List[Tuple[str, np.ndarray]] = []
    if isinstance(snap, dict):
        for key, value in snap.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                candidates.append((key, value))
    elif isinstance(snap, (list, tuple)):
        for idx, value in enumerate(snap):
            if isinstance(value, np.ndarray) and value.ndim == 2:
                candidates.append((f"pos_{idx}", value))
    else:
        raise TypeError("Snapshot type not supported for edge attribute extraction")
    return candidates


def find_edge_attr_in_snapshot(
    snap: Snapshot,
    expected_edges: int,
    edge_attr_key: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """Return the edge attribute matrix aligned to (E, F) and a description of the source."""
    if edge_attr_key:
        if not isinstance(snap, dict) or edge_attr_key not in snap:
            raise KeyError(f"edge-attr-key '{edge_attr_key}' not found in snapshot")
        arr = snap[edge_attr_key]
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(f"edge-attr-key '{edge_attr_key}' is not a 2D numpy array")
        if arr.shape[0] == expected_edges:
            return arr, edge_attr_key
        if arr.shape[1] == expected_edges:
            return arr.T, f"{edge_attr_key}(transposed)"
        raise ValueError(
            f"edge-attr-key '{edge_attr_key}' shape {arr.shape} does not align with expected edge count {expected_edges}"
        )

    candidates = _candidate_edge_arrays(snap)
    if not candidates:
        raise KeyError("No candidate 2D arrays matching edge count found in snapshot. Provide --edge-attr-key explicitly.")
    for key, value in candidates:
        if value.shape[0] == expected_edges:
            return value, key
    for key, value in candidates:
        if value.shape[1] == expected_edges:
            return value.T, f"{key}(transposed)"
    raise RuntimeError("Failed to align any candidate array to the expected edge count")


def build_edge_attr_extractor(
    expected_edges: int,
    edge_attr_key: Optional[str] = None,
) -> Callable[[Snapshot], np.ndarray]:
    def extractor(snap: Snapshot) -> np.ndarray:
        edge_attr, _ = find_edge_attr_in_snapshot(snap, expected_edges, edge_attr_key=edge_attr_key)
        return edge_attr

    return extractor


def compute_scalers(
    train_snaps: List[Snapshot],
    target_cols: List[int],
    extract_edge_attr: Callable[[Snapshot], np.ndarray],
) -> Dict[str, np.ndarray]:
    X_all: List[np.ndarray] = []
    Y_all: List[np.ndarray] = []
    for snap in train_snaps:
        edge_attr = extract_edge_attr(snap)
        F_e = edge_attr.shape[1]
        mask = np.zeros(F_e, dtype=bool)
        mask[target_cols] = True
        X = edge_attr[:, ~mask]
        Y = edge_attr[:, mask]
        X_all.append(X)
        Y_all.append(Y)
    X_concat = (
        np.concatenate(X_all, axis=0)
        if X_all and X_all[0].shape[1] > 0
        else np.zeros((len(Y_all), 0), dtype=np.float32)
    )
    Y_concat = np.concatenate(Y_all, axis=0)
    eps = 1e-6
    in_mean = X_concat.mean(0) if X_concat.shape[1] > 0 else np.array([], dtype=np.float32)
    in_std = X_concat.std(0) if X_concat.shape[1] > 0 else np.array([], dtype=np.float32)
    if in_std.size:
        in_std[in_std < eps] = 1.0
    t_mean = Y_concat.mean(0)
    t_std = Y_concat.std(0)
    t_std[t_std < eps] = 1.0
    return {
        'in_mean': in_mean.astype(np.float32),
        'in_std': in_std.astype(np.float32),
        't_mean': t_mean.astype(np.float32),
        't_std': t_std.astype(np.float32),
    }


_NUMERIC_VECTOR_KEYS = {'in_mean', 'in_std', 't_mean', 't_std'}


def save_scaler_bundle(path: str, scalers: Dict[str, np.ndarray], metadata: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {k: v.tolist() for k, v in scalers.items()}
    if metadata:
        payload.update(metadata)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_scaler_bundle(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        data: Dict[str, Any] = json.load(handle)
    for key in _NUMERIC_VECTOR_KEYS:
        if key in data:
            arr = np.array(data[key], dtype=np.float32)
            data[key] = arr
    if 'target_cols' in data:
        data['target_cols'] = [int(idx) for idx in data['target_cols']]
    return data
