"""Shared plotting helpers for training/evaluation reports."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ERROR_METRIC_ORDER = ['MAE', 'RMSE', 'MSE']


def ensure_reports_dir(dataset_dir: str, subfolder: str = 'reports') -> str:
    """Create (if needed) and return the directory where plots will be stored."""
    path = os.path.join(dataset_dir, subfolder)
    os.makedirs(path, exist_ok=True)
    return path


def _finalize_figure(fig: plt.Figure, show: bool) -> None:
    if show:
        fig.show()
    else:
        plt.close(fig)


def plot_metric_summary(
    metrics: Dict[str, float],
    output_path: str,
    *,
    accuracy_key: str = 'R2',
    show: bool = False,
    title: str = 'Metric summary',
) -> None:
    """Render a bar chart for error metrics plus a highlighted accuracy band."""
    fig, ax_err = plt.subplots(figsize=(6, 4), dpi=140)
    error_keys = [k for k in ERROR_METRIC_ORDER if k in metrics]
    error_vals = [metrics[k] for k in error_keys]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][: len(error_keys)]
    if error_keys:
        ax_err.bar(error_keys, error_vals, color=colors)
        ax_err.set_ylabel('Error magnitude')
    ax_err.set_title(title)
    accuracy_val = metrics.get(accuracy_key)
    if accuracy_val is not None:
        ax_acc = ax_err.twinx()
        margin = max(0.05 * abs(accuracy_val), 0.05)
        lower = min(accuracy_val - margin, 0.0)
        upper = max(accuracy_val + margin, 1.0)
        if lower == upper:
            upper = lower + 1.0
        ax_acc.set_ylim(lower, upper)
        ax_acc.set_ylabel(f'{accuracy_key} (accuracy proxy)')
        ax_acc.axhline(accuracy_val, color='#d62728', linestyle='--', linewidth=1.5)
        ax_acc.text(0.98, 0.92, f"{accuracy_key} = {accuracy_val:.3f}", transform=ax_acc.transAxes,
                    ha='right', va='center', color='#d62728', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    _finalize_figure(fig, show)


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    *,
    target_idx: int = 0,
    sample_size: int = 4000,
    show: bool = False,
) -> None:
    if y_true.size == 0 or y_pred.size == 0:
        return
    target_idx = max(0, min(target_idx, y_true.shape[1] - 1))
    rng = np.random.default_rng(0)
    idx = np.arange(y_true.shape[0])
    if y_true.shape[0] > sample_size:
        idx = rng.choice(y_true.shape[0], size=sample_size, replace=False)
    sub_true = y_true[idx, target_idx]
    sub_pred = y_pred[idx, target_idx]
    lo = float(min(sub_true.min(), sub_pred.min()))
    hi = float(max(sub_true.max(), sub_pred.max()))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    ax.scatter(sub_true, sub_pred, s=6, alpha=0.6, color='#1f77b4')
    ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=1.0)
    ax.set_xlabel('Ground truth')
    ax.set_ylabel('Prediction')
    ax.set_title(f'Scatter target[{target_idx}]')
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    _finalize_figure(fig, show)


def plot_history_curves(
    history: List[Dict[str, float]],
    output_dir: str,
    save_prefix: str,
    *,
    accuracy_key: str = 'val_r2',
    accuracy_label: str = 'Validation R²',
    show: bool = False,
) -> List[str]:
    if not history:
        return []
    epochs = [entry['epoch'] for entry in history]
    train_loss = [entry['train_loss'] for entry in history]
    val_loss = [entry['val_loss'] for entry in history]
    outputs: List[str] = []

    fig_loss, ax_loss = plt.subplots(figsize=(6, 4), dpi=140)
    ax_loss.plot(epochs, train_loss, label='Train loss')
    ax_loss.plot(epochs, val_loss, label='Validation loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('MSE loss')
    ax_loss.set_title('Loss per epoch')
    ax_loss.legend()
    fig_loss.tight_layout()
    loss_path = os.path.join(output_dir, f"{save_prefix}_loss_curve.png")
    fig_loss.savefig(loss_path, bbox_inches='tight')
    outputs.append(loss_path)
    _finalize_figure(fig_loss, show)

    val_acc = [entry.get(accuracy_key) for entry in history]
    if all(v is not None for v in val_acc):
        fig_acc, ax_acc = plt.subplots(figsize=(6, 4), dpi=140)
        ax_acc.plot(epochs, val_acc, color='#d62728')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel(accuracy_label)
        ax_acc.set_title(f'{accuracy_label} progression')
        lo = min(val_acc)
        hi = max(val_acc)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        ax_acc.set_ylim(lo - 0.05, hi + 0.05)
        fig_acc.tight_layout()
        acc_path = os.path.join(output_dir, f"{save_prefix}_accuracy_curve.png")
        fig_acc.savefig(acc_path, bbox_inches='tight')
        outputs.append(acc_path)
        _finalize_figure(fig_acc, show)

    return outputs