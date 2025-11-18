"""
Script to load a trained EdgeGCN model and perform A* pathfinding.

Uses command-line arguments for start/end OSM IDs and snapshot index.
Generates two plots in a single window:
1. A map coloring all edges by their predicted weight (e.g., travel time).
2. A map highlighting the shortest path found by A*.
"""
from __future__ import annotations
import os
import json
import math
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import argparse  # <-- NEW: Import argparse

# Import from your refactored project files
from model import EdgeGCN
from data_utils import load_snapshots
from metrics import denormalize

# --- Configuration ---
DATASET_DIR = 'gnn_dataset'
MODEL_PATH = os.path.join(DATASET_DIR, 'edge_gcn_best_model.pt')
SCALERS_PATH = os.path.join(DATASET_DIR, 'edge_gcn_scalers.json')
GRAPH_STRUCTURE_PATH = os.path.join(DATASET_DIR, 'graph_structure.json')


# A* Heuristic configuration
MAX_SPEED_KMPH = 30.0 


def parse_path_args():
    p = argparse.ArgumentParser(description="Run A* pathfinding on a trained EdgeGCN model")
    p.add_argument('--start', type=str, required=True, help='OSM ID of the start node.')
    p.add_argument('--end', type=str, required=True, help='OSM ID of the end node.')
    p.add_argument('--snapshot-index', type=int, default=0, help='Index of the validation snapshot to use for predictions (default: 0).')
    return p.parse_args()


def extract_edge_attr(snap, expected_edges, edge_attr_key=None):
    """
    Return (edge_attr_array, key_used) raising error if not found.
    Accepts dict or tuple/list; picks first 2D ndarray with first dim == E
    or second dim == E (transpose if needed).
    """
    if edge_attr_key:
        if isinstance(snap, dict) and edge_attr_key in snap:
            arr = snap[edge_attr_key]
        else:
            raise KeyError(f"edge-attr-key '{edge_attr_key}' not found in snapshot dict")
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(f"edge-attr-key '{edge_attr_key}' is not a 2D numpy array")
        if arr.shape[0] == expected_edges:
            return arr, edge_attr_key
        if arr.shape[1] == expected_edges:
            return arr.T, edge_attr_key + "(transposed)"
        raise ValueError(f"edge-attr-key '{edge_attr_key}' shape {arr.shape} does not align with E={expected_edges}")

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
        raise KeyError(f"No candidate 2D arrays matching edge count E={expected_edges} found in snapshot.")
    
    for k, v in candidates:
        if v.shape[0] == expected_edges:
            return v, k
    k, v = candidates[0]
    if v.shape[1] == expected_edges:
        return v.T, k + "(transposed)"
    
    raise RuntimeError(f"Failed to align any candidate array to edge count E={expected_edges}")

def load_graph_data(dataset_dir: str):
    """Loads graph structure, positions, and OSMID mapping."""
    try:
        edge_index = np.load(os.path.join(dataset_dir, 'edge_index.npy'))
        node_features = np.load(os.path.join(dataset_dir, 'node_features.npy'))
        positions = np.load(os.path.join(dataset_dir, 'positions.npy'))
        
        with open(GRAPH_STRUCTURE_PATH, 'r') as f:
            graph_structure = json.load(f)
        
        node_ids_list = graph_structure.get('node_ids')
        if not node_ids_list:
            raise ValueError("Could not find 'node_ids' in graph_structure.json")
            
        osmid_to_idx = {str(osmid): i for i, osmid in enumerate(node_ids_list)}
        print(f"Loaded OSMID map: {len(osmid_to_idx)} nodes.")

    except FileNotFoundError as e:
        print(f"Error loading base data: {e}")
        print("Please ensure edge_index.npy, node_features.npy, positions.npy, and graph_structure.json exist.")
        raise
    
    return edge_index, node_features, positions, osmid_to_idx

def load_model_and_scalers(model_class, model_path, scalers_path, node_feat_dim, device):
    """Loads scalers and the trained model state."""
    try:
        with open(scalers_path, 'r') as f:
            scalers = json.load(f)
    except FileNotFoundError:
        print(f"Error: Scalers file not found at {scalers_path}")
        raise
        
    t_mean = np.array(scalers['t_mean'])
    t_std = np.array(scalers['t_std'])
    in_mean = np.array(scalers['in_mean'])
    in_std = np.array(scalers['in_std'])
    
    edge_in_dim = len(in_mean)
    out_dim = len(t_mean)
    
    HIDDEN_DIM = 96  
    GCN_LAYERS = 8
    
    model = model_class(
        in_node=node_feat_dim,
        in_edge=edge_in_dim,
        hidden=HIDDEN_DIM,
        gcn_layers=GCN_LAYERS,
        dropout=0.0,
        out_dim=out_dim
    )
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        raise
        
    return model, (in_mean, in_std, t_mean, t_std), scalers

def get_predicted_weights(model, node_features, edge_index, edge_attr_snapshot, scalers, scalers_dict, device):
    """
    Runs the model to predict edge weights for a given snapshot of features.
    """
    in_mean, in_std, t_mean, t_std = scalers
    target_cols = scalers_dict.get('target_cols', [0, 1])
    
    F_e = edge_attr_snapshot.shape[1]
    mask = np.zeros(F_e, dtype=bool)
    mask[target_cols] = True
    
    X = edge_attr_snapshot[:, ~mask].astype(np.float32)
    Xn = ((X - in_mean) / in_std) if X.shape[1] > 0 else X
    
    x_in = torch.from_numpy(node_features).float().to(device)
    eidx_in = torch.from_numpy(edge_index).long().to(device)
    xf_in = torch.from_numpy(Xn).float().to(device)
    
    with torch.no_grad():
        yhat_norm = model(x_in, eidx_in, xf_in).cpu().numpy()
        
    yhat_denorm = denormalize(yhat_norm, t_mean, t_std)
    predicted_weights = yhat_denorm[:, 0]
    predicted_weights[predicted_weights < 0] = 0
    
    return predicted_weights

def build_nx_graph(edge_index, predicted_weights):
    """Builds a NetworkX DiGraph from edge data and predicted weights."""
    G = nx.DiGraph()
    num_nodes = edge_index.max() + 1
    G.add_nodes_from(range(num_nodes))
    
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    
    for u, v, w in zip(source_nodes, target_nodes, predicted_weights):
        G.add_edge(u.item(), v.item(), weight=w.item())
        
    print(f"Built NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def haversine_heuristic(pos_dict, node1_idx, node2_idx):
    """
    A* heuristic: Calculates Haversine distance and converts to an
    admissible (underestimated) travel time.
    """
    lon1, lat1 = pos_dict[node1_idx]
    lon2, lat2 = pos_dict[node2_idx]
    
    R = 6371 
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance_km = R * c
    
    time_hours = distance_km / MAX_SPEED_KMPH
    time_seconds = time_hours * 3600
    
    return time_seconds

def plot_predicted_weights(ax, pos_dict, edge_index, predicted_weights, title):
    """
    Renders the full graph, coloring every edge by its
    predicted weight (like the visualization in train.py).
    """
    print(f"Generating plot 1: {title}...")
    positions = np.array(list(pos_dict.values()))
    
    segs = []
    for u, v in zip(edge_index[0], edge_index[1]):
        p1 = positions[u]; p2 = positions[v]
        segs.append([p1, p2])
    segs = np.array(segs, dtype=np.float32)

    preds = predicted_weights
    vmin, vmax = np.percentile(preds, [2, 98])
    if vmax <= vmin: vmax = vmin + 1e-6
    
    cmap = plt.get_cmap('viridis')
    norm_vals = np.clip((preds - vmin) / (vmax - vmin + 1e-9), 0, 1)
    colors = cmap(norm_vals)
    
    lc = LineCollection(segs, colors=colors, linewidths=0.5, alpha=0.8)
    
    ax.add_collection(lc)
    ax.autoscale()
    
    padding_factor = 0.08
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_padding = (xlim[1] - xlim[0]) * padding_factor
    y_padding = (ylim[1] - ylim[0]) * padding_factor
    ax.set_xlim(xlim[0] - x_padding, xlim[1] + x_padding)
    ax.set_ylim(ylim[0] - y_padding, ylim[1] + y_padding)
    
    ax.set_aspect('equal', adjustable='datalim')
    
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Predicted Congestion (num cars)')

def plot_path(ax, pos_dict, G, path, title="Graph with Shortest Path"):
    """
    Renders the graph, highlighting the found path in red.
    """
    print(f"Generating plot 2: {title}...")
    
    nx.draw_networkx_nodes(G, pos_dict, ax=ax, node_size=0, node_color='gray')
    nx.draw_networkx_edges(G, pos_dict, ax=ax, width=0.3, edge_color='black', alpha=0.7, arrows=False)
    
    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_nodes(G, pos_dict, ax=ax, nodelist=path, node_size=5, node_color='red')
        nx.draw_networkx_edges(G, pos_dict, ax=ax, edgelist=path_edges, width=1.5, edge_color='red', arrows=False)
        
        nx.draw_networkx_nodes(G, pos_dict, ax=ax, nodelist=[path[0]], node_size=50, node_color='green')
        nx.draw_networkx_nodes(G, pos_dict, ax=ax, nodelist=[path[-1]], node_size=50, node_color='blue')

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal', adjustable='datalim')


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main_pathfinder():
    args = parse_path_args() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    edge_index, node_features, positions, osmid_to_idx = load_graph_data(DATASET_DIR)
    pos_dict = {i: (coords[0], coords[1]) for i, coords in enumerate(positions)}


    model, scalers, scalers_dict = load_model_and_scalers(
        EdgeGCN, MODEL_PATH, SCALERS_PATH, node_features.shape[1], device
    )


    try:
        _, val_snaps = load_snapshots(DATASET_DIR)
        

        snapshot_index = args.snapshot_index
        if not (0 <= snapshot_index < len(val_snaps)):
            print(f"Error: --snapshot-index {snapshot_index} is out of range.")
            print(f"Please use an index between 0 and {len(val_snaps) - 1}.")
            return
            
        selected_snap = val_snaps[snapshot_index]
        print(f"Using validation snapshot at index: {snapshot_index}")


        expected_edges = edge_index.shape[1]
        edge_attr_snapshot, used_key = extract_edge_attr(selected_snap, expected_edges)
        print(f"Found edge attributes using key: {used_key}")
        
    except Exception as e:
        print(f"Error loading snapshot data to get features: {e}")
        return


    print("Running model to predict edge weights...")
    predicted_weights = get_predicted_weights(
        model, node_features, edge_index, edge_attr_snapshot, scalers, scalers_dict, device
    )


    G = build_nx_graph(edge_index, predicted_weights)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=120)
    fig.canvas.manager.set_window_title('A* Pathfinding with Predicted Congestion')
    

    plot_predicted_weights(
        axes[0], pos_dict, edge_index, predicted_weights,
        title=f"Map 1: Predicted Congestion (Snapshot {snapshot_index})" 
    )


    path_astar = None 
    path_title = "Map 2: A* Route Visualization"
    start_osmid = args.start
    end_osmid = args.end

    
    try:
        start_node = osmid_to_idx[start_osmid]
        end_node = osmid_to_idx[end_osmid]
        print(f"Finding A* path from {start_osmid} (idx {start_node}) to {end_osmid} (idx {end_node})...")
        
        heuristic_func = lambda u, v: haversine_heuristic(pos_dict, u, v)

        path_astar = nx.astar_path(
            G, 
            source=start_node, 
            target=end_node,
            heuristic=heuristic_func,
            weight='weight'
        )
        
        cost_astar = nx.astar_path_length(
            G, 
            source=start_node, 
            target=end_node,
            heuristic=heuristic_func,
            weight='weight'
        )
        print(f"A* Path found: {len(path_astar)} nodes, Predicted Cost (Total Congestion): {cost_astar:.2f}")
        path_title = f"Map 2: Optimal A* Path ({start_osmid} to {end_osmid})"

    except nx.NetworkXNoPath:
        print(f"No path found between {start_osmid} and {end_osmid}.")
        path_title = f"Map 2: No Path Found ({start_osmid} to {end_osmid})"
    except KeyError as e:
        print(f"Error: OSM ID {e} not found in graph_structure.json.")
        print("Please check your start and end OSM ID values.")
        path_title = f"Map 2: Invalid OSM ID ({e})"

    plot_path(
        axes[1], pos_dict, G, path_astar, 
        title=path_title
    )
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main_pathfinder()