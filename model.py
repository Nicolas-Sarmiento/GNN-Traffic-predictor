"""PyTorch module definition for the EdgeGCN model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GCNConv
except ImportError as e:
    raise SystemExit("PyTorch Geometric no instalado. Instala antes de ejecutar este script.")

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