"""TrafficGCN: Graph Convolutional Network for road-traffic node prediction.

This model processes a road network graph where each node corresponds to a road
segment (or intersection) with input features such as length, num_lanes,
speed_limit, traffic dynamics, and scenario encodings. The network uses
Graph Convolutional layers (GCNConv) to aggregate information from neighbors
and produce continuous predictions per node (e.g., expected speed, flow,
projected congestion).

Inputs
------
- x: Node feature tensor of shape [num_nodes, in_channels]
- edge_index: Graph connectivity in COO format of shape [2, num_edges]

Outputs
-------
- Tensor of shape [num_nodes, out_channels] with continuous predictions.

Notes
-----
- At least two GCNConv layers are used, with ReLU activations and dropout.
- Default in_channels=25 to match the provided feature set; adjust as needed.
- CPU/GPU compatible without code changes (move tensors/model to the target device externally).
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TrafficGCN(torch.nn.Module):
    """Minimal, modular GCN for node-level regression tasks.

    Parameters
    ----------
    in_channels : int, default 25
        Number of input node features.
    hidden_channels : int, default 64
        Hidden dimension for the intermediate representation.
    out_channels : int, default 3
        Number of continuous outputs per node (e.g., speed, flow, congestion).
    dropout : float, default 0.3
        Dropout probability applied after the first GCN layer.
    """

    def __init__(
        self,
        in_channels: int = 25,
        hidden_channels: int = 64,
        out_channels: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Two-layer GCN: in -> hidden -> out
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward graph pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features [N, in_channels].
        edge_index : torch.Tensor
            COO edge indices [2, E].

        Returns
        -------
        torch.Tensor
            Node predictions [N, out_channels].
        """
        # Layer 1: graph conv + ReLU + dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 2: graph conv (linear output for regression)
        x = self.conv2(x, edge_index)
        return x
