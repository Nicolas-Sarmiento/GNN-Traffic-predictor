import torch 
from torch_geometric.data import Data
import numpy as np
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        # data.x = x # This line is incorrect as data is not defined in this scope and you are modifying the input tensor directly.
        return x


# Instantiate the model with the correct parameters used during training
# Based on the training cell (jq7efEhU7gFd), num_features was 25, num_hidden was 16, and output was 1 (for regression)
num_features = 25
num_hidden = 16
num_output_channels = 1 # The original model output a single value per node

model = TrafficGCN(in_channels=num_features, hidden_channels=num_hidden, out_channels=num_output_channels)

# Load the saved state dictionary into the model
model.load_state_dict(torch.load('gcn_traffic_model.pth'))

# Set the model to evaluation mode
model.eval()

#Cargamos el data en pt

path = "../data/processed_gnn_dataset/"

# Assuming these files are in the current working directory or accessible path
x = torch.tensor(np.load(path + "X_test.npy")[0], dtype=torch.float) # Load only the first time step
edge_index = torch.tensor(np.load( path  + "edge_index.npy"), dtype=torch.long)
y = torch.tensor(np.load( path + "Y_test.npy")[0], dtype=torch.float) # Load labels for the first time step

data = Data(x=x, edge_index=edge_index, y=y)

# Ensure data is on the correct device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
data = data.to(device)


with torch.no_grad():
   out = model(data.x, data.edge_index) # Pass the entire data object

# Convert predictions to numpy array on CPU
pred_values = out.cpu().numpy()


#visualizamos el grafo

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# Convertimo el grafo en networksx
G = to_networkx(data, to_undirected=True)

for i, val in enumerate(pred_values):
  # Ensure the node exists in the graph before adding attribute
  if i in G.nodes():
    G.nodes[i]['prediction'] = val[0] # Access the value if pred_values is (N, 1)

# Handle the case where the graph might be empty or node indices don't match
if G.number_of_nodes() > 0:
  # Get the node colors from the graph attributes
  node_color = [G.nodes[i].get('prediction', 0) for i in G.nodes()] 
  
  plt.figure(figsize=(8, 6))
  
  # 1. Get the current axes
  ax = plt.gca()
  
  # 2. Tell nx.draw to use these axes
  nx.draw(G, node_color=node_color, cmap='viridis', with_labels=False, node_size=50, ax=ax)
  
  # 3. Create the mappable object
  mappable = plt.cm.ScalarMappable(cmap='viridis')
  
  # 4. Set the data range for the mappable
  mappable.set_array(node_color)
  
  # 5. Tell the colorbar which axes to use
  plt.colorbar(mappable, label='Predicted value', ax=ax)
  
  plt.show()
else:
  print("Graph is empty, cannot visualize.")