import torch
from gcn_model import TrafficGCN



model = TrafficGCN(in_channels=25, hidden_channels=64, out_channels=3, dropout=0.3)
model = torch.load('gcn_traffic_model.pth')
model.eval()