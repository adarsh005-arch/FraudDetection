import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNNModel(nn.Module):
    def __init__(self, input_dim):
        super(GNNModel, self).__init__()

        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 32)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)

        return x