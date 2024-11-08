import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import add_self_loops, degree
from torch.nn.functional import relu,leaky_relu
from GCNLayer import GCNLayer 
from GATLayer import GATLayer 
import torch.nn.functional as F

class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATNet, self).__init__()
        self.gat1 = GATLayer(in_channels, hidden_channels)
        self.gat2 = GATLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First GCN layer + ReLU activation
        x = self.gat1(x, edge_index)
        x = leaky_relu(x)
        # Second GCN layer + LogSoftmax activation for multi-class classification
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

