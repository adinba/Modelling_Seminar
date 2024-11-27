import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import add_self_loops, degree
from torch.nn.functional import relu,leaky_relu
from GCNLayer import GCNLayer 
from GATLayer import GATLayer 
import torch.nn.functional as F

class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,dropout=0):
        super(GCNNet, self).__init__()
        self.conv1 = GCNLayer(in_channels, hidden_channels)
        self.conv2 = GCNLayer(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First GCN layer + ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout
        # Second GCN layer + LogSoftmax activation for multi-class classification
        x = self.conv2(x, edge_index)


        return F.log_softmax(x, dim=1)

