import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool

from InvariantMPNN import InvariantMPNN
from CartesianGNNLayer import CartesianGNNLayer

class CustomGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layer_type='invariant'):
        super(CustomGNN, self).__init__()
        if layer_type == 'invariant':
            self.conv1 = InvariantMPNN(in_channels, hidden_channels)
            self.conv2 = InvariantMPNN(hidden_channels, hidden_channels)
        elif layer_type == 'cartesian':
            self.conv1 = CartesianGNNLayer(in_channels, hidden_channels)
            self.conv2 = CartesianGNNLayer(hidden_channels, hidden_channels)
        elif layer_type == 'spherical':
            self.conv1 = SphericalGNNLayer(in_channels, hidden_channels, out_degree = 2)
            self.conv2 = SphericalGNNLayer(hidden_channels, hidden_channels, in_degree = 2, r_degree = 2, out_degree = 2)
        self.layer_type = layer_type
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, pos, edge_index, batch):
        # Message Passing Layer
        if self.layer_type == 'invariant':
            x = self.conv1(x, pos, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, pos, edge_index)
            x = torch.relu(x)
        elif self.layer_type == 'cartesian':
            x, pos = self.conv1(x, pos, edge_index)
            x = torch.relu(x)
            x, pos = self.conv2(x, pos, edge_index)
            x = torch.relu(x)
        elif self.layer_type == 'spherical':
            x = self.conv1(x, pos, edge_index)
            x_reshaped = x.reshape(x.shape[0], (self.conv1.in_degree+1)**2, -1)
            x0 = x_reshaped[:, :1, :]
            x1 = x_reshaped[:, 1:4, :]
            x2 = x_reshaped[:, 4:9, :]
            x0 = torch.relu(x0)
            x1_norm = torch.norm(x1, dim = 1, keepdim = True)
            x1 = torch.sigmoid(x1_norm) * x1
            x2_norm = torch.norm(x2, dim = 1, keepdim = True)
            x2 = torch.sigmoid(x2_norm) * x2
            x = torch.cat([x0, x1, x2], dim = 1)
            x = x.reshape(x.shape[0], -1)
            x = self.conv2(x, pos, edge_index)
            x_reshaped = x.reshape(x.shape[0], (self.conv2.in_degree+1)**2, -1)
            x0 = x_reshaped[:, :1, :]
            x0 = torch.relu(x0)
            x = x0[:,0]
            
        
        

        # Global Pooling
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x))
        return self.lin2(x)