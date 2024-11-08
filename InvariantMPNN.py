import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
import torch.nn.functional as F

class InvariantMPNN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rbf=16):
        super(InvariantMPNN, self).__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels)
        self.dist_lin = Linear(num_rbf, out_channels)  # Linear transformation for RBF-transformed distances
        self.rbf_centers = torch.linspace(0, 5, num_rbf)  # Radial basis function centers
        self.rbf_gamma = torch.tensor(1.0)  # Gamma parameter for RBF

    def forward(self, x, pos, edge_index):
        # x: Node features
        # pos: Node coordinates
        # edge_index: Edge indices

        # Calculate distances between connected nodes
        row, col = edge_index
        edge_vectors = pos[row] - pos[col]
        distances = torch.norm(edge_vectors, p=2, dim=-1).unsqueeze(-1)
        
        # Compute RBF of distances
        rbf = torch.exp(-self.rbf_gamma * (distances - self.rbf_centers) ** 2)

        # Propagate messages
        return self.propagate(edge_index, x=x, rbf=rbf)

    def message(self, x_j, rbf):
        # x_j: Source node features
        # rbf: RBF-transformed distance features

        edge_features = self.dist_lin(rbf)  # Transform RBF features
        return self.lin(x_j) + edge_features  # Combine node and edge features

    def update(self, aggr_out):
        # aggr_out: Aggregated messages
        return F.relu(aggr_out)  # Apply ReLU non-linearity

