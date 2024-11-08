from torch_geometric.utils import softmax
# import leaky relu
from torch.nn.functional import leaky_relu
import torch_geometric.nn as pyg_nn
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree
from torch.nn.functional import relu


class GATLayer(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GATLayer, self).__init__(aggr='add')
        self.lin_key = nn.Linear(in_channels, out_channels)
        self.lin_query = nn.Linear(in_channels, out_channels)
        self.lin_value = nn.Linear(in_channels, out_channels)


    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix to consider self-connections
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        # Perform message passing
        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i, edge_index):
        # x_j: Input neighbor features
        # x_i Input features of the central node

        # Perform linear transformation on node features

        value = self.lin_value(x_j)
        key = self.lin_key(x_j)
        query = self.lin_query(x_i)

        alpha_raw = leaky_relu((query*key).sum(dim=-1))
        alpha = softmax(alpha_raw, edge_index[0])
        return alpha[:,None]*value

    def update(self, aggr_out):
        # aggr_out: Aggregated messages
        # x: Original node features

        # Perform update operation on node features: simple addition
        return aggr_out

