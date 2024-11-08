import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import add_self_loops, degree
from torch.nn.functional import relu


class GCNLayer(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)


    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix to consider self-connections
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        # Perform message passing
        return self.propagate(edge_index, x=x)

    def message(self, x_j, edge_index):
        # x_j: Input node features

        # Perform linear transformation on node features

        degree_u = degree(edge_index[0])
        degree_v = degree(edge_index[1])
        out = self.lin(x_j)
        # print(out.shape, degree_u.shape)

        norm_factor = (1/torch.sqrt(degree_u[edge_index[0]]*degree_v[edge_index[1]]))[:, None]
        return norm_factor*out


    # def aggregate(self, messages, index):
    #     # Aggregates messages for each node

    #     # Perform sum aggregation
    #     return torch.scatter_add(messages, index, dim=0)

    def update(self, aggr_out, x):
        # aggr_out: Aggregated messages
        # x: Original node features

        # Perform update operation on node features: simple addition
        return aggr_out

# # Example usage
# gcn_layer = GCNLayer(in_channels=3, out_channels=2)
# print(gcn_layer)