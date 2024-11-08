import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

class CartesianGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CartesianGNNLayer, self).__init__(aggr='add')  # Aggregation function: 'add'
        # self.edge_mlp = nn.Sequential(
        #     nn.Linear(2 * in_channels + 1, out_channels),
        #     nn.ReLU(),
        #     nn.Linear(out_channels, out_channels)
        # )
        self.coord_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, 1),
            nn.Tanh()  # To limit coordinate updates
            
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )
        
    def forward(self, x, pos, edge_index):
        num_nodes = x.size(0)
        # Start message passing
        out = self.propagate(edge_index, x=x, pos=pos, size=(num_nodes, num_nodes))
        x_out, coord_updates = out
        # Update positions
        pos = pos + coord_updates
        return x_out, pos
    
    def message(self, x_i, x_j, pos_i, pos_j):
        # Relative positional differences and distances
        diff = pos_i - pos_j  # [num_edges, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [num_edges, 1]
        
        # Edge features: concatenate node features and distance
        edge_input = torch.cat([x_i, x_j, dist], dim=-1)  # [num_edges, 2 * in_channels + 1]
        # e_ij = self.edge_mlp(edge_input)  # [num_edges, out_channels]
        
        # Compute coordinate updates
        coord_update = self.coord_mlp(edge_input) * diff  # [num_edges, 3]
        
        # return e_ij, coord_update
        return coord_update
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # e_ij, coord_update = inputs
        coord_update = inputs
        num_nodes = dim_size  # Total number of nodes
        # Aggregate edge features
        # aggr_e = scatter(e_ij, index, dim=0, dim_size=num_nodes, reduce='add')
        # Aggregate coordinate updates
        aggr_coord = scatter(coord_update, index, dim=0, dim_size=num_nodes, reduce='mean')
        # out_tensor = zeros((num_nodes, coord_update.shape[1]))
        # for i, coord_update_i in zip(index, coord_update):
        #.   out_tensor[i] = out_tensor[i] + coord_update_i
        # return aggr_e, aggr_coord
        return aggr_coord
    
    def update(self, aggr_out, x, pos):
        # aggr_e, aggr_coord = aggr_out
        aggr_coord = aggr_out
        # Update node features
        # node_input = torch.cat([x, aggr_e], dim=-1)  # Concatenate along feature dimension
        node_input = x
        x_out = self.node_mlp(node_input)
        return x_out, aggr_coord + pos

