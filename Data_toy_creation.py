import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
import torch.nn.functional as F

def add_third_dimension_and_repeat(point_cloud, n_repeats):
    """Add a third dimension to the point cloud and repeat it n_repeats times."""
    z = 3*np.arange(n_repeats)
    z = np.repeat(z, point_cloud.shape[0])
    z = z[:,None]
    point_cloud = np.tile(point_cloud, (n_repeats, 1))
    point_cloud = np.concatenate((point_cloud, z), axis=1)
    return point_cloud


def generate_rotations_3d(point_cloud, alpha, beta, gamma):
    """Generate rotated point clouds."""
    # Rotation matrices
    R_alpha = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                        [np.sin(alpha), np.cos(alpha), 0],
                        [0, 0, 1]])
    R_beta = np.array([[1, 0, 0],
                       [0, np.cos(beta), -np.sin(beta)],
                       [0, np.sin(beta), np.cos(beta)]])
    R_gamma = np.array([[np.cos(gamma), 0, np.sin(gamma)],
                        [0, 1, 0],
                        [-np.sin(gamma), 0, np.cos(gamma)]])
    # Rotate point cloud
    return np.dot(np.dot(np.dot(point_cloud, R_alpha), R_beta), R_gamma)


A = np.array([[0,0],[1/4, 1],[1/2,2],[3/4, 1],[1,0]])
B = np.array([[0,0],[1,1/2],[0, 1],[1, 3/2],[0, 2]])
C = np.array([[1,0],[0,0],[0,1],[0,2],[1,2]])
D = np.array([[0,0],[2/3,0],[1,1],[2/3,2],[0,2]])
E = np.array([[1,0],[0,0],[1,1],[0,2],[1,2],])
F = np.array([[0,0],[0,1],[0,2],[1,2],[1,1]])

a_indices = np.array([0,1,2,3,4,3,1])
b_indices = np.array([0,1,2,3,4,2,0])
c_indices = np.array([0,1,2,3,4])
d_indices = np.array([0,1,2,3,4,0])
e_indices = np.array([0,1,2,3,4,])
f_indices = np.array([0,1,2,3,2,1,4])

colors = ['blue', 'black', 'green', 'purple', 'orange']#, 'brown']

dataset_size = 6000
dataset_np = []
features = np.eye(5)

for i in range(dataset_size):
    class_label = i//1000
    repeats_numbers = np.random.randint(1, 5)
    random_shift = np.random.rand(3)
    features_rep = np.tile(features, (repeats_numbers, 1))
    if class_label == 0:
        repeated_A = add_third_dimension_and_repeat(A, repeats_numbers)
        rotated_A = generate_rotations_3d(repeated_A, np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi))
        rotated_A = rotated_A + random_shift[None, :]
        # all_feat = np.concatenate((rotated_A, features_rep), axis=1)
        dataset_np.append((rotated_A, features_rep, class_label))
        
    elif class_label == 1:
        repeated_B = add_third_dimension_and_repeat(B, repeats_numbers)
        rotated_B = generate_rotations_3d(repeated_B, np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi))
        rotated_B = rotated_B + random_shift[None, :]
        # all_feat = np.concatenate((rotated_B, features_rep), axis=1)
        dataset_np.append((rotated_B, features_rep, class_label))
    elif class_label == 2:
        repeated_C = add_third_dimension_and_repeat(C, repeats_numbers)
        rotated_C = generate_rotations_3d(repeated_C, np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi))
        rotated_C = rotated_C + random_shift[None, :]
        # all_feat = np.concatenate((rotated_C, features_rep), axis=1)
        dataset_np.append((rotated_C, features_rep, class_label))
    elif class_label == 3:
        repeated_D = add_third_dimension_and_repeat(D, repeats_numbers)
        rotated_D = generate_rotations_3d(repeated_D, np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi))
        rotated_D = rotated_D + random_shift[None, :]
        # all_feat = np.concatenate((rotated_D, features_rep), axis=1)
        dataset_np.append((rotated_D, features_rep, class_label))
    elif class_label == 4:
        repeated_E = add_third_dimension_and_repeat(E, repeats_numbers)
        rotated_E = generate_rotations_3d(repeated_E, np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi))
        rotated_E = rotated_E + random_shift[None, :]
        # all_feat = np.concatenate((rotated_E, features_rep), axis=1)
        dataset_np.append((rotated_E, features_rep, class_label))
    elif class_label == 5:
        repeated_F = add_third_dimension_and_repeat(F, repeats_numbers)
        rotated_F = generate_rotations_3d(repeated_F, np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi))
        rotated_F = rotated_F + random_shift[None, :]
        # all_feat = np.concatenate((rotated_F, features_rep), axis=1)
        dataset_np.append((rotated_F, features_rep, class_label))

import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None):
        self.data_list = data_list
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
        
        if data_list is not None:
            self.data, self.slices = self.collate(self.data_list)
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])
            # data_list from data
            self.data_list = [Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y) for data in self.data]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        if self.data_list is not None:
            self.data, self.slices = self.collate(self.data_list)
            torch.save((self.data, self.slices), self.processed_paths[0])

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def save(self):
        torch.save((self.data, self.slices), self.processed_paths[0])

data_list = []
for coords, features, label in dataset_np:
    x = torch.tensor(features, dtype=torch.float)#.view(-1, 1)  # Node features
    r = torch.tensor(coords, dtype=torch.float)  # Node coordinates
    y = torch.tensor([label], dtype=torch.long)  # Label
    # edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Dummy edge_index for example
    
    data = Data(x=x, r = r, y=y)
    data_list.append(data)

dataset = CustomGraphDataset(root = './data_set_toy', data_list = data_list)

# dataset.save()


from torch_geometric.loader import DataLoader

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# for batch in dataset:
#     for i in batch:
#         print(i)

# print(dataset.data.y)


target = 0
batch_size = 16

dataset.data.y = dataset.data.y.float()  # Converts to float
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std

train_dataset = dataset[:1000]
val_dataset = dataset[1000:1100]
test_dataset = dataset[1100:1200]

# DataLoader settings
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

from CustomGNN import CustomGNN
from InvariantMPNN import InvariantMPNN
invariant_model = CustomGNN(in_channels=dataset.num_node_features, hidden_channels=5, out_channels=dataset.num_classes, layer_type='invariant')
cartesian_model = CustomGNN(in_channels=dataset.num_node_features, hidden_channels=5, out_channels=dataset.num_classes, layer_type='cartesian')


import torch.nn.functional as F
from torch.optim import Adam

edge = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

# for data in dataloader:
#     # print(data)
#     # print(data.x)
#     # print(data.y)
#     # print(data.r)
#     # print(data.ptr)
#     # print(data.batch)
#     print(cartesian_model(data.x,data.r,edge,data.batch))
    

def train(loader, model, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        # data = data.to('cuda')
        optimizer.zero_grad()
        out = model(data.x,data.r, edge,data.batch)
        class_predicted = []
        out = torch.softmax(out,dim=1)
        # for elem in out:
            # print(torch.argmax(out).item())
            # class_predicted.append(torch.argmax(out).item())
        # print(class_predicted) 
        gt = torch.nn.functional.one_hot(data.y, 6)
        gt= torch.tensor(gt, dtype = torch.float)
        print(out.shape)
        print(gt.shape)
        # class_predicted = torch.tensor(class_predicted,dtype = torch.float)
        loss = F.mse_loss(out, gt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(loader, model, optimizer):
    model.eval()
    error = 0
    for data in loader:
        # data = data.to('cuda')
        with torch.no_grad():
            out = model(data.x, data.pos, data.edge_index, data.batch)
            error += (out - data.y[:, target, None]).abs().sum().item()
    return error / len(loader.dataset)


optimizer = Adam(cartesian_model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cartesian_model.to(device)

target = 0  # Select the property index to predict

for epoch in range(1, 1001):
    loss = train(train_loader, cartesian_model, optimizer)
    test_error = test(test_loader, cartesian_model, optimizer)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test MAE: {test_error:.4f}')