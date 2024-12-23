import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
import torch.nn.functional as F



# Replace MAE by accuracy which is best suited for classification task
def get_accuracy(predicted, true):
    correct = (predicted == true).sum().item()  # Count of correct predictions
    total = true.size(0)                        # Total number of predictions
    return  correct / total  

# Replace edge_index by the actual connection between letters (which leads to 100% accuracy after the first epoch)
num_nodes=5
def get_edge_index(indices):
    return get_fully_connected_edge_index() #  Uncomment to test with fully connected
    return np.concatenate(
            (
                np.array(
                    list(indices[1:]) # one connection
                    +list(indices[:-1]) # reverse connection
                    +list(range(num_nodes)) # self connection
                    )[None],
                np.array(
                    list(indices[:-1]) # one connection
                    +list(indices[1:]) # reverse connection
                    +list(range(num_nodes)) # self connection
                    )[None]
            )
        )

# Replace edge_index by a fully connected graph, I could only reach 83% acc, 
# probably because some letters are the same when rotated, 
# that could be verified with a confusion matrix ...
def get_fully_connected_edge_index():
    # Generate all combinations of source and target nodes
    row = torch.arange(num_nodes).repeat(num_nodes)
    col = torch.arange(num_nodes).repeat_interleave(num_nodes)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index




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
        edge_index = get_edge_index(a_indices)
        dataset_np.append((rotated_A, features_rep, class_label, edge_index))
        
    elif class_label == 1:
        repeated_B = add_third_dimension_and_repeat(B, repeats_numbers)
        rotated_B = generate_rotations_3d(repeated_B, np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi))
        rotated_B = rotated_B + random_shift[None, :]
        # all_feat = np.concatenate((rotated_B, features_rep), axis=1)
        edge_index = get_edge_index(b_indices)
        dataset_np.append((rotated_B, features_rep, class_label, edge_index))
    elif class_label == 2:
        repeated_C = add_third_dimension_and_repeat(C, repeats_numbers)
        rotated_C = generate_rotations_3d(repeated_C, np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi))
        rotated_C = rotated_C + random_shift[None, :]
        # all_feat = np.concatenate((rotated_C, features_rep), axis=1)
        edge_index = get_edge_index(c_indices)
        dataset_np.append((rotated_C, features_rep, class_label, edge_index))
    elif class_label == 3:
        repeated_D = add_third_dimension_and_repeat(D, repeats_numbers)
        rotated_D = generate_rotations_3d(repeated_D, np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi))
        rotated_D = rotated_D + random_shift[None, :]
        # all_feat = np.concatenate((rotated_D, features_rep), axis=1)
        edge_index = get_edge_index(d_indices)
        dataset_np.append((rotated_D, features_rep, class_label, edge_index))
    elif class_label == 4:
        repeated_E = add_third_dimension_and_repeat(E, repeats_numbers)
        rotated_E = generate_rotations_3d(repeated_E, np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi))
        rotated_E = rotated_E + random_shift[None, :]
        # all_feat = np.concatenate((rotated_E, features_rep), axis=1)
        edge_index = get_edge_index(e_indices)
        dataset_np.append((rotated_E, features_rep, class_label, edge_index))
    elif class_label == 5:
        repeated_F = add_third_dimension_and_repeat(F, repeats_numbers)
        rotated_F = generate_rotations_3d(repeated_F, np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi))
        rotated_F = rotated_F + random_shift[None, :]
        # all_feat = np.concatenate((rotated_F, features_rep), axis=1)
        edge_index = get_edge_index(f_indices)
        dataset_np.append((rotated_F, features_rep, class_label, edge_index))

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

import random
data_list = []
for coords, features, label, edge_index in dataset_np:

    y = torch.tensor([label], dtype=torch.long)  # Label
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    if(y[0].tolist() == 2): # Differenciation of C and E
        features[0,1] = 1

    # Including a random noise in the connections
    # number_of_modified_edge = 200
    # for i in range(number_of_modified_edge):
    #     dim = features.shape
    #     indices_aleatoires = [int(np.random.choice(dim[0],size=1, replace=True)),int(np.random.choice(dim[1],size=1, replace=True))] 
    #     if(features[indices_aleatoires[0],indices_aleatoires[1]] == 1):
    #         features[indices_aleatoires[0],indices_aleatoires[1]] = 0
    #     features[indices_aleatoires[0],indices_aleatoires[1]] = 1

    x = torch.tensor(features, dtype=torch.float)

    # Including a random noise in the coordinates

    number_of_modified_coord = 20
    for i in range(number_of_modified_coord):    
        dim = coords.shape
        indices_aleatoires = [int(np.random.choice(dim[0],size=1, replace=True)),int(np.random.choice(dim[1],size=1, replace=True))] 
        coords[indices_aleatoires[0],indices_aleatoires[1]] += np.random.normal(0,10)

    r = torch.tensor(coords, dtype=torch.float)


    data = Data(x=x, r = r, y=y,edge_index = edge_index)
    data_list.append(data)

dataset = CustomGraphDataset(root = './data_set_toy', data_list = data_list)



from torch_geometric.loader import DataLoader

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

target = 0
batch_size = 16

dataset.data.y = dataset.data.y.float()  # Converts to float
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std


import random
# random.seed(42)


indices = list(range(6000))
random.shuffle(indices)
train_indices = indices[:4000]
val_indices = indices[4000:5000]
test_indices = indices[5000:6000]
# Use these indices to get random subsets from the dataset
train_dataset = [dataset[i] for i in train_indices]
val_dataset = [dataset[i] for i in val_indices]
test_dataset = [dataset[i] for i in test_indices]


# DataLoader settings
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

from CustomGNN import CustomGNN
from InvariantMPNN import InvariantMPNN
import torch.nn.functional as F
from torch.optim import Adam


def train(loader, model, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0
    num_samples = 0

    for data in loader:
        data = data.to(device)  # Move data to the same device as the model
        optimizer.zero_grad()

        # Forward pass
        out = model(data.x, data.r, data.edge_index, data.batch)
        
        # Calculate loss (use CrossEntropy for classification)
        loss = F.cross_entropy(out, data.y.view(-1))  # Ensure labels are 1D
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item() * data.num_graphs
        num_samples += data.num_graphs

    return total_loss / num_samples  # Normalize by the total number of graphs


def test(loader, model, optimizer):
    model.eval()
    # error = 0
    acc=0
    for data in loader:
        # data = data.to('cuda')
        with torch.no_grad():
            out = model(data.x, data.r, data.edge_index, data.batch)
            # gt = torch.nn.functional.one_hot(data.y, 6)
            # gt= torch.tensor(gt, dtype = torch.float)
            # error += (out - gt).abs().sum().item()
            acc += get_accuracy(out.argmax(dim=-1), data.y)
    return acc / len(loader)

# model = CustomGNN(in_channels=11, hidden_channels=64, out_channels=1, layer_type='spherical')
# model = CustomGNN(in_channels=dataset.num_node_features, hidden_channels=5, out_channels=dataset.num_classes, layer_type='cartesian')
model = CustomGNN(in_channels=dataset.num_node_features, hidden_channels=5, out_channels=dataset.num_classes, layer_type='invariant')

optimizer = Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_accuracies = []
test_accuracies = []

for epoch in range(1, 25):
    loss = train(train_loader, model, optimizer,device)
    test_error = test(test_loader, model, optimizer)
    train_acc = test(train_loader, model, optimizer)
    test_acc = test(test_loader, model, optimizer)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test ACC: {test_error:.4f}')


# Plotting accuracy over epochs
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy", marker="o")
# plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy", marker="s")
# plt.title("Accuracy Over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid()
# plt.show()


predicted_labels = []
true_labels =[]
acc=0
for data in val_loader:
    # print(data.x)
    out = model(data.x, data.r, data.edge_index, data.batch)
    # gt = torch.nn.functional.one_hot(data.y, 6)
    # gt= torch.tensor(gt, dtype = torch.float)
    # # print(out,gt)
    acc += get_accuracy(out.argmax(dim=-1), data.y)

    predicted_labels += list(out.argmax(dim=-1))
    true_labels += list(data.y)
acc /= len(val_loader)
print("Final accuracy %.2f %%"%(acc*100))

def confusion_matrix(predicted, true, num_classes):
    # Initialize the confusion matrix with zeros
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    # Populate the confusion matrix
    for t, p in zip(true, predicted):
        conf_matrix[t, p] += 1

    return conf_matrix


print("Confusion Matrix:")
print(confusion_matrix(predicted_labels, true_labels, 6))

import seaborn as sns

# Compute the confusion matrix
conf_matrix = confusion_matrix(predicted_labels, true_labels, num_classes=6)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=range(6), yticklabels=range(6))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

