import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


from GCNLayer import GCNLayer
from GATLayer import GATLayer
from GCNNet import GCNNet
from GATNet import GATNet



# Load the Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0]  # Cora dataset has only one graph

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of features: {data.num_node_features}')
print(f'Number of classes: {dataset.num_classes}')


# Training, validation, and test masks
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

print(f'Number of training nodes: {train_mask.sum().item()}')
print(f'Number of validation nodes: {val_mask.sum().item()}')
print(f'Number of test nodes: {test_mask.sum().item()}')

def test(input, edge_index, mask, ground_truth):
    model.eval()
    # prediction = model(data.x, data.edge_index)  # Forward pass
    prediction = model(input, edge_index)
    pred = prediction.argmax(dim=1)  # Use the class with the highest probability
    # correct = (pred[test_mask] == data.y[test_mask]).sum()  # Check how many predictions match the true labels
    correct = (pred[mask] == ground_truth[mask]).sum()  # Check how many predictions match the true labels
    acc = int(correct) / int(mask.sum())  # Calculate accuracy
    return acc


model = GCNNet(in_channels=dataset.num_node_features, hidden_channels=24, out_channels=dataset.num_classes)

# # Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

def train():
    model.train() # Set the model to training mode
    optimizer.zero_grad()  # Clear gradients
    prediction = model(data.x, data.edge_index)  # Forward pass
    loss = criterion(prediction[train_mask], data.y[train_mask])  # Compute the loss
    loss.backward()  # Backward pass, i.e. gradient computation
    optimizer.step()  # Update model parameters
    return loss.item()

# Store the training loss values
loss_values = []

# Training loop
# for epoch in range(1000):
#     loss = train()
#     loss_values.append(loss)
#     if epoch % 10 == 0:
#         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# best_test_acc = 0
# early_stopping_counter = 0
# patience = 100

# acc_train_values = []
# acc_test_values = []

# # Training loop with early stopping
# for epoch in range(1000):
#     loss = train()
#     loss_values.append(loss)
#     current_test_acc = test(data.x, data.edge_index, test_mask, data.y)
#     acc_test = test(data.x, data.edge_index, test_mask, data.y)
#     acc_train = test(data.x, data.edge_index, train_mask, data.y)
    
#     acc_test_values.append(acc_test)
#     acc_train_values.append(acc_train)
    
#     if current_test_acc > best_test_acc:
#         best_test_acc = current_test_acc
#         # torch.save(model.state_dict(), 'best_model.pth')
#         early_stopping_counter = 0
#     else:
#         early_stopping_counter += 1
#         if early_stopping_counter >= patience:
#             print("Early stopping triggered")
#             break

#     if epoch % 10 == 0:
#         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# test_acc = test(data.x, data.edge_index, test_mask, data.y)
# print(f'Test Accuracy: {test_acc:.4f}')

# train_acc = test(data.x, data.edge_index, train_mask, data.y)
# print(f'Train Accuracy: {train_acc:.4f}')

# torch.onnx.export(model, (data.x, data.edge_index), 'gcn_model.onnx', opset_version=11)

# epochs = list(range(len(acc_train_values)))
# plt.plot(epochs, acc_train_values, label='Training Accuracy', color='blue', marker='o')
# plt.plot(epochs, acc_test_values, label='Testing Accuracy', color='orange', marker='o')

# # Adding labels and title
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Testing Accuracy Over Epochs')
# plt.legend()  # Adding the legend
# plt.grid(True)  # Optional: Adds grid for readability
# plt.show()


gat_model = GATNet(in_channels=dataset.num_node_features, hidden_channels=11, out_channels=dataset.num_classes)
optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

def gat_train():
    gat_model.train() # Set the model to training mode
    optimizer.zero_grad()  # Clear gradients
    prediction = gat_model(data.x, data.edge_index)  # Forward pass
    loss = criterion(prediction[train_mask], data.y[train_mask])  # Compute the loss
    loss.backward()  # Backward pass, i.e. gradient computation
    optimizer.step()  # Update model parameters
    return loss.item()

def gat_test(input, edge_index, mask, ground_truth):
    gat_model.eval()
    # prediction = model(data.x, data.edge_index)  # Forward pass
    prediction = gat_model(input, edge_index)
    pred = prediction.argmax(dim=1)  # Use the class with the highest probability
    # correct = (pred[test_mask] == data.y[test_mask]).sum()  # Check how many predictions match the true labels
    correct = (pred[mask] == ground_truth[mask]).sum()  # Check how many predictions match the true labels
    acc = int(correct) / int(mask.sum())  # Calculate accuracy
    return acc

# Store the training loss values
gat_loss_values = []

gat_acc_train_values = []
gat_acc_test_values = []

# Training loop
# for epoch in range(1000):
#     loss = gat_train()
#     gat_loss_values.append(loss)
#     if epoch % 10 == 0:
#         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

best_test_acc = 0
early_stopping_counter = 0
patience = 100
# Training loop with early stopping
for epoch in range(1000):
    loss = gat_train()
    gat_loss_values.append(loss)
    current_test_acc = gat_test(data.x, data.edge_index, test_mask, data.y)
    current_train_acc = gat_test(data.x, data.edge_index, train_mask, data.y)
    
    gat_acc_test_values.append(current_test_acc)
    gat_acc_train_values.append(current_train_acc)

    if current_test_acc > best_test_acc:
        best_test_acc = current_test_acc
        # torch.save(model.state_dict(), 'best_model.pth')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


def test(input, edge_index, mask, ground_truth):
    model.eval()
    # prediction = model(data.x, data.edge_index)  # Forward pass
    prediction = model(input, edge_index)
    pred = prediction.argmax(dim=1)  # Use the class with the highest probability
    # correct = (pred[test_mask] == data.y[test_mask]).sum()  # Check how many predictions match the true labels
    correct = (pred[mask] == ground_truth[mask]).sum()  # Check how many predictions match the true labels
    acc = int(correct) / int(mask.sum())  # Calculate accuracy
    return acc

test_acc = gat_test(data.x, data.edge_index, test_mask, data.y)
print(f'Test Accuracy: {test_acc:.4f}')

train_acc = gat_test(data.x, data.edge_index, train_mask, data.y)
print(f'Train Accuracy: {train_acc:.4f}')



def get_class_errors(model, data, mask):
    model.eval()
    with torch.no_grad():
        # Get model predictions
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Predicted labels

    # Only consider nodes within the given mask (e.g., test or train mask)
    true_labels = data.y[mask]
    predicted_labels = pred[mask]

    # Identify errors
    errors = (predicted_labels != true_labels)
    
    # Count total instances per class in the masked set
    class_counts = torch.bincount(true_labels, minlength=dataset.num_classes)
    
    # Count errors per class and normalize by class size
    class_errors = torch.bincount(true_labels[errors], minlength=dataset.num_classes).float()
    class_error_rates = class_errors / class_counts  # Normalized error rate per class
    
    # Replace any NaNs (due to division by zero) with 0
    class_error_rates[torch.isnan(class_error_rates)] = 0
    
    return class_error_rates

# Get normalized errors per class for test and train masks
test_errors = get_class_errors(gat_model, data, test_mask)

# Prepare data for plotting
labels = [f'Class {i}' for i in range(dataset.num_classes)]

# Plotting the normalized errors per class for the test set
plt.figure(figsize=(10, 6))
plt.bar(labels, test_errors.tolist(), color='orange', label='Test Error Rate')
plt.xlabel('Class')
plt.ylabel('Error Rate (Normalized)')
plt.title('Normalized Error Rate Across Classes (Test Set)')
plt.legend()
plt.show()




epochs = list(range(len(gat_acc_train_values)))
plt.plot(epochs, gat_acc_train_values, label='Training Accuracy', color='blue', marker='o')
plt.plot(epochs, gat_acc_test_values, label='Testing Accuracy', color='orange', marker='o')

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy Over Epochs')
plt.legend()  # Adding the legend
plt.grid(True)  # Optional: Adds grid for readability
plt.show()

import sklearn
from sklearn.manifold import TSNE
import numpy as np

# Get model predictions (logits)
gat_model.eval()
with torch.no_grad():
    embeddings = gat_model(data.x, data.edge_index)  # Get logits (pre-softmax)

# Reduce dimensions using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())

# Get predicted and true labels
predicted_labels = embeddings.argmax(dim=1).cpu().numpy()
true_labels = data.y.cpu().numpy()

# Plot the decision space
plt.figure(figsize=(12, 8))

# Scatter plot for predicted labels
scatter = plt.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1], 
    c=predicted_labels, cmap='tab10', s=20, alpha=0.8, edgecolor='k'
)
plt.colorbar(scatter, label='Predicted Class')

# Add title and labels
plt.title('Decision Space Visualization (t-SNE)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

