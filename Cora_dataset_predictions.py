import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GraphConv
import matplotlib.pyplot as plt

# Hyperparameters
model_type = 'CWL'  # Options: 'GCN', 'GAT', 'CWL', 'GIN', 'GraphSAGE'
dropout_rate = 0.4
weight_decay = 5e-4
hidden_channels = 32
num_layers = 1
num_heads = 16  # Only relevant for GAT
learning_rate = 0.01
num_epochs = 300

# Step 1: Load Dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Split data into train, validation, and test sets
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

# Step 2: Define Network
class GNNModel(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, out_channels, 
                 num_layers, dropout, num_heads=1):
        super(GNNModel, self).__init__()
        self.model_type = model_type
        self.convs = torch.nn.ModuleList()

        # Define the first layer
        if model_type == 'GCN':
            self.convs.append(GCNConv(in_channels, hidden_channels))
        elif model_type == 'GAT':
            self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=True))
        elif model_type == 'GraphSAGE':
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        elif model_type == 'CWL':
            self.convs.append(GraphConv(in_channels, hidden_channels))  # Placeholder for CWL
        elif model_type == 'GIN':
            self.convs.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )))

        # Define hidden layers
        for _ in range(num_layers - 2):
            if model_type == 'GCN':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            elif model_type == 'GAT':
                self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True))
            elif model_type == 'GraphSAGE':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            elif model_type == 'CWL':
                self.convs.append(GraphConv(hidden_channels, hidden_channels))
            elif model_type == 'GIN':
                self.convs.append(GINConv(torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )))

        # Define the last layer
        if model_type == 'GAT':
            self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False))
        else:
            self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

# Step 3: Initialize Model
model = GNNModel(
    model_type=model_type,
    in_channels=dataset.num_node_features,
    hidden_channels=hidden_channels,
    out_channels=dataset.num_classes,
    num_layers=num_layers,
    dropout=dropout_rate,
    num_heads=num_heads
)

# Step 4: Training Loop with Accuracy Tracking
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_losses = []
val_losses = []
test_losses = []
train_accuracies = []
val_accuracies = []
test_accuracies = []

best_val_accuracy = 0.0
best_model_state = None  # To store the best model

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    
    # Training loss and accuracy
    train_loss = F.nll_loss(out[train_mask], data.y[train_mask])
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())

    train_correct = (out[train_mask].argmax(dim=1) == data.y[train_mask]).sum()
    train_acc = int(train_correct) / int(train_mask.sum())
    train_accuracies.append(train_acc)

    # Validation and Test loss and accuracy
    model.eval()
    with torch.no_grad():
        # Validation
        val_loss = F.nll_loss(out[val_mask], data.y[val_mask])
        val_losses.append(val_loss.item())
        val_correct = (out[val_mask].argmax(dim=1) == data.y[val_mask]).sum()
        val_acc = int(val_correct) / int(val_mask.sum())
        val_accuracies.append(val_acc)
        
        # Test
        test_loss = F.nll_loss(out[test_mask], data.y[test_mask])
        test_losses.append(test_loss.item())
        test_correct = (out[test_mask].argmax(dim=1) == data.y[test_mask]).sum()
        test_acc = int(test_correct) / int(test_mask.sum())
        test_accuracies.append(test_acc)
    
    # Check if current validation accuracy is the best
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_state = model.state_dict()  # Save model state

# Step 5: Plot Training and Validation Loss Curves
plt.figure(figsize=(12, 6))

# Loss curves
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'{model_type} Training and Validation Loss Curves')
plt.legend()

# Accuracy curves
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'{model_type} Training, Validation, and Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Step 6: Final Test Accuracy
print(f'Final Training Accuracy ({model_type}): {train_accuracies[-1]:.4f}')
print(f'Final Validation Accuracy ({model_type}): {val_accuracies[-1]:.4f}')
print(f'Final Test Accuracy ({model_type}): {test_accuracies[-1]:.4f}')
print(f'Best Validation Accuracy ({model_type}): {best_val_accuracy:.4f}')

# Step 7: Save the Best Model
if best_model_state is not None:
    torch.save(best_model_state, f'{model_type}_best_model.pth')
    print(f'Best model saved as {model_type}_best_model.pth')

