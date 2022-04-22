import torch
from torch_geometric.datasets import TUDataset, MoleculeNet, GNNBenchmarkDataset
import argparse
from models.gcn import GCN
from models.sage import SAGE
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='Insert Arguments')

parser.add_argument('--model', type=str, default="gcn", help='GNN used in training')
parser.add_argument("--dataset", type=str, default="enzymes", help="dataset used for training")
parser.add_argument("--split", type=float, default=0.8, help="test/train dataset split percentage")
parser.add_argument("--batch_size", type=int, default=16, help="input batch size for training (default: 16)")
parser.add_argument("--hidden_channels", type=int, default=16, help="size of GNN hidden layer")
parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate for training")
parser.add_argument("--epochs", type=int, default=70, help="epochs for training")

args = parser.parse_args()

#Load Dataset
if args.dataset.lower() == 'enzymes':
    dataset = TUDataset(root='data/TUDataset', name='ENZYMES')
elif args.dataset.lower() == 'proteins':
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
elif args.dataset.lower() == 'sider':
    dataset = MoleculeNet(root='data/Moleculenet', name='SIDER')
else:
    print("No such dataset!")
    exit()


print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:int(len(dataset)*args.split)]
test_dataset = dataset[int(len(dataset)*args.split):]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

class0 = 0
datr = dataset[5]
my = datr.y
print(my)
for data in dataset:
     if torch.equal(data.y, my):
         class0 = class0 + 1
print(class0)


from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if args.model.lower() == "gcn":
    model = GCN(hidden_channels=args.hidden_channels, features_in=dataset.num_node_features, features_out=dataset.num_classes)
elif args.model.lower() == "sage":
    model = SAGE(hidden_channels=args.hidden_channels, features_in=dataset.num_node_features, features_out=dataset.num_classes)

print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
         f1_score1 = f1_score(y_true = data.y,y_pred = pred, average='micro')
     return correct / len(loader.dataset), f1_score1  # Derive ratio of correct predictions.

import matplotlib.pyplot as plt

val_losses = []
train_losses = []
for epoch in range(1, args.epochs + 1):
    train()
    train_acc, f1_train = test(train_loader)
    test_acc, f1_test = test(test_loader)
    train_losses.append(train_acc)
    val_losses.append(test_acc)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, F1 Score:{f1_test:.4f} ')

#Printing final results
print(f"Accuracy:{test_acc:.4f}")
print(f"F1 Score:{f1_test:.4f}")




plt.figure(figsize=(10,5))
plt.title("Training and Testing Accuracy")
plt.plot(val_losses,label="Test")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.show()