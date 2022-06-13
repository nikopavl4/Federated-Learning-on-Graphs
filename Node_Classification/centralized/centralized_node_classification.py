import torch
import matplotlib.pyplot as plt
import argparse
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from helpers import trainer, tester, simple_train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from model import GCN, SAGE
import numpy as np
torch.manual_seed(12345)
torch.cuda.manual_seed_all(12345)
np.random.seed(12345)
import time
# get the start time
st = time.time()

parser = argparse.ArgumentParser(description='Insert Arguments')

parser.add_argument('--model', type=str, default="gcn", help='GNN used in training')
parser.add_argument("--dataset", type=str, default="cora", help="dataset used for training")
parser.add_argument("--split", type=float, default=0.6, help="test/train dataset split percentage")
parser.add_argument("--hidden_channels", type=int, default=16, help="size of GNN hidden layer")
parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate for training")
parser.add_argument("--epochs", type=int, default=50, help="epochs for training")

args = parser.parse_args()

#Import and Examine Dataset
if args.dataset.lower() == 'pubmed':
    dataset = Planetoid(root='data/Planetoid', name='PubMed')
elif args.dataset.lower() == 'cora':
    dataset = Planetoid(root='data/Planetoid', name='Cora')
elif args.dataset.lower() == 'citeseer':
    dataset = Planetoid(root='data/Planetoid', name='CiteSeer')
else:
    print("No such dataset!")
    exit()

print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.
data = simple_train_test_split(data, args.split)

print()
print(data)
print('=============================================================')
# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')

#GNN Initialization
if args.model.lower() == "gcn":
    model = GCN(hidden_channels=args.hidden_channels, features_in=dataset.num_node_features, features_out=dataset.num_classes)
elif args.model.lower() == "sage":
    model = SAGE(hidden_channels=args.hidden_channels, features_in=dataset.num_node_features, features_out=dataset.num_classes)
else:
    print("Model does not exist! Please select GCN or GraphSAGE")
    exit()

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


val_losses = []
train_losses = []
print("+++++++++++++ Centralized Node Classification +++++++++++++")
for epoch in range(1, args.epochs + 1):
    loss, train_acc = trainer(model, optimizer, criterion, data)
    test_acc = tester(model, data) 
    train_losses.append(train_acc)
    val_losses.append(test_acc)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

#Model Evaluation
model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)  # Use the class with highest probability.
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(data.y[data.test_mask], pred[data.test_mask])
f1 = f1_score(y_true = data.y[data.test_mask],y_pred = pred[data.test_mask], average='macro', zero_division=1)
precision = precision_score(y_true =data.y[data.test_mask],y_pred = pred[data.test_mask], average='macro', zero_division=1)
recall = recall_score(y_true = data.y[data.test_mask],y_pred = pred[data.test_mask], average='macro', zero_division=1)
print(conf_matrix)
print(f"F1 Score:{f1:.4f}")
print(f"Precision:{precision:.4f}")
print(f"Recall:{recall:.4f}")

# get the end time
et = time.time()
# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


#Plot Diagrams
ConfusionMatrixDisplay.from_predictions(y_true = data.y[data.test_mask],y_pred = pred[data.test_mask], cmap=plt.cm.Blues) 
plt.figure(figsize=(10,5))
plt.title("Training and Testing Accuracy")
plt.plot(val_losses,label="Test")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

