import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import argparse
from models.gcn import GCN
from models.sage import SAGE
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import optuna

parser = argparse.ArgumentParser(description='Insert Arguments')

parser.add_argument('--model', type=str, default="gcn", help='GNN used in training')
parser.add_argument("--dataset", type=str, default="enzymes", help="dataset used for training")
parser.add_argument("--split", type=float, default=0.8, help="test/train dataset split percentage")
# parser.add_argument("--batch_size", type=int, default=16, help="input batch size for training (default: 16)")
# parser.add_argument("--hidden_channels", type=int, default=16, help="size of GNN hidden layer")
# parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate for training")
parser.add_argument("--epochs", type=int, default=150, help="epochs for training")

args = parser.parse_args()

#Load Dataset
if args.dataset.lower() == 'enzymes':
    dataset = TUDataset(root='data/TUDataset', name='ENZYMES')
elif args.dataset.lower() == 'proteins':
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
elif args.dataset.lower() == 'mutag':
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
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
print(data.batch)

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')

torch.manual_seed(12345)
dataset = dataset.shuffle()

#Train - Test Split based on user's input
train_dataset = dataset[:int(len(dataset)*args.split)]
test_dataset = dataset[int(len(dataset)*args.split):]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
        'batch_size': trial.suggest_discrete_uniform('batch_size', 16, 128, 16),
        'hidden_channels': trial.suggest_discrete_uniform('hidden_channels', 16, 64, 16),
    }

    #Create data loaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=int(params['batch_size']), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(params['batch_size']), shuffle=False)

    #GNN Initialization
    if args.model.lower() == "gcn":
        model = GCN(hidden_channels=int(params['hidden_channels']), features_in=dataset.num_node_features, features_out=dataset.num_classes)
    elif args.model.lower() == "sage":
        model = SAGE(hidden_channels=int(params['hidden_channels']), features_in=dataset.num_node_features, features_out=dataset.num_classes)
    else:
        print("Model does not exist! Please select GCN or GraphSAGE")
        exit()

    #print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    #Train, Test Functions
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
            f1 = f1_score(y_true = data.y,y_pred = pred, average='macro', zero_division=1)
            precision = precision_score(y_true = data.y,y_pred = pred, average='macro', zero_division=1)
            recall = recall_score(y_true = data.y,y_pred = pred, average='macro', zero_division=1)
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset), f1,precision, recall  # Derive ratio of correct predictions.

    #Training and printing results


    test_accuracies = {}
    test_f1 = {}
    test_precision = {}
    test_recall = {}
    train_accuracies = []
    for epoch in range(1, args.epochs + 1):
        train()
        train_acc, x, y, z = test(train_loader)
        test_acc, testf1, testpre, testrec = test(test_loader)
        train_accuracies.append(train_acc)
        test_accuracies[epoch] = test_acc
        test_f1[epoch] = testf1
        test_precision[epoch] = testpre
        test_recall[epoch] = testrec
        #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    print(f'Best Epoch = {max(test_accuracies, key=test_accuracies.get)}')
    print(f'F1 = {test_f1[max(test_accuracies, key=test_accuracies.get)]}')
    print(f'Precision = {test_precision[max(test_accuracies, key=test_accuracies.get)]}')
    print(f'Recall = {test_recall[max(test_accuracies, key=test_accuracies.get)]}')
    return max(list(test_accuracies.values()))

    #Printing final results
    # print(f"Accuracy:{test_acc:.4f}")
    # simple_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    # for data in simple_loader:
    #     out = model(data.x, data.edge_index, data.batch)  
    #     pred = out.argmax(dim=1)
    #     f1 = f1_score(y_true = data.y,y_pred = pred, average='macro')
    #     precision = precision_score(y_true = data.y,y_pred = pred, average='macro')
    #     recall = recall_score(y_true = data.y,y_pred = pred, average='macro')
    #     matrix = confusion_matrix(y_true = data.y,y_pred = pred)
    # print(matrix)
    # print(f"F1 Score:{f1:.4f}")
    # print(f"Precision:{precision:.4f}")
    # print(f"Recall:{recall:.4f}")




# plt.figure(figsize=(10,5))
# plt.title("Training and Testing Accuracy")
# plt.plot(val_losses,label="Test")
# plt.plot(train_losses,label="train")
# plt.xlabel("iterations")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=12345))
study.optimize(objective, n_trials=10)

best_f = study.best_value
print(best_f)

best_trial = study.best_trial
print(best_trial)


