import torch
import matplotlib.pyplot as plt
import argparse
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from helpers import trainer, tester,tester2, simple_train_test_split
from model import GCN, SAGE
import numpy as np
torch.manual_seed(12345)
torch.cuda.manual_seed_all(12345)
np.random.seed(12345)
import optuna

parser = argparse.ArgumentParser(description='Insert Arguments')

parser.add_argument('--model', type=str, default="gcn", help='GNN used in training')
parser.add_argument("--dataset", type=str, default="cora", help="dataset used for training")
parser.add_argument("--split", type=float, default=0.8, help="test/train dataset split percentage")
# parser.add_argument("--hidden_channels", type=int, default=16, help="size of GNN hidden layer")
# parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate for training")
parser.add_argument("--epochs", type=int, default=150, help="epochs for training")

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

def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
        'hidden_channels': trial.suggest_discrete_uniform('hidden_channels', 16, 64, 16),
    }
    #GNN Initialization
    if args.model.lower() == "gcn":
        model = GCN(hidden_channels=int(params['hidden_channels']), features_in=dataset.num_node_features, features_out=dataset.num_classes)
    elif args.model.lower() == "sage":
        model = SAGE(hidden_channels=int(params['hidden_channels']), features_in=dataset.num_node_features, features_out=dataset.num_classes)
    else:
        print("Model does not exist! Please select GCN or GraphSAGE")
        exit()

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()


    test_accuracies = {}
    train_accuracies = {}
    test_f1 = {}
    test_precision = {}
    test_recall = {}
    print("+++++++++++++ Centralized Node Classification +++++++++++++")
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = trainer(model, optimizer, criterion, data)
        test_acc, testf1, testpre, testrec = tester2(model, data)
        train_accuracies[epoch] = train_acc
        test_accuracies[epoch] = test_acc
        test_f1[epoch] = testf1
        test_precision[epoch] = testpre
        test_recall[epoch] = testrec
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    
    print(f'Best Epoch = {max(test_accuracies, key=test_accuracies.get)}')
    print(f'F1 = {test_f1[max(test_accuracies, key=test_accuracies.get)]}')
    print(f'Precision = {test_precision[max(test_accuracies, key=test_accuracies.get)]}')
    print(f'Recall = {test_recall[max(test_accuracies, key=test_accuracies.get)]}')
    return max(list(test_accuracies.values()))

    # #Model Evaluation
    # model.eval()
    # out = model(data.x, data.edge_index)
    # pred = out.argmax(dim=1)  # Use the class with highest probability.
    # from sklearn.metrics import confusion_matrix
    # conf_matrix = confusion_matrix(data.y[data.test_mask], pred[data.test_mask])
    # f1 = f1_score(y_true = data.y[data.test_mask],y_pred = pred[data.test_mask], average='macro', zero_division=1)
    # precision = precision_score(y_true =data.y[data.test_mask],y_pred = pred[data.test_mask], average='macro', zero_division=1)
    # recall = recall_score(y_true = data.y[data.test_mask],y_pred = pred[data.test_mask], average='macro', zero_division=1)
    # print(conf_matrix)
    # print(f"F1 Score:{f1:.4f}")
    # print(f"Precision:{precision:.4f}")
    # print(f"Recall:{recall:.4f}")


# #Plot Diagrams
# ConfusionMatrixDisplay.from_predictions(y_true = data.y[data.test_mask],y_pred = pred[data.test_mask], cmap=plt.cm.Blues) 
# plt.figure(figsize=(10,5))
# plt.title("Training and Testing Accuracy")
# plt.plot(val_losses,label="Test")
# plt.plot(train_losses,label="train")
# plt.xlabel("iterations")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=12345))
study.optimize(objective, n_trials=3)

best_f = study.best_value
print(best_f)

best_trial = study.best_trial
print(best_trial)

