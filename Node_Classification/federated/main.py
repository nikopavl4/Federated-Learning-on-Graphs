import torch
import matplotlib.pyplot as plt
import argparse
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import numpy as np
import os
import time
# get the start time
st = time.time()
torch.manual_seed(12345)
np.random.seed(12345)
torch.cuda.manual_seed_all(12345)

parser = argparse.ArgumentParser(description='Insert Arguments')

parser.add_argument('--model', type=str, default="gcn", help='GNN used in training')
parser.add_argument("--dataset", type=str, default="cora", help="dataset used for training")
parser.add_argument("--split", type=float, default="0.8", help="test/train dataset split percentage")
parser.add_argument("--clients", type=int, default=4, help="number of clients")
parser.add_argument("--parameterC", type=int, default=4, help="num of clients randomly selected to participate in Federated Learning")
parser.add_argument("--hidden_channels", type=int, default=32, help="size of GNN hidden layer")
parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate for training")
parser.add_argument("--epochs", type=int, default=20, help="epochs for training")
parser.add_argument("--federated_rounds", type=int, default=40, help="federated rounds performed")

args = parser.parse_args()

##### DATA PREPARATION #####
#Import and Examine Dataset
if args.dataset.lower() == 'pubmed':
    dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=T.LargestConnectedComponents())
elif args.dataset.lower() == 'cora':
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.LargestConnectedComponents())
elif args.dataset.lower() == 'citeseer':
    dataset = Planetoid(root='data/Planetoid', name='CiteSeer',transform=T.LargestConnectedComponents())
else:
    print("No such dataset!")
    exit()

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the graph object.

print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')


#Split Graph and creating client datasets
from helpers import split_communities
client_graphs = split_communities(data, args.clients)

for i in range(args.clients):
    print(len(client_graphs[i]))

#Add one-hop neighborhood from other clients
from helpers import add_one_hop_neighbors
client_data, trusted_nodes = add_one_hop_neighbors(data, client_graphs)

for i in range(args.clients):
    print(client_data[i])

#Create train, test masks
from helpers import train_test_split
for k in range(len(client_data)):
    client_data[k]=train_test_split(client_data[k], trusted_nodes, k, args.split)



#### END OF DATA PREPARATION #####

#Run Federated Experiment
from model import GCN, SAGE
from federated_node_classification import run_federated_node_classification

model_list = []
optimizer_list = []
#GNN Initialization
if args.model.lower() == "gcn":
    for j in range(args.clients):
        model = GCN(hidden_channels=args.hidden_channels, features_in=dataset.num_node_features, features_out=dataset.num_classes)
        model_list.append(model)
        optimizer_list.append(torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4))
elif args.model.lower() == "sage":
    for j in range(args.clients):
        model = SAGE(hidden_channels=args.hidden_channels, features_in=dataset.num_node_features, features_out=dataset.num_classes)
        model_list.append(model)
        optimizer_list.append(torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4))
else:
    print("Model does not exist! Please select GCN or GraphSAGE")
    exit()

criterion = torch.nn.CrossEntropyLoss()


#Initialize server model
if args.model.lower() == "gcn":
    server_model = GCN(hidden_channels=args.hidden_channels, features_in=dataset.num_node_features, features_out=dataset.num_classes)
elif args.model.lower() == "sage":
    server_model = SAGE(hidden_channels=args.hidden_channels, features_in=dataset.num_node_features, features_out=dataset.num_classes)
else:
    print("Model does not exist! Please select GCN or GraphSAGE")
    exit()


#metrics obtained for diagrams
server_accuracy_per_round = {}
f1_per_round = {}
precision_per_round = {}
recall_per_round = {}
dict_keys = [*range(args.clients)]
client_private_acc_per_round = {k:[] for k in dict_keys}
client_global_acc_per_round= {k:[] for k in dict_keys}
inter_client_accuracy_per_round = {k:[] for k in dict_keys}

#print("+++++++++++++ Federated Node Classification +++++++++++++")

for i in range(args.federated_rounds+1):
   private_acc, global_acc, inter_acc, test_acc_server, f1_server, precision_server, recall_server  = run_federated_node_classification(model_list, server_model, optimizer_list, criterion, client_data, args)
   server_accuracy_per_round[i] = test_acc_server
   f1_per_round[i] = f1_server
   precision_per_round[i] = precision_server
   recall_per_round[i] = recall_server
   for j in range(args.clients):
       client_private_acc_per_round[j].append(private_acc[j])
       client_global_acc_per_round[j].append(global_acc[j])
       inter_client_accuracy_per_round[j].append(inter_acc[j])


print("++++++++++++++++++++++")


# get the end time
et = time.time()
# get the execution time
elapsed_time = et - st
print(f'Execution time: {elapsed_time:.2f} seconds')


print(f'Best Epoch = {max(server_accuracy_per_round, key=server_accuracy_per_round.get)}')
print(f'Accuracy = {max(list(server_accuracy_per_round.values())):.4f}')
print(f'F1 = {f1_per_round[max(server_accuracy_per_round, key=server_accuracy_per_round.get)]:.4f}')
print(f'Precision = {precision_per_round[max(server_accuracy_per_round, key=server_accuracy_per_round.get)]:.4f}')
print(f'Recall = {recall_per_round[max(server_accuracy_per_round, key=server_accuracy_per_round.get)]:.4f}')


#Plot and Save Results
dirname = os.path.dirname(__file__)
param_c = args.parameterC
num_cl = args.clients

# Write results to a txt file
filename = os.path.join(dirname, f'Result_Reports/{args.dataset}_{args.model}_{num_cl}_{param_c}_results.txt')
with open(filename, 'w') as f:
    print(f'Execution time: {elapsed_time:.2f} seconds', file=f)
    print(f'Best Epoch = {max(server_accuracy_per_round, key=server_accuracy_per_round.get)}', file=f)
    print(f'Accuracy = {max(list(server_accuracy_per_round.values())):.4f}', file=f)
    print(f'F1 = {f1_per_round[max(server_accuracy_per_round, key=server_accuracy_per_round.get)]:.4f}', file=f)
    print(f'Precision = {precision_per_round[max(server_accuracy_per_round, key=server_accuracy_per_round.get)]:.4f}', file=f)
    print(f'Recall = {recall_per_round[max(server_accuracy_per_round, key=server_accuracy_per_round.get)]:.4f}', file = f)
    for k in range(args.clients):
        print("##########", file=f)
        print(f'Client {k}', file=f)
        print(f'Best Testing on Private test set: federated;{max(client_private_acc_per_round[k]):.4f}  centralized;{client_private_acc_per_round[k][0]:.4f}', file=f)
        print(f'Best Testing on Global test set: federated;{max(client_global_acc_per_round[k]):.4f}  centralized;{client_global_acc_per_round[k][0]:.4f}', file=f)
        print(f'Best Testing on others clients test set: federated;{max(inter_client_accuracy_per_round[k]):.4f}  centralized;{inter_client_accuracy_per_round[k][0]:.4f}', file=f)

for i in range(args.clients):
    plt.figure(figsize=(10,5))
    plt.title(f"Accuracy on Client's Private Test Set per Federated Round / Client {i}")
    plt.plot(client_private_acc_per_round[i],label="Federated")
    plt.plot(list(client_private_acc_per_round[i][:1])*args.federated_rounds,label="Centralized")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.legend()
    filename = os.path.join(dirname, f'Result_Plots/{args.dataset}_{args.model}_{num_cl}_{param_c}_private_data_client_{i}.png')
    plt.savefig(filename)

    plt.figure(figsize=(10,5))
    plt.title(f"Accuracy on global test set per Federated Round / Client {i}")
    plt.plot(client_global_acc_per_round[i],label="Federated")
    plt.plot(list(client_global_acc_per_round[i][:1])*args.federated_rounds,label="Centralized")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.legend()
    filename = os.path.join(dirname, f'Result_Plots/{args.dataset}_{args.model}_{num_cl}_{param_c}_global_test_set_client_{i}.png')
    plt.savefig(filename)

    plt.figure(figsize=(10,5))
    plt.title(f"Accuracy on Other Client's Private Test Sets per Federated Round / Client {i}")
    plt.plot(inter_client_accuracy_per_round[i])
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    filename = os.path.join(dirname, f'Result_Plots/{args.dataset}_{args.model}_{num_cl}_{param_c}_inter_client_testing_client_{i}.png')
    plt.savefig(filename)


plt.figure(figsize=(10,5))
plt.title("Accuracy on Test set per Federated Round")
plt.plot(server_accuracy_per_round.values(), label="server")
for i in range(args.clients):
    plt.plot(client_global_acc_per_round[i], label=f'client {i}')
plt.xlabel("Federated Round")
plt.ylabel("Accuracy")
plt.legend()
filename = os.path.join(dirname, f'Result_Plots/{args.dataset}_{args.model}_{num_cl}_{param_c}_accuracy_per_round.png')
plt.savefig(filename)