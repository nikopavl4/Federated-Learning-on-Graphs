import torch
from random import seed
from torch_geometric.datasets import KarateClub,BAShapes
import networkx as nx
import matplotlib.pyplot as plt
torch.manual_seed(12345)

#Import and Examine Dataset
dataset = BAShapes(connection_distribution='uniform')

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
C1, C2, C3 = split_communities(data)

#Add one-hop neighborhood from other clients
from helpers import add_one_hop_neighbors
data1, data2, data3, trusted_nodes = add_one_hop_neighbors(data, C1, C2, C3)

#Create train, test masks
from helpers import train_test_split
data1 = train_test_split(data1, trusted_nodes, 0)
data2 = train_test_split(data2, trusted_nodes, 1)
data3 = train_test_split(data3, trusted_nodes, 2)

#Run Centralized Experiment
from centralized_node_classification import run_centralized_node_classifcation, simple_train_test_split
data = simple_train_test_split(data)
run_centralized_node_classifcation(dataset,data)


#Run Federated Experiment
import copy
from model import GCN
from federated_node_classification import run_federated_node_classifcation
model1 = GCN(hidden_channels=16, dataset= dataset)
model2 = copy.deepcopy(model1)
model3 = copy.deepcopy(model1)

optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01, weight_decay=5e-4)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01, weight_decay=5e-4)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

server_model = GCN(hidden_channels=16, dataset=dataset)
federated_rounds = 100
results = []
print("+++++++++++++ Federated Node Classification +++++++++++++")
for i in range(federated_rounds):
   results.append(run_federated_node_classifcation(model1, model2, model3, server_model, optimizer1, optimizer2, optimizer3, criterion, data1, data2, data3))

draw1 = []
draw2 = []
draw3 = []
draw4 = []

for k in range(len(results)):
    draw1.append(results[k][0])
    draw2.append(results[k][1])
    draw3.append(results[k][2])
    draw4.append(results[k][3])

plt.figure(figsize=(10,5))
plt.title("Testing Accuracy per Federated Round")
plt.plot(draw1,label="Client1")
plt.plot(draw2,label="Client2")
plt.plot(draw3,label="Client3")
plt.plot(draw4,label="Server")
plt.xlabel("Federated Round")
plt.ylabel("Accuracy")
plt.legend()
plt.show()