import torch
import matplotlib.pyplot as plt
import argparse
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import numpy as np
torch.manual_seed(12345)
np.random.seed(12345)
torch.cuda.manual_seed_all(12345)

parser = argparse.ArgumentParser(description='Insert Arguments')

parser.add_argument('--model', type=str, default="gcn", help='GNN used in training')
parser.add_argument("--dataset", type=str, default="cora", help="dataset used for training")
parser.add_argument("--clients", type=int, default=3, help="number of clients")
parser.add_argument("--parameterC", type=int, default=3, help="num of clients randomly selected to participate in Federated Learning")
parser.add_argument("--hidden_channels", type=int, default=16, help="size of GNN hidden layer")
parser.add_argument("--learning_rate", type=int, default=0.01, help="learning rate for training")
parser.add_argument("--epochs", type=int, default=20, help="epochs for training")
parser.add_argument("--federated_rounds", type=int, default=30, help="federated rounds performed")

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


#Add one-hop neighborhood from other clients
from helpers import add_one_hop_neighbors
client_data, trusted_nodes = add_one_hop_neighbors(data, client_graphs)

#Create train, test masks
# from helpers import train_test_split
# data1 = train_test_split(data1, trusted_nodes, 0)
# data2 = train_test_split(data2, trusted_nodes, 1)
# data3 = train_test_split(data3, trusted_nodes, 2)
node_split_transform  = T.RandomNodeSplit(split='test_rest', num_train_per_class=75, num_val=0)
for k in range(len(client_data)):
    client_data[k]=node_split_transform(client_data[k])

##### END OF DATA PREPARATION #####

#Run Federated Experiment
from model import GCN, SAGE
from federated_node_classification import run_federated_node_classifcation

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

# scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=2, gamma=0.1)
# scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=2, gamma=0.1)
# scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=2, gamma=0.1)

#Initialize server model
#GNN Initialization
if args.model.lower() == "gcn":
    server_model = GCN(hidden_channels=args.hidden_channels, features_in=dataset.num_node_features, features_out=dataset.num_classes)
elif args.model.lower() == "sage":
    server_model = SAGE(hidden_channels=args.hidden_channels, features_in=dataset.num_node_features, features_out=dataset.num_classes)
else:
    print("Model does not exist! Please select GCN or GraphSAGE")
    exit()


results = []
print("+++++++++++++ Federated Node Classification +++++++++++++")
for i in range(args.federated_rounds+1):
   results.append(run_federated_node_classifcation(model_list, server_model, optimizer_list, criterion, client_data, args))
   #mymodel = run_federated_node_classifcation(model1, model2, model3, server_model, optimizer1, optimizer2, optimizer3, criterion, data1, data2, data3)

# # mymodel.eval()
# # out = mymodel(data.x, data.edge_index)
# # pred = out.argmax(dim=1)  # Use the class with highest probability.
# # from sklearn.metrics import confusion_matrix
# # conf_matrix = confusion_matrix(data.y[data.test_mask], pred[data.test_mask])
# # print(conf_matrix)

#Print Final Results
print('++++ Final Results ++++')
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
server_model.eval()
out = server_model(data.x, data.edge_index)
pred = out.argmax(dim=1)  # Use the class with highest probability.
f1 = f1_score(y_true = data.y,y_pred = pred, average='macro')
precision = precision_score(y_true = data.y,y_pred = pred, average='macro')
recall = recall_score(y_true = data.y,y_pred = pred, average='macro')
matrix = confusion_matrix(y_true = data.y,y_pred = pred)
print(matrix)
print(f"F1 Score:{f1:.4f}")
print(f"Precision:{precision:.4f}")
print(f"Recall:{recall:.4f}")

if args.clients == 3:
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