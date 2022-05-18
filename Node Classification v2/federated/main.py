### A federated learning Example-Setting for Node Classification using FHE to avoid data leakage between Clients
import torch, copy,random
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from helpers import tester
from data_preprocess import load_data, split_communities, create_clients
torch.manual_seed(12345)
random.seed(12345)


parser = argparse.ArgumentParser(description='Insert Arguments')

parser.add_argument("--dataset", type=str, default="cora", help="dataset used for training")
parser.add_argument("--clients", type=int, default=3, help="number of clients")
parser.add_argument("--split", type=float, default=0.8, help="test/train dataset split percentage")
parser.add_argument("--parameterC", type=int, default=2, help="num of clients randomly selected to participate in Federated Learning")
parser.add_argument("--hidden_channels", type=int, default=16, help="size of GNN hidden layer")
parser.add_argument("--learning_rate", type=int, default=0.01, help="learning rate for training")
parser.add_argument("--epochs", type=int, default=20, help="epochs for training")
parser.add_argument("--federated_rounds", type=int, default=30, help="federated rounds performed")

args = parser.parse_args()


global_graph, num_of_features, num_of_classes = load_data(args)

communities = split_communities(global_graph, args.clients)

client_list = create_clients(communities)


#Initialize Clients Models - Create Train/Test masks
from helpers import simple_train_test_split
for i in range(len(client_list)):
    client_list[i].initialize(args.hidden_channels, args.learning_rate, num_of_features, num_of_classes)
    client_list[i] = simple_train_test_split(client_list[i], args.split)



#Create and initialize Aggregation Server
from Server import Aggregation_Server
MyServer = Aggregation_Server()
MyServer.initialize(num_of_features, num_of_classes, args.hidden_channels)

#Create and initialize Security Machine
from SecMachine import SecMachine
MyMachine = SecMachine(nx.to_numpy_matrix(global_graph))
for j in range(len(client_list)):
    MyMachine.add_secure_client(client_list[j])

for k in range(1, len(client_list)+1):
    for l in range(1, len(client_list)+1):
        if k!=l:
            MyMachine.find_connections(k,l)

# res=0
# localdraw1 = []
# localdraw2 = []
# localdraw3 = []
serverdraw = []
#Federated Learning
for round in range(args.federated_rounds+1):
    #Train Local Models
    for x in range(len(client_list)):
        client_list[x].train_local_model(epochs=args.epochs+1, machine = MyMachine)

    # if round == 0:
    #     res = (tester(Client2.model, Client1, MyMachine) +tester(Client2.model, Client2, MyMachine) + tester(Client2.model, Client3, MyMachine))/3
    #     localdraw1.append(res)
    # else:
    #     localdraw1.append(res)

    # localdraw1.append(tester(Client1.model, Client1, MyMachine))
    # localdraw2.append(tester(Client2.model, Client2, MyMachine))
    # localdraw3.append(tester(Client3.model, Client3, MyMachine))

    #FedAvg Local Models on Server based on parameter C
    client_list= MyServer.perform_fed_avg(client_list, args.parameterC)

    #Test Server Model on every clients data
    server_acc = 0
    server_f1 = 0
    server_precision = 0
    server_recall = 0
    for y in range(len(client_list)):
        accuracy, f1, precision, recall = tester(MyServer.model, client_list[y], MyMachine)
        server_acc = server_acc + accuracy
        server_f1 = server_f1 + f1
        server_precision = server_precision + precision
        server_recall = server_recall + recall
    server_acc = server_acc/args.clients
    print(f'+++ Final Results on Server - Round {round} +++')
    print(f'Server Accuracy: {server_acc}')
    print(f"F1 Score:{server_f1/args.clients:.4f}")
    print(f"Precision:{server_precision/args.clients:.4f}")
    print(f"Recall:{server_recall/args.clients:.4f}")
    serverdraw.append(server_acc)

#print(res)
plt.figure(figsize=(10,5))
plt.title("Testing Accuracy per Federated Round")
#plt.title("Federated vs Centralized Learning")
# plt.plot(localdraw1,label="Client1")
# plt.plot(localdraw2,label="Client2")
# plt.plot(localdraw3,label="Client3")
plt.plot(serverdraw,label="Server")
plt.xlabel("Federated Round")
plt.ylabel("Accuracy")
plt.legend()
plt.show()