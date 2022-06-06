import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from models.gcn import GCN
from models.sage import SAGE
from Client import Client
import argparse
from helpers import trainer, tester, k_fold, EarlyStopping
from server import perform_federated_round
import os
import time
# get the start time
st = time.time()

parser = argparse.ArgumentParser(description='Insert Arguments')

parser.add_argument('--model', type=str, default="gcn", help='GNN used in training')
parser.add_argument("--dataset", type=str, default="enzymes", help="dataset used for training")
parser.add_argument("--split", type=float, default="0.8", help="test/train dataset split percentage")
parser.add_argument("--clients", type=int, default=3, help="number of clients")
parser.add_argument("--parameterC", type=int, default=3, help="num of clients randomly selected to participate in Federated Learning")
parser.add_argument("--hidden_channels", type=int, default=32, help="size of GNN hidden layer")
parser.add_argument("--batch_size", type=int, default=32, help="input batch size for training (default: 16)")
parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate for optimizer")
parser.add_argument("--epochs", type=int, default=70, help="epochs for training")
parser.add_argument("--federated_rounds", type=int, default=30, help="federated rounds performed")

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

torch.manual_seed(12345)
dataset = dataset.shuffle()

#Train - Test First Split
train_dataset = dataset[:int(len(dataset)*args.split)]
test_dataset = dataset[int(len(dataset)*args.split):]

#Train Dataset split to Clients
startup = 0
Client_list = []
division = int(len(train_dataset)/args.clients)
print(division)
for i in range(args.clients):
    client_data = train_dataset[startup:division+startup]
    client_data = client_data.copy()
    print(client_data.data)
    #new_loader = DataLoader(client_data, batch_size=args.batch_size, shuffle=True)
    New_Client = Client(i,client_data)
    Client_list.append(New_Client)
    startup = startup + division

#Create and append GNN Model to every client
for MyClient in Client_list:
    print(f'Number of Client{MyClient.id} training graphs: {len(MyClient.dataset)}')
    if args.model.lower() == "gcn":
        model = GCN(hidden_channels=args.hidden_channels,dataset=dataset)
        MyClient.set_model(model,args)
    elif args.model.lower() == "sage":
        model = SAGE(hidden_channels=args.hidden_channels,dataset=dataset)
        MyClient.set_model(model, args)
    else:
        print("Model does not exist! Please select GCN or GraphSAGE")
        exit()

#Create dataloader for the global test set
print(f'Number of test graphs: {len(test_dataset)}')
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

#Initialize Server Model
if args.model.lower() == "gcn":
    server_model = GCN(hidden_channels=args.hidden_channels,dataset=dataset)
elif args.model.lower() == "sage":
    server_model = SAGE(hidden_channels=args.hidden_channels,dataset=dataset)

#metrics obtained for diagrams
server_accuracy_per_round = {}
f1_per_round = {}
precision_per_round = {}
recall_per_round = {}
dict_keys = [*range(args.clients)]
client_k_fold_acc_per_round = {k:[] for k in dict_keys}
client_global_acc_per_round= {k:[] for k in dict_keys}
inter_client_accuracy_per_round = {k:[] for k in dict_keys}


#Training and Testing process per federated round
for round_id in range(1,args.federated_rounds + 1):
    #For every client in every round
    for MyClient in Client_list:
        kfold_acc = 0
        max_kfold = -1
        global_max = -1
        inter_max = -1
        #For every fold created by cross-validation process
        for fold, (train_idx, test_idx) in enumerate(zip(*k_fold(MyClient.dataset, 5))):
            #Set Client train/test loaders
            MyClient.set_trainloader(DataLoader(MyClient.dataset[train_idx], batch_size=args.batch_size, shuffle=True))
            MyClient.set_testloader(DataLoader(MyClient.dataset[test_idx], batch_size=args.batch_size, shuffle=True))
            #Train for every fold
            early_stopping_callback = EarlyStopping(patience=20)
            for epoch in range(1, args.epochs + 1):
                loss = trainer(MyClient.model, MyClient.optimizer,MyClient.trainloader)
                #Test for every fold
                temp = tester(MyClient.model,MyClient.testloader)
                if temp > max_kfold:
                    max_kfold = temp
                early_stopping_callback(loss, MyClient.model)
                if early_stopping_callback.early_stop:
                    break
            kfold_acc = kfold_acc + max_kfold
        #Keep all epochs best
        client_k_fold_acc_per_round[MyClient.id].append(kfold_acc/5)

        #Create an overall train dataset loader
        MyClient.set_trainloader(DataLoader(MyClient.dataset, batch_size=args.batch_size, shuffle=True))

        #Create and inter client data loader - contains train data from other clients
        gdataset = dataset[0:0]
        for every_client in Client_list:
            if every_client != MyClient:
                gdataset = gdataset + every_client.dataset.copy()
        inter_data_loader = DataLoader(gdataset, batch_size=args.batch_size, shuffle=True)

        #Train on overall train set - Test on global and inter data
        early_stopping_callback = EarlyStopping(patience=20)
        for epoch in range(1, args.epochs + 1):
            loss2 = trainer(MyClient.model, MyClient.optimizer,MyClient.trainloader)
            #Test on global test set                   
            global_test_acc = tester(MyClient.model,test_loader)
            if global_test_acc > global_max:
                global_max = global_test_acc
                        #Test on global test set                   
            inter_client_test_acc = tester(MyClient.model,inter_data_loader)
            if inter_client_test_acc > inter_max:
                inter_max = inter_client_test_acc
            early_stopping_callback(loss2, MyClient.model)
            if early_stopping_callback.early_stop:
                break
        #Keep all epochs best
        client_global_acc_per_round[MyClient.id].append(global_max)
        inter_client_accuracy_per_round[MyClient.id].append(inter_max)

    server_acc, f1, precision, recall = perform_federated_round(server_model, Client_list,round_id,test_loader, args)
    server_accuracy_per_round[round_id] = server_acc
    f1_per_round[round_id] = f1
    precision_per_round[round_id] = precision
    recall_per_round[round_id] = recall


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
        print(f'Best Testing on Private data: federated;{max(client_k_fold_acc_per_round[k]):.4f}  centralized;{client_k_fold_acc_per_round[k][0]:.4f}', file=f)
        print(f'Best Testing on Global data set: federated;{max(client_global_acc_per_round[k]):.4f}  centralized;{client_global_acc_per_round[k][0]:.4f}', file=f)
        print(f'Best Testing on others clients data: federated;{max(inter_client_accuracy_per_round[k]):.4f}  centralized;{inter_client_accuracy_per_round[k][0]:.4f}', file=f)

for i in range(args.clients):
    plt.figure(figsize=(10,5))
    plt.title(f"5-fold Cross Validated Accuracy on Client's Private Data per Federated Round / Client {i}")
    plt.plot(client_k_fold_acc_per_round[i],label="Federated")
    plt.plot(list(client_k_fold_acc_per_round[i][:1])*args.federated_rounds,label="Centralized")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.legend()
    filename = os.path.join(dirname, f'Result_Plots/{args.dataset}_{args.model}_{num_cl}_{param_c}_x_valid_private_data_client_{i}.png')
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
    plt.title(f"Accuracy on Other Client's Private Data per Federated Round / Client {i}")
    plt.plot(inter_client_accuracy_per_round[i],label="Federated")
    plt.plot(list(inter_client_accuracy_per_round[i][:1])*args.federated_rounds,label="Centralized")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.legend()
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

