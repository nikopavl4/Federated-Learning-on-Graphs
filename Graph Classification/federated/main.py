import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from models.gcn import GCN
from models.sage import SAGE
from Client import Client
import argparse
from helpers import trainer, tester
from server import perform_federated_round
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

parser = argparse.ArgumentParser(description='Insert Arguments')

parser.add_argument('--model', type=str, default="gcn", help='GNN used in training')
parser.add_argument("--dataset", type=str, default="enzymes", help="dataset used for training")
parser.add_argument("--split", type=float, default="0.8", help="test/train dataset split percentage")
parser.add_argument("--clients", type=int, default=3, help="number of clients")
parser.add_argument("--parameterC", type=int, default=2, help="num of clients randomly selected to participate in Federated Learning")
parser.add_argument("--hidden_channels", type=int, default=32, help="size of GNN hidden layer")
parser.add_argument("--batch_size", type=int, default=32, help="input batch size for training (default: 16)")
parser.add_argument("--epochs", type=int, default=50, help="epochs for training")
parser.add_argument("--federated_rounds", type=int, default=30, help="federated rounds performed")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
for i in range(args.clients):
    client_data = dataset[startup:division+startup]
    new_loader = DataLoader(client_data, batch_size=args.batch_size, shuffle=True)
    New_Client = Client(i,client_data, new_loader)
    Client_list.append(New_Client)
    startup = division

for MyClient in Client_list:
    print(f'Number of Client{MyClient.id} training graphs: {len(MyClient.dataset)}')
    if args.model.lower() == "gcn":
        model = GCN(hidden_channels=args.hidden_channels,dataset=dataset)
        model.to(device)
        MyClient.set_model(model)
    elif args.model.lower() == "sage":
        model = SAGE(hidden_channels=args.hidden_channels,dataset=dataset)
        MyClient.set_model(model)
    else:
        print("Model does not exist! Please select GCN or GraphSAGE")
        exit()
print(f'Number of test graphs: {len(test_dataset)}')
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)




server_accuracy_per_round = []
server_model = GCN(hidden_channels=args.hidden_channels,dataset=dataset)


for round_id in range(1,args.federated_rounds + 1):
    for epoch in range(1, args.epochs + 1):
        for MyClient in Client_list:
            trainer(MyClient.model, MyClient.optimizer,MyClient.dataloader)
            train_acc_1 = tester(MyClient.model,MyClient.dataloader)
            test_acc_1 = tester(MyClient.model,test_loader)
            print(f'Client{MyClient.id}:')
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc_1:.4f}, Test Acc: {test_acc_1:.4f}')

    server_acc= perform_federated_round(server_model, Client_list,round_id,test_loader, args)
    server_accuracy_per_round.append(server_acc)

simple_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
for data in simple_loader:
    out = server_model(data.x, data.edge_index, data.batch)  
    pred = out.argmax(dim=1)
    f1 = f1_score(y_true = data.y,y_pred = pred, average='macro')
    precision = precision_score(y_true = data.y,y_pred = pred, average='macro')
    recall = recall_score(y_true = data.y,y_pred = pred, average='macro')
    matrix = confusion_matrix(y_true = data.y,y_pred = pred)
print(matrix)
print(f"F1 Score:{f1:.4f}")
print(f"Precision:{precision:.4f}")
print(f"Recall:{recall:.4f}")

plt.figure(figsize=(10,5))
plt.title("Server Accuracy per Federated Round")
plt.plot(server_accuracy_per_round,label="Test")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

