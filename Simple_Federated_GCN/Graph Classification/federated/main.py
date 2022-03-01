import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from models.gcn import GCN
from models.sage import SAGE
import sys
from helpers import trainer, tester
from server import perform_federated_round

#Import dataset 450 training graphs/150 testing graphs
dataset = TUDataset(root='data/TUDataset', name='ENZYMES')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset1 = dataset[:150]
train_dataset2 = dataset[150:300]
train_dataset3 = dataset[300:450]
test_dataset = dataset[450:]

print(f'Number of Client1 training graphs: {len(train_dataset1)}')
print(f'Number of Client2 training graphs: {len(train_dataset2)}')
print(f'Number of Client3 training graphs: {len(train_dataset3)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader1 = DataLoader(train_dataset1, batch_size=150, shuffle=True)
train_loader2 = DataLoader(train_dataset1, batch_size=150, shuffle=True)
train_loader3 = DataLoader(train_dataset3, batch_size=150, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=150, shuffle=False)


if sys.argv[1].lower() == "gcn":
    model1 = GCN(hidden_channels=16,dataset=dataset)
    model2 = GCN(hidden_channels=16,dataset=dataset)
    model3 = GCN(hidden_channels=16,dataset=dataset)
elif sys.argv[1].lower() == "graphsage":
    model1 = SAGE(hidden_channels=16,dataset=dataset)
    model2 = SAGE(hidden_channels=16,dataset=dataset)
    model3 = SAGE(hidden_channels=16,dataset=dataset)
else:
    print("Model does not exist! Please select GCN or GraphSAGE")
    exit()



for epoch in range(1, 70):
    #Client1
    trainer(model1,train_loader1)
    train_acc_1 = tester(model1,train_loader1)
    test_acc_1 = tester(model1,test_loader)
    #print("Client1:")
    #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc_1:.4f}, Test Acc: {test_acc_1:.4f}')
    
    #Client2
    trainer(model2,train_loader2)
    train_acc_2 = tester(model2,train_loader2)
    test_acc_2 = tester(model2,test_loader)
    #print("Client2:")
    #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc_2:.4f}, Test Acc: {test_acc_2:.4f}')

    #Client3
    trainer(model3,train_loader3)
    train_acc_3 = tester(model2,train_loader3)
    test_acc_3 = tester(model2,test_loader)
    #print("Client3:")
    #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc_3:.4f}, Test Acc: {test_acc_3:.4f}')

if sys.argv[1].lower() == "gcn":
    server_model = GCN(hidden_channels=16,dataset=dataset)
    print("Model: GCN")
elif sys.argv[1].lower() == "graphsage":
    server_model = SAGE(hidden_channels=16,dataset=dataset)
    print("Model: GraphSAGE")
else:
    print("Model does not exist! Please select GCN or GraphSAGE")
    exit()

server_accuracy_per_round = []

for round_id in range(1,11):
    server_acc= perform_federated_round(server_model,model1, model2, model3, round_id,test_loader, train_loader1, train_loader2, train_loader3)
    server_accuracy_per_round.append(server_acc)

plt.figure(figsize=(10,5))
plt.title("Server Accuracy per Federated Round")
plt.plot(server_accuracy_per_round,label="Test")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

