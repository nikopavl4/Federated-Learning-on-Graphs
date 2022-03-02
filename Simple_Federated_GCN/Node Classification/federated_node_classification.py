from model import GCN
import torch
from helpers import trainer, tester
import matplotlib.pyplot as plt


def train_local_model(model, optimizer,criterion, data):
    # val_losses = []
    # train_losses = []
    for epoch in range(1, 50):
        loss, train_acc = trainer(model, optimizer, criterion, data)
        test_acc = tester(model, data) 
        # train_losses.append(train_acc)
        # val_losses.append(test_acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    return test_acc
    
    # plt.figure(figsize=(10,5))
    # plt.title("Training and Testing Accuracy")
    # plt.plot(val_losses,label="Test")
    # plt.plot(train_losses,label="train")
    # plt.xlabel("iterations")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.show()

def run_federated_node_classifcation(model1, model2, model3, server_model, optimizer1, optimizer2, optimizer3, criterion, data1, data2, data3):
    accuracy_per_round = []
    print("+++++++++++++ Client1 +++++++++++++")
    res = train_local_model(model1, optimizer1, criterion, data1)
    accuracy_per_round.append(res)
    print("+++++++++++++ Client2 +++++++++++++")
    res = train_local_model(model2, optimizer2, criterion, data2)
    accuracy_per_round.append(res)
    print("+++++++++++++ Client3 +++++++++++++")
    res = train_local_model(model3, optimizer3, criterion, data3)
    accuracy_per_round.append(res)

    print("Server Aggregation")
    for param_tensor in model1.state_dict():
        avg = (model1.state_dict()[param_tensor] + model2.state_dict()[param_tensor] + model3.state_dict()[param_tensor])/3
        server_model.state_dict()[param_tensor].copy_(avg)
        model1.state_dict()[param_tensor].copy_(avg)
        model2.state_dict()[param_tensor].copy_(avg)
        model3.state_dict()[param_tensor].copy_(avg)

    test_acc_server = (tester(server_model, data1) + tester(server_model, data2) + tester(server_model, data3))/3
    print("Server's Accuracy")
    print(f'Test Acc: {test_acc_server:.4f}')
    accuracy_per_round.append(test_acc_server)
    return accuracy_per_round