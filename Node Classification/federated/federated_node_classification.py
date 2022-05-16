import torch
import numpy as np
from helpers import trainer, tester
import matplotlib.pyplot as plt
from helpers import EarlyStopping
from random import sample, seed
torch.manual_seed(12345)
np.random.seed(12345)
torch.cuda.manual_seed_all(12345)
seed(12345)

def train_local_model(model, optimizer,criterion, data, args):
    # val_losses = []
    # train_losses = []
    early_stopping_callback = EarlyStopping(patience=10)
    for epoch in range(1, args.epochs+1):
        loss, train_acc = trainer(model, optimizer, criterion, data)
        # train_losses.append(train_acc)
        # val_losses.append(test_acc)
        early_stopping_callback(loss, model)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}')
        if early_stopping_callback.early_stop:
            print("Early stopping")
            break
    
    # plt.figure(figsize=(10,5))
    # plt.title("Training and Testing Accuracy")
    # plt.plot(val_losses,label="Test")
    # plt.plot(train_losses,label="train")
    # plt.xlabel("iterations")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.show()

def run_federated_node_classifcation(model_list, server_model, optimizer_list, criterion, client_data, args):
    accuracy_per_round = []
    for z in range(len(model_list)):
        res = 0
        print(f'+++++++++++++ Client{z} +++++++++++++')
        train_local_model(model_list[z], optimizer_list[z], criterion, client_data[z], args)
        for y in range(len(client_data)):
            res = res + tester(model_list[z], client_data[y])
        accuracy_per_round.append(res/len(model_list))

    print("Server Aggregation")
    model_list2 = sample(model_list, args.parameterC)
    for param_tensor in model_list2[0].state_dict():
        avg = (sum(c.state_dict()[param_tensor] for c in model_list2))/len(model_list2)
        server_model.state_dict()[param_tensor].copy_(avg)
        for cl in model_list:
            cl.state_dict()[param_tensor].copy_(avg)
    test_acc_server = 0
    for x in range(len(client_data)):
        test_acc_server = test_acc_server + tester(server_model, client_data[x])
    test_acc_server = test_acc_server/len(client_data)
    print("Server's Accuracy")
    print(f'Test Acc: {test_acc_server:.4f}')
    accuracy_per_round.append(test_acc_server)
    return accuracy_per_round