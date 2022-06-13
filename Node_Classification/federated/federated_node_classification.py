import torch
import numpy as np
from helpers import trainer, tester, tester2
import matplotlib.pyplot as plt
from helpers import EarlyStopping
from random import sample, seed
torch.manual_seed(12345)
np.random.seed(12345)
torch.cuda.manual_seed_all(12345)
seed(12345)

def train_local_model(model, optimizer,criterion, data, args):
    accuracies = []
    early_stopping_callback = EarlyStopping(patience=20)
    for epoch in range(1, args.epochs+1):
        loss, test_acc = trainer(model, optimizer, criterion, data)
        accuracies.append(test_acc)
        early_stopping_callback(loss, model)
        if early_stopping_callback.early_stop:
            #print("Early stopping")
            break
    return max(accuracies)
    
    # plt.figure(figsize=(10,5))
    # plt.title("Training and Testing Accuracy")
    # plt.plot(val_losses,label="Test")
    # plt.plot(train_losses,label="train")
    # plt.xlabel("iterations")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.show()

def run_federated_node_classification(model_list, server_model, optimizer_list, criterion, client_data, args):
    private_acc = []
    global_acc = []
    inter_acc = []
    for z in range(len(model_list)):
        res = 0
        res2 = 0
        #print(f'+++++++++++++ Client {z} +++++++++++++')
        #Train and test to find best accuracy on private data
        best_acc_on_private_data = train_local_model(model_list[z], optimizer_list[z], criterion, client_data[z], args)
        private_acc.append(best_acc_on_private_data)
        #Test to find best accuracy on overall test set
        for y in range(len(client_data)):
            temp = tester(model_list[z], client_data[y])
            res = res + temp
            #Test to find best accuracy on other clients private test set
            if y!=z:
                res2 = res2 + temp
        
        acc_on_overall_test_set = res/len(client_data)
        acc_on_other_client_test_set = res2/(len(client_data)-1)
        global_acc.append(acc_on_overall_test_set)
        inter_acc.append(acc_on_other_client_test_set)


    #print("Server Aggregation")
    model_list2 = sample(model_list, args.parameterC)
    for param_tensor in model_list2[0].state_dict():
        avg = (sum(c.state_dict()[param_tensor] for c in model_list2))/len(model_list2)
        server_model.state_dict()[param_tensor].copy_(avg)
        for cl in model_list:
            cl.state_dict()[param_tensor].copy_(avg)
        
    test_acc_server = 0
    f1_server = 0
    precision_server = 0
    recall_server = 0
    for x in range(len(client_data)):
        acc, f1, precision, recall = tester2(server_model, client_data[x])
        test_acc_server = test_acc_server + acc
        f1_server = f1_server + f1
        precision_server = precision_server + precision
        recall_server = recall_server + recall
    test_acc_server = test_acc_server/len(client_data)
    f1_server = f1_server/len(client_data)
    precision_server = precision_server/len(client_data)
    recall_server = recall_server/len(client_data)
    print(f'Server Accuracy: {test_acc_server:.4f}')
    return private_acc, global_acc, inter_acc, test_acc_server, f1_server, precision_server, recall_server 