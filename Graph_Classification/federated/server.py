import fractions
from helpers import tester
import random
from random import sample
from numpy import average
from sklearn.metrics import f1_score, precision_score, recall_score
random.seed(12345)

def perform_federated_avg(server_model,Client_list,round_id,test_loader, args):
    fractions = []
    Client_list2 = sample(Client_list, args.parameterC)
    total_samp = sum(len(c.dataset) for c in Client_list2)
    for c in Client_list2:
        fractions.append(len(c.dataset)/total_samp)
    for param_tensor in Client_list2[0].model.state_dict():
        #avg = average((c.model.state_dict()[param_tensor] for c in Client_list2), weights=fractions)
        fraction_pointer = 0
        avg = 0
        for c in Client_list2:
            avg = avg + c.model.state_dict()[param_tensor] * fractions[fraction_pointer]
            fraction_pointer+=1
        server_model.state_dict()[param_tensor].copy_(avg)
        for cl in Client_list:
            cl.model.state_dict()[param_tensor].copy_(avg)

    test_acc_server = tester(server_model,test_loader)
    print(f'##Round ID={round_id}')
    print(f'Test Acc: {test_acc_server:.4f}')
    for data in test_loader:
        out = server_model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)
        f1 = f1_score(y_true = data.y,y_pred = pred, average='macro', zero_division=1)
        precision = precision_score(y_true = data.y,y_pred = pred, average='macro', zero_division=1)
        recall = recall_score(y_true = data.y,y_pred = pred, average='macro', zero_division=1)
    # print(f"F1 Score:{f1:.4f}")
    # print(f"Precision:{precision:.4f}")
    # print(f"Recall:{recall:.4f}")

    return test_acc_server, f1, precision, recall