import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

def simple_train_test_split(Client, split_percentage):
    mask = torch.randn((len(Client.y))) < split_percentage
    nmask = torch.logical_not(mask)
    Client.train_mask = mask
    Client.test_mask = nmask
    return Client

#A function to compute accuracy
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

#Universal Tester function
def tester(model, Client, machine):
        labels = torch.tensor(list(Client.y.values()))
        model.eval()
        output = model(machine, Client.id)
        acc_test = accuracy(output[Client.test_mask], labels[Client.test_mask])
        #print("Test set results:","accuracy= {:.4f}".format(acc_test.item()))
        pred = output.argmax(dim=1)
        f1 = f1_score(y_true = labels[Client.test_mask],y_pred = pred[Client.test_mask], average='macro')
        precision = precision_score(y_true = labels[Client.test_mask],y_pred = pred[Client.test_mask], average='macro', zero_division=1)
        recall = recall_score(y_true = labels[Client.test_mask],y_pred = pred[Client.test_mask], average='macro')
        return acc_test.item(), f1, precision, recall