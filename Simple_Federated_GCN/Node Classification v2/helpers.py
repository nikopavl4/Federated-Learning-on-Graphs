import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

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
def tester(model, Client):
        labels = torch.tensor(list(Client.y.values()))
        model.eval()
        output = model(torch.tensor(list(Client.x.values())), torch.tensor(Client.A.astype(np.float32)))
        acc_test = accuracy(output[Client.test_mask], labels[Client.test_mask])
        print("Test set results:","accuracy= {:.4f}".format(acc_test.item()))
        pred = output.argmax(dim=1)
        conf_matrix = confusion_matrix(labels[Client.test_mask], pred[Client.test_mask])
        print("+++ Confusion Matrix +++")
        print(conf_matrix)
        return acc_test