import time,random
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from data_preprocess import load_data, split_communities, create_clients
from sklearn.metrics import confusion_matrix

torch.manual_seed(12345)
#random.seed(12345)
#np.random.seed(12345)

global_graph, num_of_features, num_of_classes = load_data()

from model import GCN
model = GCN(nfeat=num_of_features, nhid=16, nclass=num_of_classes, dropout=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

from Client import Client
Client0 = Client(0,nx.to_numpy_matrix(global_graph),nx.get_node_attributes(global_graph, "x"),nx.get_node_attributes(global_graph, "y"))
train_mask = torch.randn((34)) < 0.7
test_mask = torch.logical_not(train_mask)

print(torch.tensor(list(Client0.x.values())))
print(torch.tensor(Client0.A.astype(np.float32)))
labels = torch.tensor(list(Client0.y.values()))

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(torch.tensor(list(Client0.x.values())), torch.tensor(Client0.A.astype(np.float32)))
    loss_train = F.nll_loss(output[train_mask], labels[train_mask])
    acc_train = accuracy(output[train_mask], labels[train_mask])
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(torch.tensor(list(Client0.x.values())), torch.tensor(Client0.A.astype(np.float32)))
    loss_test = F.nll_loss(output[test_mask], labels[test_mask])
    acc_test = accuracy(output[test_mask], labels[test_mask])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    pred = output.argmax(dim=1)
    conf_matrix = confusion_matrix(labels[test_mask], pred[test_mask])
    print(conf_matrix)


# Train model
t_total = time.time()
for epoch in range(101):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()