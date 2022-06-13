from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
torch.manual_seed(12345)

def split_communities(data, clients):
    G = to_networkx(data, to_undirected=True, node_attrs=['x','y'])
    communities = sorted(nx.community.asyn_fluidc(G, clients, max_iter = 5000, seed= 12345))

    node_groups = []
    for com in communities:
        node_groups.append(list(com))   
    list_of_clients = []

    for i in range(clients):
        list_of_clients.append(G.subgraph(node_groups[i]).copy())

    return list_of_clients

def find_inter_cluster_edges(G,client_graphs, i):
    trusted_nodes = []
    H = client_graphs[i].copy()

    for k in range(len(client_graphs)):
        if (k!=i):
            for node in list(client_graphs[k].nodes(data=True)):
                for neighbor in G.neighbors(node[0]):
                    if client_graphs[i].has_node(neighbor) and (not H.has_node(node[0])):
                        H.add_node(node[0], x = node[1]['x'], y = node[1]['y'])
                        H.add_edge(node[0], neighbor)
                        trusted_nodes.append(node[0])
    return H, trusted_nodes
        
def add_one_hop_neighbors(data, client_graphs):
    G = to_networkx(data, to_undirected=True, node_attrs=['x','y'])
    client_data = []
    trusted_nodes = []
    for i in range(len(client_graphs)):
        H, trusted_nodes2 = find_inter_cluster_edges(G, client_graphs, i)
        client_data.append(from_networkx(H))
        trusted_nodes.append(trusted_nodes2)

    return client_data, trusted_nodes

def train_test_split(data, trusted_nodes, client_id, split_percentage):
    mask = torch.randn((data.num_nodes)) < split_percentage
    nmask = torch.logical_not(mask)
    mask[data.num_nodes - len(trusted_nodes[client_id]):] = False
    nmask[data.num_nodes - len(trusted_nodes[client_id]):] = False
    train_mask = mask
    test_mask = nmask
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data

def trainer(model,optimizer,criterion, data):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return loss, test_acc

def tester(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


def tester2(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    f1 = f1_score(y_true = data.y[data.test_mask],y_pred = pred[data.test_mask], average='macro', zero_division=1)
    precision = precision_score(y_true = data.y[data.test_mask],y_pred = pred[data.test_mask], average='macro', zero_division=1)
    recall = recall_score(y_true = data.y[data.test_mask],y_pred = pred[data.test_mask], average='macro', zero_division=1)
    return test_acc, f1, precision, recall

class EarlyStopping:
    def __init__(self, patience=20, change=0., path='euclid_model', mode='minimize'):
        """
        patience: Waiting threshold for val loss to improve.
        change: Minimum change in the model's quality.
        path: Path for saving the model to.
        """
        self.patience = patience
        self.change = change
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.mode = mode

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.change and self.mode == "minimize":
            self.counter += 1

            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score > self.best_score + self.change and self.mode == "maximize":
            self.counter += 1

            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0
