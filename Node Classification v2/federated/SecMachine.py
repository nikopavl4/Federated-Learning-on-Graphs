import numpy as np
import torch

def update_adjacency_matrix(A, counter):
    n = len(A)
    column = np.zeros(n)
    column[counter] = 1
    A = np.hstack((A, np.atleast_2d(column).T))
    row = np.zeros(n+1)
    row[counter] = 1
    A= np.vstack ((A, row) )
    return A

def update_features(x, x_to_add):
    maximum_key = max(x.keys())
    x[maximum_key+1] = x_to_add
    return x

class SecMachine:
    def __init__(self,A):
        self.global_map = A
        self.temp = None
        self.maps = {}
        self.features = {}
        self.node_keys = {}

    def add_secure_client(self,Client):
        self.maps[Client.id] = Client.A
        self.features[Client.id]= Client.x
        self.node_keys[Client.id] = list(Client.x.keys())

    def find_connections(self, id1, id2):
        nodes1 = self.node_keys[id1]
        nodes2 = self.node_keys[id2]
        counter = 0
        for node in nodes1:
            neighborhood = np.nonzero(self.global_map[node][0])[1]
            for j in neighborhood:
                if j in nodes2:
                    self.maps[id1] = update_adjacency_matrix(self.maps[id1], counter)
                    self.features[id1] = update_features(self.features[id1], self.features[id2][j] )
            counter = counter +1

    def compute_safe_convolution(self, id, weight, label):
        
        if label == 1:
            support = torch.mm(torch.tensor(list(self.features[id].values())), weight)   
            output = torch.mm(torch.tensor(self.maps[id].A.astype(np.float32)), support)
            # true_nodes = len(self.node_keys[id])
            # output = output[0:true_nodes]
            self.temp = output
        elif label == 2:
            support = torch.mm(self.temp, weight)   
            output = torch.mm(torch.tensor(self.maps[id].A.astype(np.float32)), support)
            true_nodes = len(self.node_keys[id])
            output = output[0:true_nodes]
        
        return output

