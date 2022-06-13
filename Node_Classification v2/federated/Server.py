from model import GCN
from random import sample

class Aggregation_Server:
    def __init__(self):
        self.model = None
        self.optimizer = None

    def initialize(self,num_of_features, num_of_classes, hidden_channels):
        self.model = GCN(nfeat=num_of_features, nhid=hidden_channels, nclass=num_of_classes, dropout=0.5)

    def perform_fed_avg(self, client_list, parameterC):
        print("Server Aggregation - IDs Participating:", end=" ")
        client_list2 = sample(client_list, parameterC)
        for n in client_list2:
            print(n.id, end=" ")
        print("\n")
        for param_tensor in client_list2[0].model.state_dict():
            avg = (sum(c.model.state_dict()[param_tensor] for c in client_list2))/len(client_list2)
            self.model.state_dict()[param_tensor].copy_(avg)
            for cl in client_list:
                cl.model.state_dict()[param_tensor].copy_(avg)
        return client_list