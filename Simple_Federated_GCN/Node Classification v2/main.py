### A federated learning Example-Setting for Node Classification using FHE to avoid data leakage between Clients
from data_preprocess import load_data, split_communities, create_clients
global_graph, num_of_features, num_of_classes,num_of_nodes = load_data()
G1, G2, G3 = split_communities(global_graph)

Client1, Client2, Client3 = create_clients(G1,G2,G3)

print(Client1.model)

from model import GCN
mymodel = GCN(3,16,4,5)
Client1.set_model(mymodel)
print(Client1.model)