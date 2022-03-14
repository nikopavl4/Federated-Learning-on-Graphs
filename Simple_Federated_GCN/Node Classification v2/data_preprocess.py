from ast import Return
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import networkx as nx

def load_data():
    #Import and Examine Dataset
    dataset = KarateClub()

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the graph object.

    print(data)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    return to_networkx(data, to_undirected=True, node_attrs=['x','y'])

def split_communities(G):
    communities = sorted(nx.community.asyn_fluidc(G, 3, max_iter = 5000, seed= 12345))

    node_groups = []
    for com in communities:
        node_groups.append(list(com))   

    C1 = G.subgraph(node_groups[0]).copy()
    C2 = G.subgraph(node_groups[1]).copy()
    C3 = G.subgraph(node_groups[2]).copy()

    return C1, C2, C3

from Client import Client
def create_clients(G1,G2,G3):
    Client1 = Client(1,nx.adjacency_matrix(G1),nx.get_node_attributes(G1, "x"),nx.get_node_attributes(G1, "y"))
    Client2 = Client(2,nx.adjacency_matrix(G2),nx.get_node_attributes(G2, "x"),nx.get_node_attributes(G2, "y"))
    Client3 = Client(3,nx.adjacency_matrix(G3),nx.get_node_attributes(G3, "x"),nx.get_node_attributes(G3, "y"))
    return Client1, Client2, Client3