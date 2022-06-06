from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
import networkx as nx


def load_data(args):
    #Import and Examine Dataset
    if args.dataset.lower() == 'pubmed':
        dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=T.Compose([T.LargestConnectedComponents(),T.SVDFeatureReduction(20)]))
    elif args.dataset.lower() == 'cora':
        dataset = Planetoid(root='data/Planetoid', name='Cora',  transform=T.Compose([T.LargestConnectedComponents(),T.SVDFeatureReduction(20)]))
    elif args.dataset.lower() == 'citeseer':
        dataset = Planetoid(root='data/Planetoid', name='CiteSeer',transform=T.LargestConnectedComponents())
    else:
        print("No such dataset!")
        exit()

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
    G = to_networkx(data, to_undirected=True, node_attrs=['x','y'])
    return G, dataset.num_features, dataset.num_classes

def split_communities(G, clients):
    communities = sorted(nx.community.asyn_fluidc(G, clients, max_iter = 5000, seed= 12345))
    comm = []
    node_groups = []
    for com in communities:
        node_groups.append(list(com))   

    for k in range(len(node_groups)):
        C = G.subgraph(node_groups[k]).copy()
        comm.append(C)

    return comm

from Client import Client
def create_clients(comm):
    client_list = []
    for z in range(len(comm)):
        Cl = Client(z+1,nx.to_numpy_matrix(comm[z]),nx.get_node_attributes(comm[z], "x"),nx.get_node_attributes(comm[z], "y"))
        client_list.append(Cl)
    print(f'Created {len(client_list)} clients!')
    return client_list