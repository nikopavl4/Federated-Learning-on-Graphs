from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import torch
torch.manual_seed(12345)

def split_communities(data):
    G = to_networkx(data, to_undirected=True, node_attrs=['x','y'])
    communities = sorted(nx.community.asyn_fluidc(G, 3, max_iter = 5000, seed= 12345))

    node_groups = []
    for com in communities:
        node_groups.append(list(com))   

    C1 = G.subgraph(node_groups[0]).copy()
    C2 = G.subgraph(node_groups[1]).copy()
    C3 = G.subgraph(node_groups[2]).copy()

    return C1, C2, C3

def find_inter_cluster_edges(G,cluster1,cluster2,cluster3):
    trusted_nodes = []
    H = cluster1.copy()
    for node in list(cluster2.nodes(data=True)):
        for neighbor in G.neighbors(node[0]):
            if cluster1.has_node(neighbor) and (not H.has_node(node[0])):
                H.add_node(node[0], x = node[1]['x'], y = node[1]['y'])
                H.add_edge(node[0], neighbor)
                trusted_nodes.append(node[0])

    for node in list(cluster3.nodes(data=True)):
        for neighbor in G.neighbors(node[0]):
            if cluster1.has_node(neighbor) and (not H.has_node(node[0])):
                H.add_node(node[0], x = node[1]['x'], y = node[1]['y'])
                H.add_edge(node[0],neighbor)
                trusted_nodes.append(node[0])
    return H, trusted_nodes
        
def add_one_hop_neighbors(data,C1,C2,C3):
    G = to_networkx(data, to_undirected=True, node_attrs=['x','y'])
    H1, trusted_nodes1 = find_inter_cluster_edges(G, C1, C2, C3)
    H2, trusted_nodes2 = find_inter_cluster_edges(G, C2, C1, C3)
    H3, trusted_nodes3 = find_inter_cluster_edges(G, C3, C1, C2)

    data1 = from_networkx(H1)
    data2 = from_networkx(H2)
    data3 = from_networkx(H3)
    
    trusted_nodes = [trusted_nodes1, trusted_nodes2, trusted_nodes3]

    return data1, data2, data3, trusted_nodes

def train_test_split(data):
    mask = torch.randn((data1.num_nodes)) < 0.7
    nmask = torch.logical_not(mask)
    mask[data1.num_nodes - len(trusted_nodes1):] = False
    nmask[data1.num_nodes - len(trusted_nodes1):] = False
    train_mask_1 = mask
    test_mask_1 = nmask
