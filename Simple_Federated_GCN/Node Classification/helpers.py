from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx

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
