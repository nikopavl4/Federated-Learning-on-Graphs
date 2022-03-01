import torch
from random import seed
from torch_geometric.datasets import KarateClub,BAShapes
import networkx as nx
import matplotlib.pyplot as plt
torch.manual_seed(12345)

#Import and Examine Dataset
dataset = BAShapes(connection_distribution='uniform')

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

#Split Graph and creating client datasets
from helpers import split_communities
C1, C2, C3 = split_communities(data)