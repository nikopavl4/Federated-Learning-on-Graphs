# Introduction
A collection of experiments/examples for Graph Neural Networks and Federated Learning using Python Libraries PyTorch, PyTorch Geometric and NetworkX. In node Classification v2, there is an attempt to develop a secure framework for performing Federated Learning in a Node Classification Setting with no overlaps between clients' nodes using Fully Homomorphic Encryption to protect feature vectors privacy. In current repository, we have developed the baseline of the aforementioned framework, while more specific alternations concerning privacy protection will be added later. More documentation can be found under each specific subfolder. 

## Machine Learning Tasks
We studied the following Graph Learning tasks on both centralized and federated setting with multiple variations:
- Graph Classification
- Node Classification

## Datasets
Datasets used:
- **Graph Classification**

| DATASET  | # of Graphs | # of Classes | # of Features | Avg # of Nodes |
|----------|-------------|--------------|---------------|----------------|
| [ENZYMES](https://paperswithcode.com/dataset/enzymes)  | 600         | 6            | 3             | 32.6           |
| [PROTEINS](https://paperswithcode.com/dataset/proteins) | 1113        | 2            | 3             | 39.1           |
| [MUTAG](https://paperswithcode.com/dataset/mutag)    | 188         | 2            | 7             | 17.9           |



- **Node Classification**

| DATASET  | # of Graphs | # of Classes | # of Features | # of Nodes |
|----------|-------------|--------------|---------------|------------|
| [Cora](https://paperswithcode.com/dataset/cora)     | 1           | 7            | 1433          | 2708       |
| [CiteSeer](https://paperswithcode.com/dataset/citeseer) | 1           | 6            | 3703          | 3327       |
| [PubMed](https://paperswithcode.com/dataset/pubmed)   | 1           | 3            | 500           | 19717      |

