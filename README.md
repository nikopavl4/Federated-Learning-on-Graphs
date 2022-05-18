# Introduction
A collection of experiments/examples for Graph Neural Networks and Federated Learning using Python Libraries PyTorch, PyTorch Geometric and NetworkX. In node Classification v2, there is an attempt to develop a secure framework for performing Federated Learning in a Node Classification Setting with no overlaps between clients' nodes using Fully Homomorphic Encryption to protect feature vectors privacy. In current repository, we have developed the baseline of the aforementioned framework, while more specific alternations concerning privacy protection will be added later. More documentation can be found under each specific subfolder. 

## Machine Learning Tasks
We studied the following Graph Learning tasks on both centralized and federated setting with multiple variations:
- Graph Classification
- Node Classification

## Datasets
Datasets used:
- **Graph Classification**
| Dataset | # of Graphs | # of Classes | # of Features | Avg # of Nodes
| ------ | ------ | ------ | ------ | ------
| [ENZYMES](https://paperswithcode.com/dataset/enzymes) | 600 (6*100) | 6 | 3 | 32.6 |
| [PROTEINS](https://paperswithcode.com/dataset/proteins) | 600 (6*100) | 6 | 3 | 32.6 |
| [MUTAG](https://paperswithcode.com/dataset/mutag) | 600 (6*100) | 6 | 3 | 32.6 |


- Node Classification


![Centralized Results](/result_images/federated_experiment1.png)
