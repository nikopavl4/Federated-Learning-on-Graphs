# Federated-Learning-on-Graphs
Codes about Graph Neural Networks and Federated Learning

## Simple Federated GCN - Graph Classification
A very first example of a federated setup using a Graph Convolution Network.
We use [ENZYMES dataset](https://paperswithcode.com/dataset/enzymes) that contains 600 graphs and 6 labels. Our task is Graph Classification.

**Centralized Experiment**

- Training Set: 500 graphs

- Testing Set: 100 graphs

Results after 70 epochs of training:

![Centralized Results](/result_images/centralized_experiment1.png)

**Federated Experiment**

- Clients: 3

- Training Set: 150 graphs/per client

- Testing Set: 150 graphs

Results after 10 federated rounds:

![Centralized Results](/Simple_Federated_GCN/result_images/federated_experiment1.png)
