# Quick Start

```
python main.py  --model --dataset --split --clients --parameterC --hidden_channels --batch_size --epochs --federated_rounds
```

## Parameters

| Parameter       | Type  | Description                         |
|-----------------|-------|-------------------------------------|
| model           | str   | GNN used for training               |
| dataset         | str   | Dataset used for training           |
| split           | float | test/train dataset split percentage |
| clients           | int | number of clients |
| parameterC           | int | num of clients randomly selected to participate in Federated Learning |
| hidden_channels | int   | size of GNN hidden layer            |
| batch_size      | int   | input batch size for training       |
| epochs          | int   | epochs for training                 |
| federated_rounds   | int | federated rounds performed          |

## Experiments - Results

### ENZYMES

| Parameters                              | Accuracy   | F1 Score  | Precision | Recall     |
|-----------------------------------------|------------|-----------|-----------|------------|
| gcn enzymes 0.8 16 16 0.01 50 (default) | 0.3        | 0.2766    | 0.2881    | 0.2949     |


### PROTEINS

| Parameters                                 | Accuracy   | F1 Score   | Precision  | Recall     |
|--------------------------------------------|------------|------------|------------|------------|
| gcn   proteins 0.8 16 16 0.01 50 (default) | 0.6816     | 0.6568     | 0.675      | 0.655      |


### MUTAG

| Parameters                            | Accuracy   | F1 Score   | Precision  | Recall     |
|---------------------------------------|------------|------------|------------|------------|
| gcn mutag 0.8 16 16 0.01 50 (default) | 0.7632     | 0.6842     | 0.6916     | 0.6786     |




## Diagrams

#### Centralized_graph_gcn_enzymes_16_16_001_50

<img src="/result_images/Centralized_graph_classification_enzymes/centralized_graph_gcn_enzymes_1.png" height="350">

#### Centralized_graph_gcn_enzymes_32_32_001_150

<img src="/result_images/Centralized_graph_classification_enzymes/centralized_graph_gcn_enzymes_5.png" height="350">

#### Centralized_graph_sage_enzymes_32_32_0001_150

<img src="/result_images/Centralized_graph_classification_enzymes/centralized_graph_sage_enzymes_3.png" height="350">
