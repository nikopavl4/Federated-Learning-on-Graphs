# Quick Start

```
python graph_classification.py  --model --dataset --split --batch_size --hidden_channels --learning_rate --epochs
```

## Parameters

| Parameter       | Type  | Description                         |
|-----------------|-------|-------------------------------------|
| model           | str   | GNN used for training               |
| dataset         | str   | Dataset used for training           |
| split           | float | test/train dataset split percentage |
| batch_size      | int   | input batch size for training       |
| hidden_channels | int   | size of GNN hidden layer            |
| learning_rate   | float | learning rate for training          |
| epochs          | int   | epochs for training                 |

## Experiments - Results

### ENZYMES

| Parameters                              | Accuracy   | F1 Score  | Precision | Recall     |
|-----------------------------------------|------------|-----------|-----------|------------|
| gcn enzymes 0.8 16 16 0.01 50 (default) | 0.3        | 0.2766    | 0.2881    | 0.2949     |
| gcn enzymes 0.8 16 32 0.01 50           | 0.3333     | 0.3126    | 0.3673    | 0.3276     |
| gcn enzymes 0.8 16 64 0.01 50           | 0.2667     | 0.2094    | 0.1808    | 0.275      |
| gcn enzymes 0.8 16 32 0.01 150          | 0.35       | 0.3032    | 0.4557    | 0.3557     |
| gcn enzymes 0.8 32 32 0.01 150          | 0.3583     | 0.3162    | 0.4629    | 0.345      |
| gcn enzymes 0.8 64 32 0.01 150          | 0.325      | 0.2708    | 0.2467    | 0.32       |
| gcn enzymes 0.8 32 32 0.001 150         | 0.3667     | 0.3331    | 0.3107    | 0.3688     |
| sage enzymes 0.8 16 16 0.01 50          | 0.3417     | 0.3048    | 0.3283    | 0.3318     |
| **sage enzymes 0.8 32 32 0.01 150**     | **0.3917** | **0.373** | **0.487** | **0.3911** |
| sage enzymes 0.8 32 32 0.001 150        | 0.3417     | 0.3028    | 0.2857    | 0.3472     |

### PROTEINS

| Parameters                                 | Accuracy   | F1 Score   | Precision  | Recall     |
|--------------------------------------------|------------|------------|------------|------------|
| gcn   proteins 0.8 16 16 0.01 50 (default) | 0.6816     | 0.6568     | 0.675      | 0.655      |
| gcn proteins 0.8 32 32 0.01 150            | 0.6502     | 0.6403     | 0.6403     | 0.6403     |
| gcn proteins 0.8 32 32 0.001 150           | 0.6726     | 0.6538     | 0.6624     | 0.6519     |
| sage proteins 0.8 16 16 0.01 50            | 0.6951     | 0.6607     | 0.7004     | 0.6604     |
| **sage proteins 0.8 32 32 0.01 150**       | **0.7085** | **0.6978** | **0.6999** | **0.6964** |
| sage proteins 0.8 32 32 0.001 150          | 0.6996     | 0.6707     | 0.7006     | 0.6689     |

### MUTAG

| Parameters                            | Accuracy   | F1 Score   | Precision  | Recall     |
|---------------------------------------|------------|------------|------------|------------|
| gcn mutag 0.8 16 16 0.01 50 (default) | 0.7632     | 0.6842     | 0.6916     | 0.6786     |
| gcn mutag 0.8 32 32 0.01 150          | 0.7105     | 0.614      | 0.6188     | 0.6107     |
| gcn mutag 0.8 32 32 0.001 150         | 0.7632     | 0.6842     | 0.6916     | 0.6786     |
| sage mutag 0.8 16 16 0.01 50          | 0.7632     | 0.659      | 0.6889     | 0.6464     |
| sage mutag 0.8 32 32 0.01 150         | 0.7368     | 0.6955     | 0.6875     | 0.725      |
| **sage mutag 0.8 32 32 0.001 150**    | **0.7632** | **0.6842** | **0.6916** | **0.6786** |



## Diagrams

#### Centralized_graph_gcn_enzymes_16_16_001_50

<img src="/result_images/Centralized_graph_classification_enzymes/centralized_graph_gcn_enzymes_1.png" height="350">

#### Centralized_graph_gcn_enzymes_32_32_001_150

<img src="/result_images/Centralized_graph_classification_enzymes/centralized_graph_gcn_enzymes_5.png" height="350">

#### Centralized_graph_sage_enzymes_32_32_0001_150

<img src="/result_images/Centralized_graph_classification_enzymes/centralized_graph_sage_enzymes_3.png" height="350">

