# Quick Start

```
python graph_classification.py  --model --dataset --split --batch_size --hidden_channels --learning_rate --epochs
```

## Parameters

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

## Diagrams
