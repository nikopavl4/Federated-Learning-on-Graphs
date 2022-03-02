from model import GCN
import torch
from helpers import trainer, tester
import matplotlib.pyplot as plt

def run_centralized_node_classifcation(dataset, data):
    model = GCN(hidden_channels=16, dataset=dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    val_losses = []
    train_losses = []
    print("+++++++++++++ Centralized Node Classification +++++++++++++")
    for epoch in range(1, 101):
        loss, train_acc = trainer(model, optimizer, criterion, data)
        test_acc = tester(model, data) 
        train_losses.append(train_acc)
        val_losses.append(test_acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    
    plt.figure(figsize=(10,5))
    plt.title("Training and Testing Accuracy")
    plt.plot(val_losses,label="Test")
    plt.plot(train_losses,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def simple_train_test_split(data):
    mask = torch.randn((data.num_nodes)) < 0.7
    nmask = torch.logical_not(mask)
    data.train_mask = mask
    data.test_mask = nmask
    return data