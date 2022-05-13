import torch
torch.manual_seed(12345)


def trainer(model,optimizer,criterion, data):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    train_correct = pred[data.train_mask] == data.y[data.train_mask]  # Check against ground-truth labels.
    train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  # Derive ratio of correct predictions.
    return loss, train_acc

def tester(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

def simple_train_test_split(data, split):
    mask = torch.randn((data.num_nodes)) < split
    nmask = torch.logical_not(mask)
    data.train_mask = mask
    data.test_mask = nmask
    return data


