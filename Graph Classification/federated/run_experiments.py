import subprocess as sub
import os


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'main.py')

print("Starting Running Federated Experiments for Graph Classification")
dataset = "enzymes"
model = "gcn"
batch_size = "16"
hidden_channels = "16"
learning_rate = "0.01"
for i in range(2,5):
    for j in range(i, 1, -1):
        print('--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--batch_size', f'{batch_size}', '--learning_rate', f'{learning_rate}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--batch_size', f'{batch_size}', '--learning_rate', f'{learning_rate}'])





dataset = "enzymes"
model = "sage"
batch_size = "16"
hidden_channels = "16"
learning_rate = "0.01"
for i in range(2,5):
    for j in range(i, 1, -1):
        print('--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--batch_size', f'{batch_size}', '--learning_rate', f'{learning_rate}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--batch_size', f'{batch_size}', '--learning_rate', f'{learning_rate}'])



dataset = "proteins"
model = "gcn"
batch_size = "16"
hidden_channels = "16"
learning_rate = "0.01"
for i in range(2,5):
    for j in range(i, 1, -1):
        print('--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--batch_size', f'{batch_size}', '--learning_rate', f'{learning_rate}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--batch_size', f'{batch_size}', '--learning_rate', f'{learning_rate}'])



dataset = "proteins"
model = "sage"
batch_size = "16"
hidden_channels = "16"
learning_rate = "0.01"
for i in range(2,5):
    for j in range(i, 1, -1):
        print('--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--batch_size', f'{batch_size}', '--learning_rate', f'{learning_rate}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--batch_size', f'{batch_size}', '--learning_rate', f'{learning_rate}'])



dataset = "mutag"
model = "gcn"
batch_size = "16"
hidden_channels = "16"
learning_rate = "0.01"
for i in range(2,5):
    for j in range(i, 1, -1):
        print('--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--batch_size', f'{batch_size}', '--learning_rate', f'{learning_rate}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--batch_size', f'{batch_size}', '--learning_rate', f'{learning_rate}'])


dataset = "mutag"
model = "sage"
batch_size = "16"
hidden_channels = "16"
learning_rate = "0.01"
for i in range(2,5):
    for j in range(i, 1, -1):
        print('--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--batch_size', f'{batch_size}', '--learning_rate', f'{learning_rate}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--batch_size', f'{batch_size}', '--learning_rate', f'{learning_rate}'])