import subprocess as sub
import os


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'main.py')

print("Starting Running Federated Experiments for Node Classification with 1-hop Neighborhood Knowledge")
dataset = "cora"
model = "gcn"
hidden_channels = "64"
learning_rate = "0.007927656394399478"
for i in range(2,5):
    for j in range(i, 1, -1):
        print('--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--learning_rate', f'{learning_rate}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--learning_rate', f'{learning_rate}'])





dataset = "cora"
model = "sage"
hidden_channels = "48"
learning_rate = "0.003242623803127977"
for i in range(2,5):
    for j in range(i, 1, -1):
        print('--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--learning_rate', f'{learning_rate}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--learning_rate', f'{learning_rate}'])



dataset = "pubmed"
model = "gcn"
hidden_channels = "16"
learning_rate = "0.006136818914240232"
for i in range(2,5):
    for j in range(i, 1, -1):
        print('--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--learning_rate', f'{learning_rate}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--learning_rate', f'{learning_rate}'])



dataset = "pubmed"
model = "sage"
hidden_channels = "48"
learning_rate = "0.005539564367148789"
for i in range(2,5):
    for j in range(i, 1, -1):
        print('--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--learning_rate', f'{learning_rate}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--learning_rate', f'{learning_rate}'])



dataset = "citeseer"
model = "gcn"
hidden_channels = "64"
learning_rate = "0.0023180472987803877"
for i in range(2,5):
    for j in range(i, 1, -1):
        print('--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--learning_rate', f'{learning_rate}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--learning_rate', f'{learning_rate}'])


dataset = "citeseer"
model = "sage"
hidden_channels = "32"
learning_rate = "0.00987146046345244"
for i in range(2,5):
    for j in range(i, 1, -1):
        print('--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--learning_rate', f'{learning_rate}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--clients', f'{i}', '--parameterC', f'{j}', '--hidden_channels', f'{hidden_channels}', '--learning_rate', f'{learning_rate}'])