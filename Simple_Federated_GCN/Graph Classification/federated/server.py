from helpers import tester
import random
from random import sample
random.seed(12345)

def perform_federated_round(server_model,Client_list,round_id,test_loader, args):
    #Client_list = sample(Client_list, args.parameterC)
    for param_tensor in Client_list[0].model.state_dict():
        avg = (sum(c.model.state_dict()[param_tensor] for c in Client_list))/len(Client_list)
        server_model.state_dict()[param_tensor].copy_(avg)
        for cl in Client_list:
            cl.model.state_dict()[param_tensor].copy_(avg)

    test_acc_server = tester(server_model,test_loader)
    print(f'##Round ID={round_id}')
    print(f'Test Acc: {test_acc_server:.4f}')

    return test_acc_server