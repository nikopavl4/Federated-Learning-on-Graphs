from helpers import trainer, tester


def perform_federated_round(server_model,model1, model2, model3, round_id,test_loader, train_loader1, train_loader2, train_loader3):

    for param_tensor in model1.state_dict():
        avg = (model1.state_dict()[param_tensor] + model2.state_dict()[param_tensor] + model3.state_dict()[param_tensor])/3
        server_model.state_dict()[param_tensor].copy_(avg)
        model1.state_dict()[param_tensor].copy_(avg)
        model2.state_dict()[param_tensor].copy_(avg)
        model3.state_dict()[param_tensor].copy_(avg)

    test_acc_server = tester(server_model,test_loader)
    print(f'##Round ID={round_id}')
    print(f'Test Acc: {test_acc_server:.4f}')

    for epoch in range(1, 70):
        #Client1
        trainer(model1,train_loader1)
        train_acc_1 = tester(model1,train_loader1)
        test_acc_1 = tester(model1,test_loader)
        #print("Client1:")
        #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc_1:.4f}, Test Acc: {test_acc_1:.4f}')
        
        #Client2
        trainer(model2,train_loader2)
        train_acc_2 = tester(model2,train_loader2)
        test_acc_2 = tester(model2,test_loader)
        #print("Client2:")
        #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc_2:.4f}, Test Acc: {test_acc_2:.4f}')

        #Client3
        trainer(model3,train_loader3)
        train_acc_3 = tester(model2,train_loader3)
        test_acc_3 = tester(model2,test_loader)
        #print("Client3:")
        #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc_3:.4f}, Test Acc: {test_acc_3:.4f}')
    return test_acc_server