import torch

class Client:
    def __init__(self, id, dataset):
        self.id = id
        self.dataset = dataset
        self.trainloader = None
        self.testloader = None
        self.model = None
        self.optimizer = None

    def set_model(self,model, args):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    def set_trainloader(self, train_loader):
        self.trainloader = train_loader

    def set_testloader(self, test_loader):
        self.testloader = test_loader