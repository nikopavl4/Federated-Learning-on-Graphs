import torch
import numpy as np
import torch.nn.functional as F

class Client:
    def __init__(self, id, dataset, dataloader):
        self.id = id
        self.dataset = dataset
        self.dataloader = dataloader
        self.model = None
        self.optimizer = None

    def set_model(self,model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)