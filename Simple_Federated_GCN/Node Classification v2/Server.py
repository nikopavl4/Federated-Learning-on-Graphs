class Aggregation_Server:
    def __init__(self):
        self.model = None
        self.optimizer = None

    def set_model(self,model):
        self.model = model

    def perform_fed_avg(self, model1, model2, model3):
        print("Server Aggregation")
        for param_tensor in model1.state_dict():
            avg = (model1.state_dict()[param_tensor] + model2.state_dict()[param_tensor] + model3.state_dict()[param_tensor])/3
            self.model.state_dict()[param_tensor].copy_(avg)
            model1.state_dict()[param_tensor].copy_(avg)
            model2.state_dict()[param_tensor].copy_(avg)
            model3.state_dict()[param_tensor].copy_(avg)
        return model1, model2, model3