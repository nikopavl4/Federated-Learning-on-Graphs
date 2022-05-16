class Client:
    def __init__(self, id, A, x, y):
        self.id = id
        self.A = A
        self.x = x
        self.y = y
        self.model = None

    def set_model(self,model):
        self.model = model