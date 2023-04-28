import torch.nn as nn


class Losses():
    losses = [ "L1", "L2", "CE", "wCE"]
    loss_type = ""
    loss = None
    weights = None
    
    def getlosses(self):
        return self.losses
    
    def setWeights(self, weights):
        self.weights = weights
         
    def initLoss(self, loss_type : str):
        self.loss_type = loss_type
        
        if self.loss_type == "L1":
            self.loss = nn.L1Loss().cuda()
        elif self.loss_type == "L2":
            self.loss = nn.MSELoss().cuda()
        elif self.loss_type == "CE":
            self.loss = nn.CrossEntropyLoss().cuda()
        elif self.loss_type == "wCE":
            if self.weights is None:
                raise Exception("weights not set")
            self.loss = nn.CrossEntropyLoss(weight=self.weights).cuda()

    def loss_name(self):
        return self.loss_type
    
    def __call__(self, inputs, targets):
        return self.loss(inputs, targets)
