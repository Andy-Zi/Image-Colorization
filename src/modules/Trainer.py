from modules.losses import Losses
import torch
from modules.colorDistribution import colorDistribution

import modules.utils as utils

from models.eccv16 import eccv16
from models.alexnet import alexnet

from modules.trainloop import Trainhelper

class Trainer():
    def __init__(self,
                 CLASSIFICATION_BUCKET_SIZE,
                 DEVICE,
                 EPOCHS,
                 LEARNING_RATE,
                 DATA_DIR,
                 SAVE_DIR,
                 MODEL_PATH,
                 NUM_WORKERS,
                 SAVE_EPOCHS,
                 trainloader,
                 testloader,
                 models= ["eccv16","alexnet"],
                 losses = ['L1', 'L2', 'CE', 'wCE']) -> None:
        
        self.CLASSIFICATION_BUCKET_SIZE = CLASSIFICATION_BUCKET_SIZE
        self.DEVICE = DEVICE
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.losses = losses
        self.models = models
        self.trainloader = trainloader
        self.testloader = testloader
        self.MODEL_PATH = MODEL_PATH
        self.SAVE_EPOCHS = SAVE_EPOCHS
        
        self.trainsets = {}
        
        self.colorcount = colorDistribution(DATA_DIR,SAVE_DIR,220,NUM_WORKERS)
        self.trainhelper = Trainhelper(self.DEVICE, self.MODEL_PATH,self.CLASSIFICATION_BUCKET_SIZE)
    
    def initTrainsets(self):
        for loss_name in self.losses:
            loss = Losses()
            weights = torch.tensor(self.colorcount.getClassWeights(self.CLASSIFICATION_BUCKET_SIZE),dtype=torch.float)
            loss.setWeights(weights)
            loss.initLoss(loss_name)

            for model_name in self.models:
                if model_name == 'alexnet':
                    model = alexnet(loss.loss_name(),utils.get_bucket_amount(self.CLASSIFICATION_BUCKET_SIZE)).to(self.DEVICE)
                elif model_name == 'eccv16':
                    model = eccv16(loss.loss_name(),utils.get_bucket_amount(self.CLASSIFICATION_BUCKET_SIZE)).to(self.DEVICE)
                else:
                    raise ValueError(f"Model {model_name} not supported.")

                self.trainsets[f"{model_name}_{loss_name}"] = {"model":model, "loss":loss}
    
    def getModelNames(self):
        return self.trainsets.keys()
        
    def getModel(self,name):
        return self.trainsets[name]["model"]
      
    def trainModel(self,name):
        trainlossGraph, testlossGraph = self._train(self.trainsets[name]["model"],self.trainsets[name]["loss"],name)
        return trainlossGraph, testlossGraph
        
    def trainAll(self):
        lossgraphs = {}
        for key in self.trainsets:
            trainlossGraph, testlossGraph = self._train(self.trainsets[key]["model"],self.trainsets[key]["loss"],key)
            lossgraphs[key] = {"trainlossGraph":trainlossGraph, "testlossGraph":testlossGraph}
        return lossgraphs
                
    def _train(self,model,loss,modelname):
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr = self.LEARNING_RATE)

        # scaler
        scaler = torch.cuda.amp.GradScaler()

        # Training
        trainlossGraph, testlossGraph = self.trainhelper.trainloop(model, self.trainloader, self.testloader, self.EPOCHS, optimizer, scaler, loss, self.SAVE_EPOCHS,modelname)
        return trainlossGraph, testlossGraph