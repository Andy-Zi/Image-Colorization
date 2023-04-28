import torch
import torch.nn as nn
import time
import numpy as np
import modules.utils as utils
import os

class Trainhelper:
    def __init__(self,DEVICE,MODEL_PATH,CLASSIFICATION_BUCKET_SIZE) -> None:
        self.DEVICE = DEVICE
        self.MODEL_PATH = MODEL_PATH
        self.CLASSIFICATION_BUCKET_SIZE = CLASSIFICATION_BUCKET_SIZE

    def trainloop(self,model, train_loader, test_loader, EPOCHS, optimizer, scaler, criterion, SAVE_EPOCHS, modelname):
        trainlossGraph = []
        testlossGraph = []
        for epoch in range(EPOCHS):
            starttime = time.time()
            batchtrainloss = []
            logsteploss = []
            for trainenumerator, (grayscaleImage,abValues) in enumerate(train_loader):
                # inputs
                grayscaleImage = grayscaleImage.to(self.DEVICE)
                # targets
                abValues = abValues.to(self.DEVICE)
                # forward
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(grayscaleImage)
                    if criterion.loss_name() == "wCE" or criterion.loss_name() == "CE":
                        abValues = utils.abArrayToClasses(abValues,self.CLASSIFICATION_BUCKET_SIZE)
                    loss=criterion(outputs,abValues)
                    batchtrainloss.append(loss.item())
                    logsteploss.append(loss.item())
                # backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                utils.trainprogress(epoch = epoch,
                                    epochs = EPOCHS, 
                                    step = trainenumerator, 
                                    steps = len(train_loader), 
                                    TrainLoss = np.mean(batchtrainloss), 
                                    TestLoss = "tbd.", 
                                    Steptime = (time.time()-starttime)/(trainenumerator+1), 
                                    Epochruntime = time.time()-starttime)
                
            trainlossGraph.append(np.mean(batchtrainloss))
            del outputs
            torch.cuda.empty_cache()
            stoptime = time.time()
            
            batchtetsloss = self.eval(test_loader, model, criterion)
            
            testlossGraph.append(np.mean(batchtetsloss))
            utils.trainprogress(epoch = epoch, 
                                epochs = EPOCHS, 
                                step = trainenumerator, 
                                steps = len(train_loader), 
                                TrainLoss = np.mean(batchtrainloss), 
                                TestLoss = np.mean(batchtetsloss), 
                                Steptime = (stoptime-starttime)/(trainenumerator+1), 
                                Epochruntime = stoptime-starttime)
            print("")
            

            #Save the model checkpoints
            if (epoch+1) % SAVE_EPOCHS == 0:
                filename = f'{modelname}-{epoch+1}.ckpt'
                path = os.path.join(self.MODEL_PATH, modelname)
                os.makedirs(path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(path, filename))
                
        return trainlossGraph, testlossGraph
                
    def eval(self, test_loader, model, criterion):
        vallos = nn.L1Loss().cuda()
        batchtetsloss = []
        with torch.no_grad():
            for testenumerator, (grayscaleImage,abValues) in enumerate(test_loader):
                # inputs
                grayscaleImage = grayscaleImage.to(self.DEVICE)
                # targets
                abValues = abValues.to(self.DEVICE)
                # calulate loss
                with torch.cuda.amp.autocast():
                    outputs = model(grayscaleImage)
                    if criterion.loss_name() == "wCE" or criterion.loss_name() == "CE":
                        outputs = utils.batchClassesToLAB(outputs,self.CLASSIFICATION_BUCKET_SIZE)
                    loss=vallos(outputs,abValues)
                    batchtetsloss.append(loss.item())
                utils.testprogrss(step = testenumerator, steps = len(test_loader))
            del outputs
            torch.cuda.empty_cache()
        return batchtetsloss
    