import os
import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

import modules.utils as utils
from models.eccv16_pretrained import eccv16_pretrained


def getTrainedData(Model_path, Loss_path):
    """
    Returns the files, in which trained data for the model and loss are stored.
    """
    models = []
    for root, dirs, files in os.walk(Model_path):
        if root != Model_path:
            models.append([root.split("\\")[-1],files])
    lossGraphs = []
    for root, dirs, files in os.walk(Loss_path):
        if root != Loss_path:
            lossGraphs.append([root.split("\\")[-1],files])
    return np.array(models,dtype=object), lossGraphs


def plotLoss(losses,SAVE_DIR):
    """
    Plots the loss of the models.
    """
    plt.figure(figsize=(20,15))
    for i,file in enumerate([5,1,6,2,4,0,7,3]):
        path = os.path.join(SAVE_DIR,losses[file][0])

        for j in range(2):
            data = np.loadtxt(os.path.join(path,losses[file][1][j]))
            plot = 1 if "TrainLoss.csv" in losses[i][1][j] else 2
            plt.subplot(len(losses)//2,4,i*2+plot)
            plt.title(f"{losses[file][0]} {'TrainLoss' if 'TrainLoss' in losses[file][1][j] else 'Test L1-Loss'}")
            plt.plot(data)
    plt.show()
    
def plotRegressionValLoss(SAVE_DIR,losses):
    """
    Plots the loss of the models.
    """
    for i,file in enumerate([5,1,6,2]):
        path = os.path.join(SAVE_DIR,losses[file][0])

        j = 1
        data = np.loadtxt(os.path.join(path,losses[file][1][j]))
        plt.plot(data,label=losses[file][0])
    plt.title("regression validation L1 loss")
    plt.legend()
    plt.show()

def plotClassificationValLoss(SAVE_DIR,losses):
    """
    Plots the loss of the models.
    """
    for i,file in enumerate([4,0,7,3]):
        path = os.path.join(SAVE_DIR,losses[file][0])

        j = 1
        data = np.loadtxt(os.path.join(path,losses[file][1][j]))
        plt.plot(data,label=losses[file][0])
    plt.title("classification validation L1 loss")
    plt.legend()
    plt.show()
    
def plotModelCheckpoints(IMAGE,IMAGESIZE,DEVICE,trainer,models,MODEL_PATH,CLASSIFICATION_BUCKET_SIZE):
    """
    Plots images for every checkpoint for every model.
    """
    img = utils.load_img(IMAGE)
    (tens_l_orig, tens_l_rs) = utils.preprocess_img(img, HW=IMAGESIZE)
    tens_l_rs = tens_l_rs.to(DEVICE)

    plt.figure(figsize=(20,40))
    for i,modelname in enumerate(trainer.trainsets.keys()):
        model = trainer.getModel(modelname)
        cpkts = models[np.where(models == modelname)[0]][0]
        for j,ckpt in enumerate(cpkts[1]):
            model.load_state_dict(torch.load(os.path.join(MODEL_PATH,cpkts[0],ckpt)))
            model_out = model(tens_l_rs).cpu()
            if "CE" in modelname:
                model_out = utils.classesImageToLAB(model_out,CLASSIFICATION_BUCKET_SIZE)
            out_img = utils.postprocess_tens(tens_l_orig, model_out)

            plt.subplot(len(trainer.trainsets.keys()),len(cpkts[1]),i*len(cpkts[1])+j+1)
            plt.imshow(out_img)
            plt.title(ckpt)
            plt.axis('off')
        if i == 3:
            plt.show()
            plt.figure(figsize=(20,40))
    plt.show()
    
def plotDifferentPicturesPerModel(testsize,IMAGESIZE,DEVICE,trainer,models,MODEL_PATH,CLASSIFICATION_BUCKET_SIZE):
    """
    Plots multiple images for every model.
    """
    subplots_x = testsize*2
    subplots_y = math.ceil((len(trainer.trainsets.keys()))/2)+1
    plots_per_img = len(trainer.trainsets.keys())+2
    plt.figure(figsize=(30,50))
    for i in range(testsize):
        IMAGE = f'./data/test/{random.choice(os.listdir("./data/test"))}'

        img = utils.load_img(IMAGE)
        (tens_l_orig, tens_l_rs) = utils.preprocess_img(img, HW=IMAGESIZE)
        tens_l_rs = tens_l_rs.to(DEVICE)
        
        plt.subplot(subplots_x,subplots_y,i*plots_per_img+1)
        plt.imshow(img)
        plt.title("Ground Truth")
        plt.axis('off')
        
        # load colorizers
        colorizer_eccv16 = eccv16_pretrained().eval()
        colorizer_eccv16.to(DEVICE)
        out_img_eccv16 = utils.postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
        plt.subplot(subplots_x,subplots_y,i*plots_per_img+1+subplots_y)
        plt.imshow(out_img_eccv16)
        plt.title("Original Paper")
        plt.axis('off')
            
        for j,modelname in enumerate(trainer.trainsets.keys()):
            model = trainer.getModel(modelname)
            cpkts = models[np.where(models == modelname)[0]][0]

            model.load_state_dict(torch.load(os.path.join(MODEL_PATH,cpkts[0],cpkts[1][-1])))
            model_out = model(tens_l_rs).cpu()
            if "CE" in modelname:
                model_out = utils.classesImageToLAB(model_out,CLASSIFICATION_BUCKET_SIZE)
            out_img = utils.postprocess_tens(tens_l_orig, model_out)

            plt.subplot(subplots_x,subplots_y,i*plots_per_img+j+2+(1 if j >= subplots_y-1 else 0))
            plt.imshow(out_img)
            plt.title(modelname)
            plt.axis('off')
    plt.show()