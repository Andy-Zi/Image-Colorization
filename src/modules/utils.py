from PIL import Image
from skimage import color
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
import math
import matplotlib.pyplot as plt
import os

from models.eccv16_pretrained import eccv16_pretrained

def plotPretrainedImage(IMAGE,DEVICE,IMAGESIZE):
    """
    Plots the pretrained image.
    """
    # load colorizers
    colorizer_eccv16 = eccv16_pretrained().eval()
    colorizer_eccv16.to(DEVICE)

    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img(IMAGE)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=IMAGESIZE)
    tens_l_rs = tens_l_rs.to(DEVICE)

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

    plt.figure(figsize=(30,10))
    f_size = 32
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('Original',fontsize=f_size)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(img_bw)
    plt.title('Input',fontsize=f_size)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(out_img_eccv16)
    plt.title('Output (ECCV 16)',fontsize=f_size)
    plt.axis('off')

    plt.show()

def load_img(img_path : str) -> np.ndarray:
    """
    Loads an image.
    """
    out_np = np.asarray(Image.open(img_path))
    if(out_np.ndim==2):
        out_np = np.tile(out_np[:,:,None],3)
    return out_np

def resize_img(img : np.array, HW : Tuple[int,int] = (256,256), resample : int =3) -> np.array:
    """
    Resizes an image.
    """
    return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig : np.array, HW : Tuple[int,int] = (256,256), resample : int = 3) -> np.array:
    """
    Preprocesses an image.
    """
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:,:,0]
    img_l_rs = img_lab_rs[:,:,0]

    tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
    tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

    return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l : np.array, out_ab : np.array, mode : str = 'bilinear') -> np.array:
    """
    Postprocesses the output of the model.
    """
    # tens_orig_l     1 x 1 x H_orig x W_orig
    # out_ab         1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode=mode)
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    out_rgb_compose = out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0))
    return color.lab2rgb(out_rgb_compose)

def trainprogress(epoch : int, epochs : int, step : int, steps : int, TrainLoss : float, TestLoss : float, Steptime : float, Epochruntime : float,*,length : int = 45,fill : str = '█') -> None:
    """
    Prints the progress of the training.
    """
    print(f"Epoch: {epoch+1}/{epochs}",end="")
    filledLength = int(length * (step+1) // steps)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f"\t[{bar}]",end="")
    print(f" Step: {step+1}/{steps}",end="")
    print(f"\t Train Loss: {TrainLoss:.4f}",end="")
    print(f"\t Test L1-Loss: {TestLoss:.4f}",end="") if type(TestLoss) is not str else print(f"\t Test L1-Loss: {TestLoss}",end="")
    print(f"\t Step Time: {Steptime:.2f}s",end="")
    print(f"\t Epoch Time: {Epochruntime:.2f}s",end="\r")
    
def testprogrss(step : int, steps : int, *, length : int = 45, fill : str = '█') -> None:
    """
    Prints the progress of the testing.
    """
    print(f"testing...",end="")
    filledLength = int(length * (step+1) // steps)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f"\t[{bar}]",end="")
    print(f" Step: {step+1}/{steps}",end="\r")

def get_bucket(point : list, bucket_size : int) -> int:
    """
    Returns the bucket index of a point.
    """
    ABMAX = 110
    x_part = (point[0]+ABMAX)//bucket_size
    y_part = (point[1]+ABMAX)//bucket_size
    if point[0] == ABMAX:
        x_part -= 1
    if point[1] == ABMAX:
        y_part -= 1
       
    return (y_part*math.ceil((2*ABMAX)/bucket_size))+x_part
    
def get_bucket_torch(data : torch.Tensor, bucket_size : int) -> int:
    """
    Returns the bucket index of a point.
    """
    classes_per_row = math.ceil((2*110)/bucket_size)
    ABMAX = 110
    data[data==ABMAX] = ABMAX-1
    x_part = torch.div(data[:,0,:,:]+ABMAX, bucket_size, rounding_mode='floor')
    y_part = torch.div(data[:,1,:,:]+ABMAX, bucket_size, rounding_mode='floor')

    return ((y_part*classes_per_row)+x_part).to(torch.int64)

def get_bucket_amount(bucket_size):
    ABMAX = 110
    x = 110
    y = 110
    return get_bucket((x,y),bucket_size)+1

def plotWeights(weights : np.array, bucket_size : int) -> None:
    """
    Plots the weights of a model.
    """
    data =weights.reshape((int(220/bucket_size),-1))
    data = np.log(data)
    plt.title("Weights for each bucket")
    plt.imshow(data)
    
def abArrayToClasses(abArray : np.array, bucket_size : int) -> np.array:
    """
    Converts an ab array to a class array.
    """
    ABMAX = 110

    bucket_array = get_bucket_torch(abArray,bucket_size)
    
    bucket_array = F.one_hot(bucket_array.flatten(),num_classes=get_bucket_amount(bucket_size))
    
    return bucket_array.to(torch.float).reshape((abArray.shape[0],get_bucket_amount(bucket_size),abArray.shape[2],abArray.shape[3]))

def getColorFromBucket(bucket : int, bucket_size : int) -> np.array:
    """
    Returns the color of a bucket.
    """
    ABMAX = 110
    x = ((bucket%math.ceil((2*ABMAX)/bucket_size))*bucket_size+bucket_size/2)-ABMAX
    y = ((bucket//math.ceil((2*ABMAX)/bucket_size))*bucket_size+bucket_size/2)-ABMAX
    return np.array([x,y])

def classesImageToLAB(img, bucket_size):
    img = img.argmax(dim=1).expand(2,-1,-1)
    ABMAX = 110
    bucket_count_per_row = math.ceil((2*ABMAX)/bucket_size)**2
    img[0] = ((img[0]%bucket_count_per_row)*bucket_size+bucket_size/2)-ABMAX
    img[1] = ((torch.div(img[1], bucket_count_per_row, rounding_mode='floor'))*bucket_size+bucket_size/2)-ABMAX
    return img[None,:].to(torch.float32)

def batchClassesToLAB(batch, bucket_size):
    batch = batch.argmax(dim=1)[:,None].expand(-1,2,-1,-1)
    ABMAX = 110
    bucket_count_per_row = math.ceil((2*ABMAX)/bucket_size)**2
    for i in range(batch.shape[0]):
        batch[i,0] = ((batch[i,0]%bucket_count_per_row)*bucket_size+bucket_size/2)-ABMAX
        batch[i,1] = ((torch.div(batch[i,1], bucket_count_per_row, rounding_mode='floor'))*bucket_size+bucket_size/2)-ABMAX
    return batch.to(torch.float32)

def plotImagefromModel(IMAGE,DEVICE,IMAGESIZE,model,*,classes = False,bucket_size = 10):
    """
    Plots an image from a model.
    """
    img = load_img(IMAGE)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=IMAGESIZE)
    tens_l_rs = tens_l_rs.to(DEVICE)
    
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    model_out = model(tens_l_rs).cpu()
    if classes:
        model_out = classesImageToLAB(model_out,bucket_size)
    out_img = postprocess_tens(tens_l_orig, model_out)
    
    plt.figure(figsize=(15,15))

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(img_bw)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(out_img)
    plt.title('Output')
    plt.axis('off')
    
    plt.show()

def saveTrainValLoss(trainloss,valloss,epochs,modelname,*,outdir : str = './loss.csv'):
    """
    Saves the train and validation loss to a csv file.
    """
    filename_train = f"{modelname}_{epochs}_TrainLoss.csv"
    filename_val = f"{modelname}_{epochs}_ValLoss.csv"
    
    path = os.path.join(outdir, modelname)
    os.makedirs(path, exist_ok=True)    
    
    np.savetxt(os.path.join(path,filename_train),trainloss,delimiter=',')
    np.savetxt(os.path.join(path,filename_val),valloss,delimiter=',')