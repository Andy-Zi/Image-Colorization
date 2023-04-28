import os
import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
from PIL import Image
from skimage import color
from skimage.color import lab2rgb
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules.utils import get_bucket,get_bucket_amount

ABRANGE = 110
MAXL = 100
    
class colorDistribution:
    
    def __init__(self,DataDir : str , outdir : str ,steps : int, numWorkers : int = cpu_count()/2) -> None:
        self.steps = steps
        self.DataDir = DataDir
        self.numWorkers = numWorkers
        self.outdir = outdir
        self.lab_array = np.zeros((MAXL,self.steps,self.steps))
    
    def read_lab_array(self,*,nrFiles : int = 0) -> None:
        """
        Reads the lab_array from a file
        """
        if os.path.exists(self.outdir+'/lab_array.npy'):
            self.lab_array = np.load(self.outdir+'/lab_array.npy')
        else:
            self.__colorDistribution(nrFiles=nrFiles)
            np.save(self.outdir+'/lab_array.npy',self.lab_array)
    
    def get_lab_array(self,*,lmin : int = 0, lmax : int = 100) -> np.ndarray:
        """
        Returns the lab_array with the given lmin and lmax
        """
        out = np.sum(self.lab_array[lmin:lmax],axis=0)
        return np.where(out>0,np.log(np.where(out>0,out,1)),-np.inf)
    
    def plot_colorDistribution(self) -> None:
        """
        Plots the color distribution of the images in the DataDir
        """
        XYMIN = 0
        XYMAX = 220
        STEPS = 5
        ABMIN = -110
        ABMAX = 110

        
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(16, 16)
        fig.patch.set_facecolor('xkcd:white')
        fig.suptitle('Color Distribution', fontsize=40)
        subfigs = fig.subfigures(2, 2)
        
        # Plot the Color Distribution
        ax = subfigs[0][0].add_axes([0.1, 0.1, 0.9, 0.9])
        ax.set_title('L [0,100]', fontsize=20)
        ax.imshow(self.get_lab_array(), cmap='jet')
        ax.grid(True)

        ax.set_xlabel("b", fontsize=20)
        ax.set_xticks(np.linspace(XYMIN, XYMAX, STEPS))
        ax.set_xticklabels(np.linspace(ABMIN, ABMAX, STEPS), fontsize=15)

        ax.set_ylabel("a", fontsize=20, rotation=0)
        ax.set_yticks(np.linspace(XYMIN, XYMAX, STEPS))
        ax.set_yticklabels(np.linspace(ABMIN, ABMAX, STEPS), fontsize=15)


        steps = [[0,25],[25,50],[50,75],[75,100]]
        axs = subfigs[0][1].subplots(2, 2)
        for i, ax in enumerate(axs.flat):
            ax.set_title(f'L {steps[i]}', fontsize=20)
            ax.imshow(self.get_lab_array(lmin = steps[i][0], lmax = steps[i][1]), cmap='jet')
            ax.grid(True)
            
            ax.set_xlabel("b", fontsize=15)
            ax.set_xticks(np.linspace(XYMIN, XYMAX, STEPS))
            ax.set_xticklabels(np.linspace(ABMIN, ABMAX, STEPS), fontsize=10)

            ax.set_ylabel("a", fontsize=15, rotation=0)
            ax.set_yticks(np.linspace(XYMIN, XYMAX, STEPS))
            ax.set_yticklabels(np.linspace(ABMIN, ABMAX, STEPS), fontsize=10)

        # Plot the LAB Colorspace
        ax = subfigs[1][0].add_axes([0.1, 0.1, 0.9, 0.9])
        ax.set_title('Lab Colorspace for L = 50', fontsize=20)
        l = 50
        LAB_Colorspace = np.array([[[l,float(a),float(b)] for b in range(ABMIN, ABMAX)] for a in range(ABMIN, ABMAX)])
        LAB_Colorspace = lab2rgb(LAB_Colorspace)
        ax.imshow(LAB_Colorspace)
        ax.grid(True)

        ax.set_xlabel("b", fontsize=20)
        ax.set_xticks(np.linspace(XYMIN, XYMAX, STEPS))
        ax.set_xticklabels(np.linspace(ABMIN, ABMAX, STEPS), fontsize=15)

        ax.set_ylabel("a", fontsize=20, rotation=0)
        ax.set_yticks(np.linspace(XYMIN, XYMAX, STEPS))
        ax.set_yticklabels(np.linspace(ABMIN, ABMAX, STEPS), fontsize=15)
        
        # Plot the intersection of the LAB Colorspace and the Color Distribution
        ax = subfigs[1][1].add_axes([0.1, 0.1, 0.9, 0.9])
        ax.set_title('Used colors', fontsize=20)
        used_colors = np.expand_dims(self.get_lab_array(), axis=2)

        ax.imshow(np.where(used_colors!=-np.inf,LAB_Colorspace,[1,1,1]))
        ax.grid(True)

        ax.set_xlabel("b", fontsize=20)
        ax.set_xticks(np.linspace(XYMIN, XYMAX, STEPS))
        ax.set_xticklabels(np.linspace(ABMIN, ABMAX, STEPS), fontsize=15)

        ax.set_ylabel("a", fontsize=20, rotation=0)
        ax.set_yticks(np.linspace(XYMIN, XYMAX, STEPS))
        ax.set_yticklabels(np.linspace(ABMIN, ABMAX, STEPS), fontsize=15)
        
        plt.show()

    def getClassDistribution(self,bucket_size : int) -> np.ndarray:
        """
        Returns the class distribution of the images in the DataDir
        """
        ABMAX = 110
        classes = {key: 0 for key in range(get_bucket_amount(bucket_size))}
        ab_count = self.lab_array.sum(axis=0)
        for y in range(ab_count.shape[0]):
            for x in range(ab_count.shape[1]):
                Point = [x-ABMAX,y-ABMAX]
                classes[get_bucket(Point,bucket_size)] += ab_count[x,y]
        return classes
    
    def getClassWeights(self,bucket_size : int) -> np.ndarray:
        """
        Returns the class weights of the images in the DataDir
        """
        classDistribution = np.array(list(self.getClassDistribution(bucket_size).values()))
        classDistribution = np.where(classDistribution==0,1,classDistribution)
        weights = 1. / classDistribution
        return weights
            
      
    def __colorDistribution(self ,*,nrFiles : int = 0) -> None:
        """
        Calculates the color distribution of the images in the DataDir
        """

        input_files = self.__get_input_Files()      
        if nrFiles:
            input_files = input_files[:nrFiles] 
        print(f"Found {len(input_files)} files in {self.DataDir}")
        
        func = partial(calcABdistribution, steps=self.steps)
        
        if self.numWorkers > 1:
            pool = Pool(self.numWorkers)
            for lab in tqdm(pool.imap_unordered(func, input_files), total=len(input_files)):
                self.lab_array = np.add(self.lab_array,lab)

            pool.close()
            pool.join()
        else:
            for file in tqdm(input_files):
                self.lab_array = np.add(self.lab_array,func(file))

    def __get_input_Files(self) -> List[str]:
        """
        Returns a list of all files in the DataDir
        """
        input_files = []
        for root, dirs, files in os.walk(self.DataDir):
            for file in files:
                input_files.append(os.path.join(root, file))
        return input_files

def calcABdistribution(file : str, steps : int) -> np.ndarray:
    """
    Calculates the color distribution of the image in the file
    """
    lab = np.zeros((MAXL,steps,steps))
    img_rgb = np.asarray(Image.open(file))
    if(img_rgb.ndim==2):
        img_rgb = np.tile(img_rgb[:,:,None],3)
    img_rgb = np.asarray(Image.fromarray(img_rgb).resize((224,224), resample=3))
    img_lab = color.rgb2lab(img_rgb)
    for l,a,b in img_lab.reshape(-1,3):
        a = int(np.round((a+ABRANGE)/(2*ABRANGE)*steps))
        b = int(np.round((b+ABRANGE)/(2*ABRANGE)*steps))
        lab[int(np.round(l))-1,a,b] += 1
        
    return lab