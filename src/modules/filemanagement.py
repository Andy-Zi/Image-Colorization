from functools import partial
import os
import random
from typing import Tuple
from tqdm import tqdm
from PIL import Image
import numpy as np

from multiprocessing import Pool, cpu_count

def makeFileStructure(dataDir : str, *extradir : str) -> None:
    """
    This function creates a directory if it does not exist.
    """
    make_dir = lambda path: os.makedirs(path, exist_ok=True)
    
    make_dir(dataDir)
    make_dir(dataDir + "/train")
    make_dir(dataDir + "/test")
    
    for dir in extradir:
        make_dir(dir)
     
def TrainTestSplit(inputDir : str, extentions : list, dataDir : str, testRatio : float, shuffle : bool = True, numWorkers : int = cpu_count()/2) -> Tuple[int, int, int]:
    """
    This function splits the data in the input directory into train and test data.
    The test data is a random sample of the data.
    Returns: Trainsize, Testsize, UnusedFiles
    """
    input_files = []
    for root, dirs, files in os.walk(inputDir):
        for file in files:
            input_files.append(os.path.join(root, file))
    print("Found {} files in {}".format(len(input_files), inputDir))
    
    if shuffle: random.shuffle(input_files)

    pool = Pool(numWorkers)
    func = partial(SplitData, extentions=extentions, dataDir=dataDir, testRatio=testRatio)
    for _ in tqdm(pool.imap_unordered(func, input_files), total=len(input_files)):
        pass
    pool.close()
    pool.join()   
      
    return len(os.listdir(dataDir + "/train")), len(os.listdir(dataDir + "/test")), len(os.listdir(inputDir))

def SplitData(file : str, extentions : list, dataDir : str, testRatio : float) -> None:
    """
    This function splits the data in the input directory into train and test data.
    """
    if not any([file.endswith(extention) for extention in extentions]):
        return
    if not np.asarray(Image.open(file)).ndim == 3:
        return
    if not np.asarray(Image.open(file)).shape[2] == 3:
        return
    if random.random() < testRatio:
        os.rename(file, dataDir + "/test/" + file.split("\\")[-1])
    else:
        os.rename(file, dataDir + "/train/" + file.split("\\")[-1])
    return
