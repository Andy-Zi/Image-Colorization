import os
from typing import List
import numpy as np
from PIL import Image
from skimage import color
import torch
import torchdatasets as td
from typing import Tuple
from modules.utils import load_img

class ImageDataset(td.Dataset):
    def __init__(self, data_dir : str, extentions : List[str], img_size : Tuple[int,int] = (256,256)) -> None:
        super().__init__()
        self.data_dir=data_dir
        self.extentions=extentions
        self.file_list=[file for file in os.listdir(self.data_dir) if any([file.endswith(extention) for extention in self.extentions])]
        self.img_size=img_size
    
    def __getitem__(self, index : int) -> Tuple[torch.Tensor, torch.Tensor]:
        file = self.data_dir+'/'+self.file_list[index]
        img = load_img(file)
        img_rgb_rs = np.asarray(Image.fromarray(img).resize((self.img_size[1],self.img_size[0]), resample=3))
        
        img_lab = color.rgb2lab(img_rgb_rs)

        img_l = img_lab[:,:,0]
        img_ab = img_lab[:,:,1:]/2

        tens_l = torch.Tensor(img_l)[None,:,:]
        tens_ab = torch.Tensor(img_ab.transpose(2,0,1))

        return (tens_l, tens_ab)

    def __len__(self) -> int:
        return len(self.file_list)