from torch.utils.data import Dataset
from typing import Callable, List, Optional, Tuple, Union, Dict
from pathlib import Path
import torch
import os
import cv2
import glob
import numpy as np
from torch.utils.data import DataLoader
import torchvision

from src.coco.trimap import makeTrimap

class SegmentationDataset(Dataset):
    """A custom Dataset(torch.utils.data) implement three functions: __init__, __len__, and __getitem__.
    Datasets are created from PTFDataModule.
    """

    def __init__(
            self,
            frame_dir: Union[str, Path],
            mask_dir: Union[str, Path]
    ) -> None:

        self.frame_dir = Path(frame_dir)
        self.mask_dir = Path(mask_dir)
        self.image_names = glob.glob(f"{self.frame_dir}/*.jpg") 
        self.mask_names = glob.glob(f"{self.mask_dir}/*.png")  
        self.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_pth = self.image_names[index]
        mask_pth = self.mask_names[index]

        frame =  cv2.imread(frame_pth)
        frame = self.transform(frame).cuda()

        mask =  cv2.imread(mask_pth,cv2.IMREAD_GRAYSCALE)
        trimap = torch.from_numpy(makeTrimap(mask)).float().cuda()
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask,0).float().cuda()

        return frame, trimap, mask

    def __len__(self):
        return len(self.image_names)
