from torch.utils.data import Dataset
from typing import Callable, List, Optional, Tuple, Union, Dict
from pathlib import Path
import random
import torch
import os
import cv2
import glob
import numpy as np
from torch.utils.data import DataLoader
import torchvision

from src.coco.trimap import makeTrimap

class HumanSegmentationDataset(Dataset):
    """A custom Dataset(torch.utils.data) implement three functions: __init__, __len__, and __getitem__.
    Datasets are created from PTFDataModule.
    """

    def __init__(
            self,
            dataset_dir: Union[str, Path],
            human_size: int,
            non_human_size: int
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.human_size = human_size
        self.non_human_size = non_human_size
        self.indices = list(range(human_size + non_human_size))
        random.shuffle(self.indices)
        self.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        datasetIndex = self.indices[index]
        if datasetIndex >= self.human_size:
            datasetIndex -= self.human_size

            frame_pth = f"{self.dataset_dir}/nonHuman/image{datasetIndex}.jpg"
            mask_pth = f"{self.dataset_dir}/nonHuman/mask{datasetIndex}.png"
        else:
            frame_pth = f"{self.dataset_dir}/human/image{datasetIndex}.jpg"
            mask_pth = f"{self.dataset_dir}/human/mask{datasetIndex}.png"

        frame =  cv2.imread(frame_pth)
        frame = self.transform(frame).cuda()

        mask =  cv2.imread(mask_pth,cv2.IMREAD_GRAYSCALE)
        trimap = torch.from_numpy(makeTrimap(mask)).float().cuda()
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask,0).float().cuda()

        return frame, trimap, mask

    def __len__(self):
        return self.human_size + self.non_human_size
