from torch.utils.data import Dataset
from typing import Callable, List, Optional, Tuple, Union, Dict
from pathlib import Path
import random
import re
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
    ) -> None:
        self.dataset_dir = Path(dataset_dir)

        human_images = glob.glob(f"{self.dataset_dir}/human/image*.jpg")
        human_masks = glob.glob(f"{self.dataset_dir}/human/mask*.png")
        human_images, human_masks = self.filterNames(human_images, human_masks)

        non_human_images = glob.glob(f"{self.dataset_dir}/nonHuman/image*.jpg")
        non_human_masks = glob.glob(f"{self.dataset_dir}/nonHuman/mask*.png")
        non_human_images, non_human_masks = self.filterNames(non_human_images, non_human_masks)
        print(f"human_images {len(human_images)}, non_human_images {len(non_human_images)}")
        print(f"human_masks {len(human_masks)}, non_human_masks {len(non_human_masks)}")
        self.image_names = human_images + non_human_images
        self.mask_names = human_masks + non_human_masks

        self.dataset_size = min(len(self.image_names), len(self.mask_names))
        print(f"dataset_size {self.dataset_size}")
        self.indices = list(range(self.dataset_size))
        random.shuffle(self.indices)

        self.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def nameIndex(self, name: str):
        match = re.search("(\\d+)", name)
        return int(match.group(0))

    def filterNames(self, imageNames, maskNames):
        idToImage = {
            self.nameIndex(name): name for name in imageNames
        }

        idToMask = {
            self.nameIndex(name): name for name in maskNames
        }

        commonIndices = list(set(idToImage.keys()) & set(idToMask.keys()))

        imageNames = [idToImage[index] for index in commonIndices]
        maskNames = [idToMask[index] for index in commonIndices]

        return imageNames, maskNames

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        datasetIndex = self.indices[index]

        frame_pth = self.image_names[datasetIndex]
        mask_pth = self.mask_names[datasetIndex]

        frame =  cv2.imread(frame_pth)
        frame = self.transform(frame).cuda()

        mask =  cv2.imread(mask_pth,cv2.IMREAD_GRAYSCALE)
        trimap = torch.from_numpy(makeTrimap(mask)).float().cuda()
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask,0).float().cuda()

        return frame, trimap, mask

    def __len__(self):
        return self.dataset_size
