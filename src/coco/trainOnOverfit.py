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
import torch
from src.models.modnet import MODNet
from src.trainer import supervised_training_iter

from src.coco.SegmentationDatasetPytorch import SegmentationDataset

bs = 16         # batch size
lr = 0.01       # learn rate
epochs = 40     # total epochs
modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=False)).cuda()
optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

dataset = SegmentationDataset("/data/human","/data/human")
dataloader = DataLoader(dataset)     # NOTE: please finish this function

for epoch in range(0, epochs):
  for idx, (image, trimap, gt_matte) in enumerate(dataloader):
    print(image.shape,trimap.shape,gt_matte.shape)
    semantic_loss, detail_loss, matte_loss = supervised_training_iter(modnet, optimizer, image, trimap, gt_matte)
  lr_scheduler.step()