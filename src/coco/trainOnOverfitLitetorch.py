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
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer

from src.coco.trimap import makeTrimap
from src.coco.ModelLiteTorch import LitModel

class SegDataset(Dataset):
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
        self.mask_names = [os.path.join(self.mask_dir,"mask"+x.split('/')[-1][:-4][5:]+".png") for x in self.image_names] 
        print(self.mask_names)
        self.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_pth = self.image_names[index]
        mask_pth = self.mask_names[index]

        frame =  cv2.imread(frame_pth)
        frame = self.transform(frame)

        mask =  cv2.imread(mask_pth,cv2.IMREAD_GRAYSCALE)
        trimap = torch.from_numpy(makeTrimap(mask)).float()
        trimap =  torch.unsqueeze(trimap,0)
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask,0).float()

        return frame, trimap, mask

    def __len__(self):
        return len(self.image_names)

dataset = SegDataset("data/images","data/masks")



  def train_dataloader(self):
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,

            pin_memory=True
        )
  def val_dataloader(self):
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )
  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5, T_mult=2, eta_min= 3e-7)
    return [optimizer], [scheduler]


bs = 16        # batch size
lr = 0.01       # learn rate
epochs = 40     # total epochs
checkpoint_callback = ModelCheckpoint(
        dirpath="top_models/new",
        filename="{epoch}-{val_loss:.3f}-{val_IOU:.4f}",
        save_top_k=5,
        verbose=True,
        monitor="val_IOU",
        mode="max"
    )
trainer = pl.Trainer(gpus = 1,checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=20)
model = LitModel(learning_rate = lr,batch_size = bs,)

trainer.fit(model)

torch.save(model.state_dict(), "data/modnetLitetorch.ckpt")