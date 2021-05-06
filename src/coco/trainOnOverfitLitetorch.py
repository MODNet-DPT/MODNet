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
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.models.modnet import MODNet
from src.trainer import supervised_training_iter, blurer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer

from src.coco.trimap import makeTrimap

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    return x

def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union

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

class LitModel(pl.LightningModule):
  def __init__(self,semantic_scale=1.0, detail_scale=10.0, matte_scale=1.0,learning_rate=0.01,batch_size = 16):
    super().__init__()
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.semantic_scale=semantic_scale
    self.detail_scale=detail_scale
    self.matte_scale=matte_scale
    self.module = MODNet(backbone_pretrained=False)

  def forward(self, x, inference):
    return self.module(x, inference)

  def shared_step(self, batch):
    image, trimap, gt_matte = batch
    pred_semantic, pred_detail, pred_matte = self(image, False)

    # calculate the boundary mask from the trimap
    boundaries = (trimap < 0.5) + (trimap > 0.5)

    # calculate the semantic loss
    gt_semantic = F.interpolate(gt_matte, scale_factor=1/16, mode='bilinear')
    gt_semantic = blurer(gt_semantic)
    semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
    semantic_loss = self.semantic_scale * semantic_loss

    # calculate the detail loss
    pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
    gt_detail = torch.where(boundaries, trimap, gt_matte)
    detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))
    detail_loss = self.detail_scale * detail_loss

    # calculate the matte loss
    pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
    matte_l1_loss = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_matte)
    matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
        + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_matte)
    matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
    matte_loss = self.matte_scale * matte_loss

    # calculate the final loss, backward the loss, and update the model 
    loss = semantic_loss + detail_loss + matte_loss
    
    iou_arr = []
    for i in range(image.shape[0]):
      metric = iou(pred_matte[i], gt_matte[i], threshold=0.5)
      iou_arr.append(metric)
    metrics = torch.Tensor(iou_arr).reshape(-1, 1)
    
    return loss, metrics

  def training_step(self, batch, batch_idx):
    loss,metrics = self.shared_step(batch)
    return {"loss": loss, "metrics": metrics}
    
  def validation_step(self, batch, batch_nb):
    loss, metrics = self.shared_step(batch=batch)
    return {"val_loss": loss, "val_metrics": metrics}
  
  def shared_epoch_end(self, outputs, *, prefix: str = ""):
    avg_losses = torch.stack([x[f"{prefix}loss"] for x in outputs]).mean().item()
    metrics = torch.cat([x[f"{prefix}metrics"] for x in outputs], dim=0)
    avg_metrics_all = metrics.mean(dim=0)
    avg_metric = avg_metrics_all.mean().item()
    
    return avg_losses, avg_metric, 
  
  def training_epoch_end(self, outputs):
        avg_losses, avg_metric = self.shared_epoch_end(
            outputs=outputs, prefix=""
        )

        self.log(
            name="train_loss",
            value=avg_losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            name="train_IOU",
            value=avg_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
  def validation_epoch_end(self, outputs):
    if self.trainer.running_sanity_check:
            return
    avg_losses, avg_metric = self.shared_epoch_end(
            outputs=outputs, prefix="val_"
        )

    self.log(
            name="val_loss",
            value=avg_losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
    self.log(
            name="val_IOU",
            value=avg_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )      

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