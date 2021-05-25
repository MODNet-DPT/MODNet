import argparse
# import shutil
# import subprocess
import torch
import os
import logging
import logging.handlers
from torch.utils.data import DataLoader

from src.models.modnet import MODNet
from src.trainer import supervised_training_iter

from src.coco.HumanSegmentationDatasetPytorch import HumanSegmentationDataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
LOG_FILENAME = os.path.join("logs", "{}.log".format(__name__))
handler = logging.handlers.RotatingFileHandler(
    LOG_FILENAME, maxBytes=10*1024*1024, backupCount=10)
logger.addHandler(handler)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, help='path to dataset')
parser.add_argument('--models-path', type=str, help='path to save trained MODNet models')
args = parser.parse_args()


bs = 8          # batch size
lr = 0.01       # learn rate
epochs = 1000     # total epochs
modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=False)).cuda()
optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

dataset = HumanSegmentationDataset(args.dataset_path)
dataloader = DataLoader(dataset, batch_size=bs)
print(f'Dataset length: {dataset.__len__()}')

print(f'Train for {epochs} epochs')
for epoch in range(0, epochs):
  for idx, (image, trimap, gt_matte) in enumerate(dataloader):
    semantic_loss, detail_loss, matte_loss, semantic_iou = supervised_training_iter(modnet, optimizer, image, trimap, gt_matte, semantic_scale=1)
    if idx % 100 == 0:
      logger.info(f'idx: {idx}, semantic_loss: {semantic_loss:.5f}, detail_loss: {detail_loss:.5f}, matte_loss: {matte_loss:.5f}, semantic_iou: {semantic_iou:.5f}')
  logger.info(f'Epoch: {epoch}, semantic_loss: {semantic_loss:.5f}, detail_loss: {detail_loss:.5f}, matte_loss: {matte_loss:.5f}, semantic_iou: {semantic_iou:.5f}')
  logger.info("save model")
  torch.save(modnet.state_dict(), os.path.join(args.models_path, f"model_epoch{epoch}.ckpt"))
    # shutil.copy(args.model_path, "/root/Yandex.Disk/data/")
    # shutil.copy(LOG_FILENAME, "/root/Yandex.Disk/data/")
    # subprocess.run(["yandex-disk", "sync"])
    # subprocess.run(["yandex-disk", "status"])

  lr_scheduler.step()

torch.save(modnet.state_dict(), os.path.join(args.models_path, "model.ckpt"))