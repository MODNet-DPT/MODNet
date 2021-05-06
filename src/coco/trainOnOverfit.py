import torch
from torch.utils.data import DataLoader

from src.models.modnet import MODNet
from src.trainer import supervised_training_iter

from src.coco.SegmentationDatasetPytorch import SegmentationDataset

bs = 16         # batch size
lr = 0.01       # learn rate
epochs = 1000     # total epochs
modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=False)).cuda()
optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

dataset = SegmentationDataset("data/human","data/human")
dataloader = DataLoader(dataset)     # NOTE: please finish this function
print(f'Dataset length: {dataset.__len__()}')

print(f'Train for {epochs} epochs')
for epoch in range(0, epochs):
  for idx, (image, trimap, gt_matte) in enumerate(dataloader):
    semantic_loss, detail_loss, matte_loss, semantic_iou = supervised_training_iter(modnet, optimizer, image, trimap, gt_matte,semantic_scale =1)
  print(f'Epoch: {epoch}, semantic_loss: {semantic_loss:.5f}, semantic_iou: {semantic_iou:.5f}')
  lr_scheduler.step()