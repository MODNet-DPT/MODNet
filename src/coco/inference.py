import cv2
import argparse
import torch
import torchvision
import numpy as np
import cv2

from src.models.modnet import MODNet

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', type=str, help='path of input images')
parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet')
args = parser.parse_args()

modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=False))

modnet.load_state_dict(
    torch.load(args.ckpt_path, map_location=torch.device('cpu'))
)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = cv2.imread(args.input_path)
imageTensor = transform(image)
imageTensor = imageTensor.unsqueeze(0)

_, _, mask = modnet(imageTensor, True)

mask = mask.squeeze().cpu().detach().numpy()

cv2.imshow("image", image)
cv2.imshow("mask", mask)
imageMask = cv2.merge([np.ones_like(mask) / 2, np.ones_like(mask) / 2, (mask + 1) / 2])
cv2.imshow("masked", (image * imageMask).astype(np.uint8))
cv2.waitKey(0)

cv2.destroyAllWindows()
