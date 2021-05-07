import cv2
import argparse
import torch
import torchvision

from src.coco.ModelLiteTorch import LitModel
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', type=str, help='path of input images')
parser.add_argument('--output-path', type=str, help='path of output images')
parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet')
args = parser.parse_args()


bs = 16        # batch size
lr = 0.01       # learn rate

state_dict = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
# state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

model = LitModel(learning_rate = lr,batch_size = bs,)
model.load_state_dict(state_dict)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = cv2.imread(args.input_path)
imageTensor = transform(image)
imageTensor = imageTensor.unsqueeze(0)

_, _, mask = model(imageTensor, True)

mask = mask.squeeze().cpu().detach().numpy()

cv2.imshow("image", image)
cv2.imshow("mask", mask)
cv2.waitKey(0)

cv2.destroyAllWindows()