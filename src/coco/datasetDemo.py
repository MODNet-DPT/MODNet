import os
import cv2
import numpy as np

from coco.SegmentationDataset import SegmentationDataset

datasetSize = 11

humanDataset = SegmentationDataset(os.path.join("../data", 'human'), datasetSize, shuffle=True)

x_train, y_train = humanDataset.readBatch(datasetSize)

for i in range(datasetSize):
    cv2.imshow("image", x_train[i])
    cv2.imshow("mask", y_train[i] * 255)
    cv2.waitKey()

cv2.destroyAllWindows()