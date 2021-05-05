import numpy as np
import cv2

def makeEdgeMask(mask):
    kernel = np.ones((5,5), np.uint8)

    erosion = cv2.erode(mask, kernel, iterations = 1)
    dilation = cv2.dilate(mask, kernel, iterations = 1)

    return dilation - erosion
