import time

import cv2

# Load the image
import numpy as np

from calibrate import matrix_path, transform_image
from utils import get_frame

m = np.load(matrix_path)
frame_gen = get_frame()

img = cv2.imread('images/test.png')

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow('Image', img.shape[1], img.shape[0])

while True:
    img = transform_image(next(frame_gen), m)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) == 27:
        break





