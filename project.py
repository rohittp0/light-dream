import time

import cv2

# Load the image
import numpy as np

from calibrate import matrix_path, transform_image
from fill import get_fill_overlay
from utils import get_frame


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

m = np.load(matrix_path)
frame_gen = get_frame()

img = cv2.imread('images/test.png')

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow('Image', img.shape[1], img.shape[0])

while True:
    # img = transform_image(next(frame_gen), m)
    img_ = apply_brightness_contrast(img, 20, 30)
    overlay = get_fill_overlay(img_)

    cv2.imshow('Image', overlay)
    if cv2.waitKey(1) == 27:
        break





