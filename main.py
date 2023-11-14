import os

import cv2
import numpy as np

from calibrate import get_calibration_points, get_transformation_matrix, transform_image
from fill import get_fill_overlay
from utils import get_frame, show


def get_calibration_cache():
    if os.path.exists('calibration_cache.npy'):
        return np.load('calibration_cache.npy')

    while True:
        frame = next(get_frame(cam=1))
        points = get_calibration_points(frame)
        if len(points) == 4:
            break
        else:
            print('Please select 4 points.')

    matrix = get_transformation_matrix(points)

    np.save('calibration_cache.npy', matrix)

    return matrix


def main():
    matrix = get_calibration_cache()
    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    for frame in get_frame(cam=1):
        frame = transform_image(frame, matrix)
        overlay = get_fill_overlay(frame)

        cv2.imshow('image', overlay)
        cv2.waitKey(2)


if __name__ == '__main__':
    main()
