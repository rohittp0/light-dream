import os
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

from app import VideoApp
from calibrate import get_calibration_points, get_transformation_matrix, transform_image
from fill import get_fill_overlay
from utils import get_frame


def get_calibration_cache(ex: VideoApp):
    if os.path.exists('calibration_cache.npy'):
        return np.load('calibration_cache.npy')

    while True:
        points = get_calibration_points(ex)
        if len(points) == 4:
            break
        else:
            print('Please select 4 points.')

    matrix = get_transformation_matrix(points)

    np.save('calibration_cache.npy', matrix)

    return matrix


def main():
    app = QApplication(sys.argv)
    ex = VideoApp()

    matrix = get_calibration_cache(ex)

    for frame in get_frame(cam=1):
        frame = transform_image(frame, matrix)
        overlay = get_fill_overlay(frame)
        ex.display_frame(overlay)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
