import os
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

from app import VideoApp
from calibrate import get_calibration_points, get_transformation_matrix, transform_image
from fill import get_fill_overlay
from utils import get_frame


def get_calibration_cache(image: np.ndarray):
    if os.path.exists('calibration_cache.npy'):
        return np.load('calibration_cache.npy')

    points = get_calibration_points(image)
    matrix = get_transformation_matrix(points)

    np.save('calibration_cache.npy', matrix)

    return matrix


def main():
    app = QApplication(sys.argv)
    ex = VideoApp()

    frames = get_frame(cam=1)
    matrix = get_calibration_cache(next(frames))

    for frame in frames:
        frame = transform_image(frame, matrix)
        overlay = get_fill_overlay(frame)
        ex.display_frame(overlay)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
