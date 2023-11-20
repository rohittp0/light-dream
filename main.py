import os
import sys
from time import sleep

import PyQt5
import numpy as np
from PyQt5.QtWidgets import QApplication

from app import VideoApp
from calibrate import get_calibration_points, get_transformation_matrix, transform_image
from fill import get_fill_overlay
from utils import get_frame, check_for_change


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

    frames = get_frame(ex, cam=1)

    matrix = get_calibration_cache(next(frames))

    last_frame = next(frames)
    change = True
    white = np.zeros((480, 640, 3), dtype=np.uint8)

    for frame in frames:

        if check_for_change(frame, last_frame):
            change = True
            last_frame = frame
            continue

        if not change:
            continue

        ex.display_frame(white)
        frame = next(frames)
        frame = next(frames)

        change = False
        frame = transform_image(frame, matrix)
        overlay = get_fill_overlay(frame)
        ex.display_frame(overlay)
        last_frame = next(frames)

        QApplication.processEvents()  # Update the GUI and check for user interactions


    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
