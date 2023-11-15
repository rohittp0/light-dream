import os
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

from calibrate import get_calibration_points, get_transformation_matrix, transform_image
from fill import get_fill_overlay
from utils import get_frame


def get_calibration_cache():
    if os.path.exists('calibration_cache.npy'):
        return np.load('calibration_cache.npy')

    while True:
        frame = next(get_frame(cam=1))
        points = get_calibration_points()
        if len(points) == 4:
            break
        else:
            print('Please select 4 points.')

    matrix = get_transformation_matrix(points)

    np.save('calibration_cache.npy', matrix)

    return matrix


class VideoApp(QMainWindow):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix
        self.setWindowTitle('Video Fullscreen PyQt')
        self.showFullScreen()
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)  # Align the image to center
        self.label.setScaledContents(True)
        self.setCentralWidget(self.label)

        # Move the window to the second screen if available
        screens = QApplication.screens()
        if len(screens) > 1:
            # Assuming the second monitor is to the right of the primary
            second_screen = screens[1]
            self.move(second_screen.geometry().x(), second_screen.geometry().y())
            self.showFullScreen()

    def display_frame(self, frame):
        # Convert the frame format from OpenCV to QImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.close()


def main():
    matrix = get_calibration_cache()

    app = QApplication(sys.argv)
    ex = VideoApp(matrix)

    for frame in get_frame(cam=1):
        frame = transform_image(frame, matrix)
        overlay = get_fill_overlay(frame)
        ex.display_frame(overlay)
        QApplication.processEvents()  # Update the GUI and check for user interactions

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
