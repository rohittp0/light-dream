import os
import sys
import time
from typing import List, Tuple
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
import cv2
import numpy as np


width, height = 1500, 800  # Example dimensions, adjust as necessary


def show_scaled_image(image, zoom_level=1):
    # Scale the image
    image = cv2.resize(image, None, fx=zoom_level, fy=zoom_level, interpolation=cv2.INTER_LINEAR)

    # Display the image
    cv2.imshow('Image', image)


def get_calibration_points(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Get the calibration points from the user.
    :param image: np.ndarray
    :return: List of 4 points.
    """
    points = []
    zoom_level = 1

    def get_points(event, x, y, flags, _):
        nonlocal points, zoom_level, image

        if event == cv2.EVENT_LBUTTONDOWN:
            # Adjust coordinates based on the zoom level
            adjusted_x = int(x / zoom_level)
            adjusted_y = int(y / zoom_level)
            points.append((adjusted_x, adjusted_y))

            # Draw a circle where the user clicked
            cv2.circle(image, (adjusted_x, adjusted_y), 5, (255, 0, 0), -1)

            # If 4 points have been clicked, perform the transformation
            if len(points) == 4:
                cv2.destroyAllWindows()

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Use bitwise operations to check the scroll direction
            if flags > 0:
                zoom_level *= 1.25
            else:
                zoom_level /= 1.25

        show_scaled_image(image, zoom_level)

    cv2.namedWindow("Image")
    cv2.setMouseCallback('Image', get_points)

    show_scaled_image(image)
    cv2.waitKey(0)

    return points


def get_transformation_matrix(points: List[Tuple[int, int]]) -> np.ndarray:
    """
    Get the transformation matrix for the given points.
    :param points: List of 4 points.
    :return:
    """
    # Define points for the destination
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Get the transformation matrix
    return cv2.getPerspectiveTransform(np.float32(points), dst_points)


def transform_image(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Transform the image using the given matrix.
    :param image: np.ndarray
    :param matrix: np.ndarray
    :return: Transformed image as np.ndarray
    """
    return cv2.warpPerspective(image, matrix, (width, height))



class ImageTransformer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Transformer")
        self.image_label = QLabel(self)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.points = []
        self.zoom_level = 1
        self.loadImage()
        self.displayImage()


    def resizeEvent(self, event):
        # Call the superclass method
        super(ImageTransformer, self).resizeEvent(event)

        # Update the image display
        if hasattr(self, 'cv_image'):  # Check if the image is loaded
            self.displayImage()

    def loadImage(self):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ret, frame = cap.read()
        self.cv_image = frame

        height, width, channel = self.cv_image.shape
        self.setFixedSize(width, height)


    def displayImage(self):
        height, width, channel = self.cv_image.shape
        bytesPerLine = 3 * width
        qImg = QImage(self.cv_image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(qImg).scaled(self.image_label.size() * self.zoom_level, Qt.KeepAspectRatio))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = int(event.x() / self.zoom_level)
            y = int(event.y() / self.zoom_level)
            self.points.append((x, y))
            # cv2.circle(self.cv_image, (x, y), 5, (255, 0, 0), -1)
            print(self.points)
            self.displayImage()
            if len(self.points) == 4:
                self.performTransformation()

    def wheelEvent(self, event):
        angle = event.angleDelta() / 8
        angleY = angle.y()
        if angleY > 0:
            self.zoom_level *= 1.25
        else:
            self.zoom_level /= 1.25
        self.displayImage()

    def performTransformation(self):
        matrix = get_transformation_matrix(self.points)
        transformed_image = transform_image(self.cv_image, matrix)
        cv2.imwrite('adjusted.jpg', transformed_image)
        np.save('calibration_cache.npy', matrix)  # Save the matrix
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageTransformer()
    ex.show()
    sys.exit(app.exec_())

