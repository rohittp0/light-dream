import sys
from typing import List, Tuple

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from app import VideoApp
from utils import get_frame


def show_scaled_image(image, zoom_level=1):
    # Scale the image
    image = cv2.resize(image, None, fx=zoom_level, fy=zoom_level, interpolation=cv2.INTER_LINEAR)

    # Display the image
    cv2.imshow('Image', image)


def get_calibration_points(window: VideoApp) -> List[Tuple[int, int]]:
    """
    Get the calibration points from the image.
    :param window: VideoApp
    :return: List of 4 points.
    """
    # Create a green image of screen size
    image = np.zeros((1200, 800, 3), dtype=np.uint8)
    image[:] = (0, 255, 0)  # (B, G, R)

    # Display the image in an OpenCV window
    window.display_frame(image)
    image = next(get_frame(cam=1))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(shape_contour, True)
    approx = cv2.approxPolyDP(shape_contour, epsilon, True)

    return [tuple(point[0]) for point in approx]


def get_transformation_matrix(points: List[Tuple[int, int]]) -> Tuple[np.ndarray, int, int]:
    """
    Get the transformation matrix for the given points.
    :param points: List of 4 points forming a quadrilateral.
    :return: Transformation matrix, width, and height.
    """
    # Calculate the width and height of the region defined by the points
    width_a = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    width_b = np.sqrt(((points[1][0] - points[0][0]) ** 2) + ((points[1][1] - points[0][1]) ** 2))
    width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    height_b = np.sqrt(((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    height = max(int(height_a), int(height_b))

    # Define points for the destination
    dst_points = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])

    print("points", points)

    # Get the transformation matrix
    M = cv2.getPerspectiveTransform(np.float32(points), dst_points)

    return M, width, height


def transform_image(image: np.ndarray, matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Transform the image using the given matrix.
    :param height: Height of the image
    :param width: Width of the image
    :param image: np.ndarray
    :param matrix: np.ndarray
    :return: Transformed image as np.ndarray
    """
    return cv2.warpPerspective(image, matrix, (width, height))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoApp()

    p = get_calibration_points(ex)
    m = get_transformation_matrix(p)
    img = transform_image(cv2.imread('test/test.jpg'), m)

    cv2.imwrite('test/adjusted.jpg', img)
    sys.exit(app.exec_())
