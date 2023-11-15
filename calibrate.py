from typing import List, Tuple

import time
import cv2
import numpy as np
from screeninfo import get_monitors
from utils import get_frame

width, height = 1200, 800  # Example dimensions, adjust as necessary


def show_scaled_image(image, zoom_level=1):
    # Scale the image
    image = cv2.resize(image, None, fx=zoom_level, fy=zoom_level, interpolation=cv2.INTER_LINEAR)

    # Display the image
    cv2.imshow('Image', image)


def get_calibration_points() -> List[Tuple[int, int]]:
    """
    Get the calibration points from the image.
    :param image: np.ndarray
    :return: List of 4 points.
    """
    monitor = get_monitors()[0]
    screen_width, screen_height = monitor.width, monitor.height

    # Create a green image of screen size
    image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    image[:] = (0, 255, 0)  # (B, G, R)

    # Display the image in an OpenCV window
    cv2.imshow('Image', image)
    cv2.waitKey(1000)
    image = next(get_frame(cam=1))
    cv2.imwrite('test/test.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image = cv2.imread('test/test.jpg')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(shape_contour, True)
    approx = cv2.approxPolyDP(shape_contour, epsilon, True)

    points = [tuple(pt[0]) for pt in approx]
    temp1 = points[1]
    points[1] = points[3]
    points[3] = temp1
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


if __name__ == '__main__':
    img = cv2.imread('test/test.jpg')
    p = get_calibration_points()
    print(p)
    m = get_transformation_matrix(p)
    img = transform_image(img, m)

    cv2.imwrite('test/adjusted.jpg', img)
