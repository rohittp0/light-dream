from typing import List, Tuple

import cv2
import numpy as np

from utils import get_frame

matrix_path = 'proj_matrix.npy'

# Save NumPy array to file
aspect = 1.58
width = 1280
height = int(width/aspect)

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
    destroy = False

    def get_points(event, x, y, flags, _):
        nonlocal points, zoom_level, image, destroy


        if event == cv2.EVENT_LBUTTONDOWN:
            # Adjust coordinates based on the zoom level
            adjusted_x = int(x / zoom_level)
            adjusted_y = int(y / zoom_level)
            points.append((adjusted_x, adjusted_y))

            # Draw a circle where the user clicked
            cv2.circle(image, (adjusted_x, adjusted_y), 5, (255, 0, 0), -1)

            # If 4 points have been clicked, perform the transformation
            if len(points) == 4:
                destroy = True

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Use bitwise operations to check the scroll direction
            if flags > 0:
                zoom_level *= 1.25
            else:
                zoom_level /= 1.25

        show_scaled_image(image, zoom_level)

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', get_points)

    # show_scaled_image(image)
    while True:
        if destroy:
            cv2.destroyWindow("Image")
            break
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyWindow("Image")
            break

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
    # img = cv2.imread('./images/board.jpg')
    frame_gen = get_frame()
    img = next(frame_gen)
    p = get_calibration_points(img)
    m = get_transformation_matrix(p)
    np.save(matrix_path, m)

    # img = transform_image(img, m)
    # cv2.imwrite('images/test.png', img)
    # while True:
    #     key = cv2.waitKey(0)
    #     if key == 27:
    #         break
