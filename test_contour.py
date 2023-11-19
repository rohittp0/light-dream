import cv2
import numpy as np

from fill import preprocess_image, find_and_fill_contours

# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# ret, frame = cap.read()

img = cv2.imread('test/test_color.jpeg')


def get_fill_overlay(image: np.ndarray) -> np.ndarray:
    """
    Get the overlay of the filled contours.
    :param image: Image to process
    :return: The overlay as a np.ndarray
    """
    cv2.imshow('img', img)
    cv2.moveWindow('img', 100, 100)

    thresh = preprocess_image(image)
    cv2.imshow('proc', thresh)
    cv2.moveWindow('proc', 800, 100)


    overlay = find_and_fill_contours(thresh, image)
    cv2.imshow('overlay', overlay)
    cv2.moveWindow('overlay', 300, 600)

    return overlay


# cv2.imwrite('test/cont.jpg', img)

overlay = get_fill_overlay(img)
# cv2.imwrite('test/cont_proc.jpg', overlay)

while cv2.waitKey(0) != 27:
    pass
