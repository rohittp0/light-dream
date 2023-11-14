import cv2
import numpy as np


def get_average_color(contour, img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean = cv2.mean(img, mask=mask)
    return mean[0:3]


def closest_color(rgb_color):
    color_mappings = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'black': (0, 0, 0)
    }
    colors = np.array(list(color_mappings.values()))
    color_diffs = colors - np.array(rgb_color)
    dist = np.sqrt(np.sum(color_diffs ** 2, axis=1))
    index_of_smallest = np.argmin(dist)
    closest_color_name = list(color_mappings.keys())[index_of_smallest]
    return color_mappings[closest_color_name]


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


def find_and_fill_contours(thresh, image):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        average_color = get_average_color(contour, image)
        color = closest_color(average_color)
        cv2.drawContours(overlay, [contour], -1, color, -1)
    return overlay


def get_fill_overlay(image: np.ndarray) -> np.ndarray:
    """
    Get the overlay of the filled contours.
    :param image: Image to process
    :return: The overlay as a np.ndarray
    """
    thresh = preprocess_image(image)
    overlay = find_and_fill_contours(thresh, image)
    return overlay


if __name__ == '__main__':
    img = cv2.imread('test/adjusted.jpg')
    overlay = get_fill_overlay(img)
    cv2.imwrite('test/overlay.jpg', overlay)
