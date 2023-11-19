import cv2
import numpy as np


def remove_contour(contours_, hierarchy_, min_area, max_area, image):
    exclude = set()
    contours = list(contours_)
    hierarchy = list(hierarchy_)
    contour_map = {-1: -1}
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area or area < min_area:
            exclude.add(i)


    start = 0
    for i, ex in enumerate(exclude):
        for j in range(start, ex):
            contour_map[j] = j - i
        start = ex + 1

    for j in range(start, len(contours)):
        contour_map[j] = j - len(exclude)


    for i in range(0, len(contours)):
        next = hierarchy[0][i][0]
        while next in exclude:
            next = hierarchy[0][next][0]
        hierarchy[0][i][0] =  contour_map[next]

        prev = hierarchy[0][i][1]
        while prev in exclude:
            prev = hierarchy[0][prev][1]
        hierarchy[0][i][1] =  contour_map[prev]


        if hierarchy[0][i][2] in exclude:
            hierarchy[0][i][2] = -1
        else:
            hierarchy[0][i][2] = contour_map[hierarchy[0][i][2]]

        if hierarchy[0][i][3] in exclude:
            hierarchy[0][i][3] = -1
        else:
            hierarchy[0][i][3] = contour_map[hierarchy[0][i][3]]

    filtered_contour  = [value for index, value in enumerate(contours) if index not in exclude]
    filtered_hierarchy = [[value for index, value in enumerate(hierarchy[0]) if index not in exclude]]


    return filtered_contour, filtered_hierarchy


def sort_contours_into_levels(contours, hierarchy):
    levels = []
    # Initialize levels list
    for i in range(0, len(contours)):
        levels.append([])

    # Iterate through contours and hierarchy
    for i, (contour, hier) in enumerate(zip(contours, hierarchy[0])):
        level = 0
        node = i
        while hierarchy[0][node][3] != -1:
            level += 1
            node = hierarchy[0][node][3]

        levels[level].append(i)

    return levels

def get_average_color(idx, contours, hierarchy, image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contours[idx]], -1, 255, -1)
    child = hierarchy[0][idx][2]
    while child != -1:
        cv2.drawContours(mask, [contours[child]], -1, 0, -1)
        child = hierarchy[0][child][0]

    mean = cv2.mean(image, mask=mask)
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
    # image = whiteboard_enhance(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


def is_contour_closed(contour, closed_tolerance=0.0002):
    # Calculate the arc length of the contour
    closed_perimeter = cv2.arcLength(contour, True)
    open_perimeter = cv2.arcLength(contour, False)

    # If the closed perimeter is not significantly larger than the open perimeter,
    # it suggests the contour is closed.
    return (open_perimeter - closed_perimeter) / open_perimeter < closed_tolerance


def find_and_fill_contours(thresh, image):
    kernel = np.ones((4, 4), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Here we retrieve the hierarchy information along with the contours
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = remove_contour(contours, hierarchy, 500, 2000000, image)

    # Sort the contours into levels
    levels = sort_contours_into_levels(contours, hierarchy)

    overlay = np.zeros_like(image)


    # Iterate through each contour and its corresponding hierarchy element
    for level in levels:
        for i in level:
            contour = contours[i]

            if not is_contour_closed(contour):
                continue

            average_color = get_average_color(i, contours, hierarchy, image)
            color = closest_color(average_color)

            # ov = np.zeros_like(image)
            # cv2.drawContours(ov, [contour], -1, color, -1)
            # cv2.imwrite('test/contour/cont' + str(i) + '.jpg', ov)

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
    over = get_fill_overlay(img)
    cv2.imwrite('test/overlay.jpg', over)
