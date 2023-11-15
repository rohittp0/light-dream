import platform

import cv2


def get_frame(cam=0):
    """
    Reads a frame from the camera using OpenCV
    :return: Frame
    """
    mode = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_GSTREAMER

    cap = cv2.VideoCapture(cam, mode)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        yield frame


def show(*images):
    """
    Shows one or more images using OpenCV
    :param images: One or more images
    :return: None
    """
    for i, image in enumerate(images):
        cv2.imshow(f'Image {i}', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_hierarchy(image, contours, hierarchy):
    # Create a copy of the image to draw on
    hierarchy_image = image.copy()

    # Draw all contours in blue
    cv2.drawContours(hierarchy_image, contours, -1, (255, 0, 0), 3)

    # Go through all the contours
    for i, (contour, hier) in enumerate(zip(contours, hierarchy[0])):
        # The 3rd value in the hierarchy array is the index of the first child
        # Draw a line from each contour to its first child (if it has one)
        if hier[2] != -1:
            # Get the center of the current contour
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Get the center of the child contour
            child_contour = contours[hier[2]]
            M = cv2.moments(child_contour)
            child_cx = int(M['m10'] / M['m00'])
            child_cy = int(M['m01'] / M['m00'])

            # Draw the line
            cv2.line(hierarchy_image, (cx, cy), (child_cx, child_cy), (0, 255, 0), 2)

            # Optionally, put the index number of the contour
            cv2.putText(hierarchy_image, str(i), (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the image with hierarchy
    cv2.imshow('Hierarchy', hierarchy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
