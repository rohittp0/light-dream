import cv2


def get_frame():
    """
    Reads a frame from the camera using OpenCV
    :return: Frame
    """

    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        yield frame

