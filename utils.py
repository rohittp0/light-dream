import cv2


def get_frame():
    """
    Reads a frame from the camera using OpenCV
    :return: Frame
    """

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        yield frame

