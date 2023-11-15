import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QLabel, QApplication


class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Video Fullscreen PyQt')
        self.showFullScreen()
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)  # Align the image to center
        self.label.setScaledContents(True)
        self.setCentralWidget(self.label)

        # Move the window to the second screen if available
        screens = QApplication.screens()
        if len(screens) > 1:
            # Assuming the second monitor is to the right of the primary
            second_screen = screens[1]
            self.move(second_screen.geometry().x(), second_screen.geometry().y())
            self.showFullScreen()

    def display_frame(self, frame):
        # Convert the frame format from OpenCV to QImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))
        QApplication.processEvents()

    def closeEvent(self, event):
        self.close()
