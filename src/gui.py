import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtCore import pyqtSlot
from webcam_cv2 import VideoProcessing, show_webcam_gray, show_webcam_color

global vp

gray_scale = False
mirror = False

def window():
    app = QApplication(sys.argv)
    w = QWidget()
    b = QLabel(w)
    b.setText("Basic AF!")
    w.setGeometry(500,20,500,500)
    b.move(80,10)

    button = QPushButton(w)
    button.setText("Start cam")
    button.move(80,30)
    button.clicked.connect(on_click_main)

    button_capture = QPushButton(w)
    button_capture.setText("Save Please")
    button_capture.move(120,30)
    button_capture.clicked.connect(on_click_save)

    button_gray = QPushButton(w)
    button_gray.setText("50 shades of Grey!")
    button_gray.move(80,50)
    button_gray.clicked.connect(on_click_gray)

    button_mirror = QPushButton(w)
    button_mirror.setText("Mirror mirror on the wall!")
    button_mirror.move(80,70)
    button_mirror.clicked.connect(on_click_mirror)

    button_random = QPushButton(w)
    button_random.setText("That's so Random!")
    button_random.move(80,90)
    button_random.clicked.connect(on_click_random)



    w.setWindowTitle("Wassup!")
    w.show()
    sys.exit(app.exec_())

@pyqtSlot()
def on_click_main():
    global vp
    vp = VideoProcessing()
    vp.show_webcam()

@pyqtSlot()
def on_click_save():
    vp.capture_image()

@pyqtSlot()
def on_click_random():
    vp.random = not(vp.random)

@pyqtSlot()
def on_click_gray():

    vp.gray_scale = not(vp.gray_scale)


@pyqtSlot()
def on_click_mirror():

    vp.mirror = not(vp.mirror)


if __name__ == '__main__':
    window()