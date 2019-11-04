import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QGraphicsDropShadowEffect, \
    QGraphicsColorizeEffect
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QColor, QFont
from webcam_cv2 import VideoProcessing, show_webcam_gray, show_webcam_color

global vp

gray_scale = False
mirror = False

def window():
    app = QApplication(sys.argv)
    w = QWidget()
    w.setGeometry(500,10,300,500)

    font = QFont()
    font.setFamily("Calibri")
    w.setFont(font)
    b = QLabel(w)
    b.setText("Basic AF!")

    colorize = QGraphicsColorizeEffect()
    colorize.setColor(QColor(0,0,192))

    b.setGraphicsEffect(colorize)
    b.move(120,10)

    button = QPushButton(w)
    button.setText("Start cam")
    button.move(10,30)
    button.clicked.connect(on_click_main)

    button_capture = QPushButton(w)
    button_capture.setText("Save Please")
    button_capture.move(170,30)
    button_capture.clicked.connect(on_click_save)

    button_gray = QPushButton(w)
    button_gray.setText("50 shades of Grey!")
    button_gray.move(70,80)
    button_gray.clicked.connect(on_click_gray)

    button_mirror = QPushButton(w)
    button_mirror.setText("Mirror mirror on the wall!")
    button_mirror.move(70,110)
    button_mirror.clicked.connect(on_click_mirror)

    button_random = QPushButton(w)
    button_random.setText("That's so Random!")
    button_random.move(70,140)
    button_random.clicked.connect(on_click_random)

    button_halfNhalf = QPushButton(w)
    button_halfNhalf.setText("Invert Half")
    button_halfNhalf.move(70, 170)
    button_halfNhalf.clicked.connect(on_click_halfnhalf)

    button_border = QPushButton(w)
    button_border.setText("Diamond Border")
    button_border.move(70,200)
    button_border.clicked.connect(on_click_border_pattern)

    button_border = QPushButton(w)
    button_border.setText("Smile please!")
    button_border.move(70,230)
    button_border.clicked.connect(on_click_smile)

    w.setWindowTitle("My App")
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

@pyqtSlot()
def on_click_halfnhalf():
    vp.halfNhalf = not(vp.halfNhalf)

@pyqtSlot()
def on_click_border_pattern():
    vp.patternBorder = not(vp.patternBorder)

@pyqtSlot()
def on_click_smile():
    vp.smile = not(vp.smile)

if __name__ == '__main__':
    window()