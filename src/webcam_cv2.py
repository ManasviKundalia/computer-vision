"""
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2
import time
from os import listdir
import numpy as np

class VideoProcessing:

    def __init__(self):
        self.gray_scale = False
        self.mirror = False
        self.random = False
        self.img = None
        self.img_directory = "/home/manasvi/Pictures/mywebcam/"

    def show_webcam(self):
        cam = cv2.VideoCapture(0)
        global gray_scale, mirror
        while True:
            ret_val, self.img = cam.read()
            if self.mirror:
                self.img = cv2.flip(self.img, 1)

            if self.gray_scale:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

            if self.random:
                self.img = np.rot90(self.img)

            cv2.imshow('my webcam', self.img)
            if cv2.waitKey(1) == 27:
                break  # esc to quit

        cv2.destroyAllWindows()

    def capture_image(self):
        images = listdir(self.img_directory)
        if len(images)==0:
            img_no = 0
        else:
            images = sorted([int(a.strip('.jpeg'))for a in images if a.strip('.jpeg').isdigit()])
            img_no = images[-1]+1
        cv2.imwrite(self.img_directory+str(img_no)+".jpeg", self.img)


def show_webcam_color(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def show_webcam_gray():
    cam = cv2.VideoCapture(0)
    t = time.time()
    while True:
        ret_csv, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        #if (time.time()-t)>10:
        #    break
        if cv2.waitKey(1)==27:
            break


    cv2.destroyAllWindows()

def main():
    show_webcam_gray()
    #show_webcam_color(mirror=False)


if __name__ == '__main__':
    main()