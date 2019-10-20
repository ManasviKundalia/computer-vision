"""
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2
import time
from os import listdir
import numpy as np


def diamond_mask(img_shape):
    mask = np.ones((img_shape[0], img_shape[1]))
    x_mid = int(img_shape[0]/2)
    y_mid = int(img_shape[1]/2)

    #diamond pattern?

    slope = y_mid/x_mid    #y=slope*x+c  ---> x = 0 y=y_mid x=x_mid y = 0
    for i in range(x_mid):
        j = 0
        while(j<=y_mid) and (j<=abs(slope*i-y_mid)):
            mask[i][j] = 0
            j+=1

    for i in range(x_mid, img_shape[0]):
        j = 0
        while(j<=y_mid) and (j<=slope*(i-x_mid)):
            mask[i][j] = 0
            j+=1

    for i in range(x_mid):
        j = int(slope*i) + y_mid
        while(j<img_shape[1]):
            mask[i][j] = 0
            j+=1

    for i in range(x_mid, img_shape[0]):
        j = 3*y_mid - int(slope*i) # x =x_mid y = 2*y_mid x = 2*x_mid y=y_mid
        while(j<img_shape[1]):
            mask[i][j] = 0
            j+=1

    return mask

class VideoProcessing:

    def __init__(self):
        #filters
        self.gray_scale = False
        self.mirror = False
        self.random = False
        self.halfNhalf = False
        self.patternBorder = False
        #img
        self.img = None
        self.img_directory = "/home/manasvi/Pictures/mywebcam/"

    def show_webcam(self):
        cam = cv2.VideoCapture(0)
        global gray_scale, mirror
        while True:
            ret_val, self.img = cam.read()
            if self.mirror:
                self.img = cv2.flip(self.img, 1)

            if self.random:
                self.img = np.rot90(self.img)

            if self.halfNhalf:
                self.half_n_half()

            if self.patternBorder:
                self.pattern_boundry()

            if self.gray_scale:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('my webcam', self.img)
            if cv2.waitKey(1) == 27:
                break  # esc to quit

        cv2.destroyAllWindows()

    def add_black_line(self):
        img_shape = self.img.shape
        x_mid = int(img_shape[0]/2)
        self.img[x_mid][:][:] = 0

    def half_n_half(self):
        img_shape = self.img.shape
        x_mid = int(img_shape[0]/2)

        #invert the first half
        self.img[0:x_mid][:][:] = 255-self.img[0:x_mid][:][:]

    def pattern_boundry(self):
        mask = diamond_mask(self.img.shape)
        mask_3d = np.ones(self.img.shape)
        mask_3d[:, :, 0] = mask
        mask_3d[:, :, 1] = mask
        mask_3d[:, :, 2] = mask
        self.img = np.multiply(np.array(self.img), np.array(mask_3d, dtype=type(self.img[0][0][0])))
        #print(type(self.img[0][0][0]))
        #self.img = np.multiply(np.array(self.img),np.ones(self.img.shape, dtype=type(self.img[0][0][0])))
        #print(self.img[0,0,0])

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