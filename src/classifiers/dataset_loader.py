"""
Load datasets for image classification
"""
import os
import cv2
from copy import deepcopy
import numpy as np

def read_images(directory, label = None):
    """
    function to read images
    :param directory: path to images
    :param label: target label (optional)
    :return:
    """

    imgs = []
    labels = [] if label is not None else None

    files = os.listdir(directory)

    for file in files:
        imgs.append(cv2.imread(os.path.join(directory,file)))
        if label is not None:
            labels.append(label)

    return imgs, labels


def normalize_img_dataset_by_cropping(imgs, size=(244,244)):
    normalized_imgs = []

    for img in imgs:
        img_copy = deepcopy(img)
        img_shape = img.shape
        print(img_shape)
        if img_shape[0]> size[0]: # crop along x_axis
            x_diff = img_shape[0] - size[0]
            x_diff_half = int(x_diff/2)

            img_copy = img_copy[x_diff_half:x_diff_half+size[0], :, :]

        if img_shape[1]> size[1]: # crop along y axis
            y_diff = img_shape[1] - size[1]
            y_diff_half = int(y_diff/2)
            print(y_diff_half)
            img_copy = img_copy[:, y_diff_half:y_diff_half+size[1], :]
        print(img_copy.shape)
        if img_shape[0]<size[0]: # pad along x axis
            x_diff = size[0] - img_shape[0]
            x_diff_half = int(x_diff/2)
            pad_img = np.zeros((x_diff_half, img_copy.shape[1], img_shape[2]), dtype=type(img[0][0][0]))
            img_copy = np.concatenate([ pad_img, img_copy, pad_img ], axis=0)

        if img_shape[1]<size[1]: # pad along y axis
            y_diff = size[1] - img_shape[1]
            y_diff_half = int(y_diff/2)
            pad_img = np.zeros((img_copy.shape[0], y_diff_half, img_shape[2]), dtype=type(img[0][0][0]))
            img_copy = np.concatenate([pad_img, img_copy, pad_img], axis=1)

        normalized_imgs.append(img_copy)

    return normalized_imgs


if __name__=='__main__':
    print("Testing normalize imgs:")
    img_path = "../datasets/smile-dataset/smile/7.jpg"
    img = cv2.imread(img_path)
    print("Size of original img: ", img.shape)
    cv2.imshow('image', img)
    cv2.imwrite("/home/manasvi/Pictures/orignal1.jpg",img)
    cv2.waitKey(0)

    img_norm = normalize_img_dataset([img])[0]
    print("Size of normalized img: ", img_norm.shape)
    cv2.imshow('image', img)
    cv2.imwrite("/home//manasvi/Pictures/norm1.jpg", img_norm)
    cv2.waitKey(0)
