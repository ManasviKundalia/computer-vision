"""
Transfer Learning with vgg16 to build custom classifiers

1. Smile classifier - Is the person smiling or not? :)

"""
#import sys
import os

#sys.path.append(os.path.abspath(__file__))

import argparse

from keras.applications.vgg16 import VGG16
from classifiers.classifier_utils import transfer_weights
from classifiers.dataset_loader import read_images

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str)
args = parser.parse_args()

DATASET_PATH = args.dataset_path

#load dataset

classes = os.listdir(DATASET_PATH)
print("Classes of the Dataset : ", classes)

imgs = []
labels = []
classwise_counts = []

for cls in classes:
    cls_imgs, cls_labels = read_images(os.path.join(DATASET_PATH, cls), cls)
    imgs.extend(cls_imgs)
    labels.extend(cls_labels)
    classwise_counts.append(len(cls_imgs))

print("Size of dataset: ", len(imgs))

for i, cls in enumerate(classes):
    print(cls, classwise_counts[i])

# load the model
model = VGG16()
print(model.summary())



