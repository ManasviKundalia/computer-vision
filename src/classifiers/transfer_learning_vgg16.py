"""
Transfer Learning with vgg16 to build custom classifiers

1. Smile classifier - Is the person smiling or not? :)

"""
#import sys
import os
import numpy as np
#sys.path.append(os.path.abspath(__file__))
import pickle as pk

import argparse

from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from classifiers.classifier_utils import transfer_weights, train
from classifiers.dataset_loader import read_images, normalize_by_resize, shuffle_dataset
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--tensorboard", type=bool, default=False)

args = parser.parse_args()

DATASET_PATH = args.dataset_path
tb = args.tensorboard

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
model_res = transfer_weights(model, n_layers=15)
model_res.add(Dense(units=64, activation='tanh'))
model_res.add(Dense(units=len(classwise_counts), activation='softmax'))
print(model_res.summary())

# shuffle images
imgs, labels = shuffle_dataset(imgs, labels)

# normalize by resize

imgs = normalize_by_resize(imgs, size=model.layers[0].input_shape[1:3])

label_classes = set(labels)
label2id = dict([(l, i) for (i, l) in enumerate(label_classes)])
labels_cat = to_categorical([label2id[l] for l in labels], num_classes=len(classwise_counts))

# train test split

imgs_train, imgs_test, labels_train, labels_test = train_test_split(imgs, labels_cat, random_state=22)

callbacks = None

if tb:
    print("Using tensorboard callback")
    tensorboard = TensorBoard(log_dir='/home/manasvi/logs/comp-vis', histogram_freq=1, write_grads=True,
                              write_graph=True)
    callbacks = [tensorboard]

model_res.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train(model_res, np.array(imgs_train), labels_train, n_epochs=1, unfreeze_and_train=False, n_freeze_layers=15,
      callbacks=callbacks)

print("Model performance on test data")
print(model_res.evaluate(np.array(imgs_test), labels_test))

for i in range(5):
    img_p = np.array([imgs[i]])
    print(img_p.shape, type(img_p))
    print(model_res.predict(np.array([imgs[i]])))
    print(labels_cat[i])

# model_path = '/home/manasvi/IdeaProjects/computer-vision/src/models/vgg16.bin'
# print("Saving model to :", model_path)
# model_res.save(model_path, include_optimizer=False)
# model_res.save_weights('/home/manasvi/IdeaProjects/computer-vision/src/models/vgg16_weights.bin')
# pk.dump(model_res, open('/home/manasvi/IdeaProjects/computer-vision/src/models/vgg16_pk.bin', 'wb'))

