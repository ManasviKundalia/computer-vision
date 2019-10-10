"""
Transfer Learning with vgg16 to build custom classifiers

1. Smile classifier - Is the person smiling or not? :)

"""
from keras.applications.vgg16 import VGG16

# load the model
model = VGG16()
print(model.summary())



