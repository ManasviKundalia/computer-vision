"""
Utils to prepare data, train classifiers
"""
from keras.models import Sequential
from keras.layers import Flatten, Dense, Input
from keras.callbacks import LambdaCallback
from keras.applications.vgg16 import VGG16
global n_layers
n_layers=11

def transfer_weights(model, n_layers=7):
    """
    to do transfer weights from pretrained model to new model
    new model arch = model[:n_layers] + flatten + dense1 + dense2
    :param model: original model
    :param n_layers: number of layers to be transferred
    :return:
    """
    model_res = Sequential()
    for i in range(n_layers):
        model_res.add(model.layers[i])

    for i in range(len(model_res.layers)-4):
        model_res.layers[i].trainable = False

    model_res.add(Flatten())
    model_res.add(Dense(units=512, activation='relu'))
    # model_res.add(Dense(units=128))
    return model_res

def build_model_vgg16(weights_path, n_layers=15):
    model = Sequential()
    vgg = VGG16()
    for i in range(n_layers):
        model.add(vgg.layers[i])
    model.add(Flatten())
    model.add(Dense(units=512))
    model.add(Dense(units=64))
    model.add(Dense(units=2))

    model.load_weights(weights_path)
    return model


def freeze_layers(model, n_l=None):
    # if n_l is None:
    #     global n_layers
    # else:
    n_layers = n_l
    for i in range(n_layers):
        model.layers[i].trainable = False

    boundry_layer_name = model.layers[n_layers].name
    if boundry_layer_name.find('block')!=-1:
        block_num = boundry_layer_name[:boundry_layer_name.find('_')+1]
        for i in range(n_layers):

            if model.layers[i].name.find(block_num)!=-1:
                model.layers[i].trainable = True

    for i in range(n_layers, len(model.layers)):
        model.layers[i].trainable = True
    # if n_l is None:
    #     n_layers-=1
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def unfreeze_and_train_fn(model, n_freeze):
    return freeze_layers(model, n_freeze)


def train(model_, imgs, labels, n_epochs=1, unfreeze_and_train=False, n_freeze_layers=0, callbacks=None):
    if unfreeze_and_train:
        total_epochs = 0
        while n_freeze_layers >= 7:
            model_ = unfreeze_and_train_fn(model_, n_freeze_layers)
            model_.fit(imgs, labels, verbose=True, batch_size=16, epochs=n_epochs+total_epochs, callbacks=callbacks, initial_epoch=total_epochs)
            n_freeze_layers -= 1
            total_epochs+=n_epochs
        model_.fit(imgs, labels, verbose=True, batch_size=16, epochs=3*n_epochs+total_epochs, callbacks=callbacks, initial_epoch=total_epochs)

    else:
        model_.fit(imgs, labels, verbose=True, batch_size=16, epochs=n_epochs, callbacks=callbacks)

    return model_
