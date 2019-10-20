"""
Utils to prepare data, train classifiers
"""
from keras.models import Sequential
from keras.layers import Flatten, Dense, Input


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

    model_res.add(Flatten())
    model_res.add(Dense(units=1024))
    model_res.add(Dense(units=512))
    return model_res


def freeze_layers(model, n_layers):
    for i in range(n_layers):
        model.layers[i].trainable = False

    for i in range(n_layers, len(model.layers)):
        model.layers[i].trainable = True
    return model


def unfreeze_and_train_fn(model, n_freeze):
    return freeze_layers(model, n_freeze)


def train(model, imgs, labels, n_epochs=20, unfreeze_and_train=False, n_freeze_layers=0):
    if unfreeze_and_train:
        while n_freeze_layers >= 7:
            model = unfreeze_and_train_fn(model, n_freeze_layers)
            for i in range(n_epochs):
                model.fit(imgs, labels, verbose=True, batch_size=16)
            n_freeze_layers -= 1
    else:
        for i in range(n_epochs):
            model.fit(imgs, labels, verbose=True, batch_size=16)

    return model
