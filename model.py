import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten


def model(k, p, padd):
    _model = Sequential()
    _model.add(Conv2D(64, (k, k), activation="relu", input_shape=(224, 224, 3), padding=padd))
    _model.add(Conv2D(64, (k, k), activation="relu", padding=padd))
    _model.add(MaxPool2D(pool_size=(p, p)))

    _model.add(Conv2D(128, (k, k), activation="relu", padding=padd))
    _model.add(MaxPool2D(pool_size=(p, p)))

    _model.add(Conv2D(128, (k, k), activation="relu", padding=padd))
    _model.add(MaxPool2D(pool_size=(p, p)))

    _model.add(Conv2D(256, (k, k), activation="relu", padding=padd))
    _model.add(MaxPool2D(pool_size=(p, p)))

    _model.add(Flatten())
    _model.add(Dense(64, activation="relu"))
    _model.add(Dense(1, activation="sigmoid"))
    _model.compile(loss=tf.keras.losses.binary_crossentropy,
                   optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=["accuracy"])
    # print(_model.summary())

    return _model


if __name__ == '__main__':
    model(3, 2, "same")
