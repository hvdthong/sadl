import argparse

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.optimizers import SGD

CLIP_MIN = -0.5
CLIP_MAX = 0.5
weight_decay = 0.0005

def train(args):
    if args.d == "cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()        

        layers = [
            Conv2D(64, (3, 3), padding="same", input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),
            Dropout(0.3),

            Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),

            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),            
            BatchNormalization(),
            Dropout(0.4),

            Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),

            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),
            Dropout(0.4),

            Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),
            Dropout(0.4),

            Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),

            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),
            Dropout(0.4),

            Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),
            Dropout(0.4),

            Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
            Activation("relu"),
            BatchNormalization(),

            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),

            Flatten(),         
            Dense(512, kernel_regularizer=l2(0.0005)),
            Activation('relu'),
            BatchNormalization(),

            Dropout(0.5),
            Dense(10),
        ]
    
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.add(Activation("softmax"))

    print(model.summary())
    opt = SGD(lr=0.0001, decay=1e-6, momentum=0.9)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )

    # checkpoint
    filepath= "./model_tracking/%s_model_improvement-{epoch:02d}-{val_accuracy:.2f}.h5" % (args.d)
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(
        x_train,
        y_train,
        epochs=500,
        batch_size=32,
        shuffle=True,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, type=str)
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"

    train(args)
