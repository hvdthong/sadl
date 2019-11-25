import argparse
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from keras import utils
import numpy as np
import pandas as pd 
import random
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.optimizers import SGD
import tensorflow as tf
import keras.backend as K

CLIP_MIN = -0.5
CLIP_MAX = 0.5
weight_decay = 0.0005

def cut_off_score(args):
    if args.d == 'cifar':
        dict_label = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
        
        
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
        model.summary()

        x_train = x_train.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = x_test.astype("float32")
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        size_train = int((args.p * y_train.shape[0]) / len(dict_label))

        # number of class
        num_class = 10
        y_train = utils.to_categorical(y_train, num_class)
        y_test = utils.to_categorical(y_test, num_class)                
        
        index_label = [i for i in range(0, 50000)]
        true_label = list(np.argmax(y_train, axis=1))

        layer_names = ['dense_2']

        temp_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
        )

        if len(layer_names) == 1:
            layer_outputs = [temp_model.predict(x_train, batch_size=args.batch_size, verbose=1)]
        else:
            layer_outputs = temp_model.predict(x_train, batch_size=args.batch_size, verbose=1)

        y_train_pred_no_norm = layer_outputs[0]        
        y_train_pred_no_norm = list(np.amax(y_train_pred_no_norm, axis=1))
        
        df = pd.DataFrame({'Index': index_label, 'Label': true_label, 'Confidence':y_train_pred_no_norm})
        df = df.sort_values(by=['Label', 'Confidence'])
        df_dataset = []
        for i in range(0, 10):
            if i == 0:
                df_dataset = df.loc[df['Label'] == i]
                df_dataset = df_dataset[:size_train]
            else:
                df_dataset_i = df.loc[df['Label'] == i]
                df_dataset_i = df_dataset_i[:size_train]

                frames = [df_dataset, df_dataset_i]
                df_dataset = pd.concat(frames)

        for i in range(0, 10):
            print(df_dataset.loc[df_dataset['Label'] == i].tail())
            print(i, len(df_dataset.loc[df_dataset['Label'] == i]))

def loading_data_for_training(args):
    if args.d == 'cifar':
        dict_label = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
        
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
        model.summary()

        x_train = x_train.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = x_test.astype("float32")
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        size_train = int((args.p * y_train.shape[0]) / len(dict_label))

        # number of class
        num_class = 10
        y_train = utils.to_categorical(y_train, num_class)
        y_test = utils.to_categorical(y_test, num_class)        

        # # evaluate
        # scores = model.evaluate(x_train, y_train, batch_size=128, verbose=1)
        # print('\nEvaluation result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
        # exit()
        
        index_label = [i for i in range(0, 50000)]
        true_label = list(np.argmax(y_train, axis=1))

        layer_names = ['dense_2']

        temp_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
        )

        if len(layer_names) == 1:
            layer_outputs = [temp_model.predict(x_train, batch_size=args.batch_size, verbose=1)]
        else:
            layer_outputs = temp_model.predict(x_train, batch_size=args.batch_size, verbose=1)

        y_train_pred_no_norm = layer_outputs[0]        
        y_train_pred_no_norm = list(np.amax(y_train_pred_no_norm, axis=1))
        
        df = pd.DataFrame({'Index': index_label, 'Label': true_label, 'Confidence':y_train_pred_no_norm})
        df = df.sort_values(by=['Label', 'Confidence'])
        df_dataset = []
        for i in range(0, 10):
            if i == 0:
                df_dataset = df.loc[df['Label'] == i]
                df_dataset = df_dataset[:size_train]
            else:
                df_dataset_i = df.loc[df['Label'] == i]
                df_dataset_i = df_dataset_i[:size_train]

                frames = [df_dataset, df_dataset_i]
                df_dataset = pd.concat(frames)        
        
        train_index = df_dataset['Index'].tolist()

        # shuffle data         
        random.seed(args.seed)
        random.shuffle(train_index)
        
        new_x_train, new_y_train = x_train[train_index], y_train[train_index]
        return (new_x_train, new_y_train), (x_test, y_test), train_index


def train(args, data):
    if args.d == "cifar":
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

    train_data, test_data = data
    x_train, y_train = train_data
    x_test, y_test = test_data

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
    filepath= "./model_tracking_ver3/%s_model_improvement-{epoch:02d}-{val_acc:.2f}.h5" % (args.d)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(
        x_train,
        y_train,
        epochs=1000,
        batch_size=64,
        shuffle=True,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, type=str)
    parser.add_argument("-p", required=True, type=float)
    parser.add_argument("--batch_size", "-batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("--seed", "-seed", help="Random seed", type=int, default=0)

    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"  

    cut_off_score(args)

    # train_data, test_data, _ = loading_data_for_training(args)
    # train(args, (train_data, test_data))