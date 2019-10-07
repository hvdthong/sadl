from box_plot import mnist_get_correct_and_incorrect_test_images
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
import argparse
from keras import utils
import numpy as np
from utils import load_file, load_all_files
import matplotlib.pyplot as plt
import numpy as np

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )

    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"

    batch_size = 64

    if args.d == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        x_train = x_train.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = x_test.astype("float32")
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        # number of class
        num_class = 10
        y_train = utils.to_categorical(y_train, num_class)
        y_test = utils.to_categorical(y_test, num_class)

        model = load_model('./model_tracking/model_improvement-04-0.99_mnist.h5')
        model.summary()

        layer_names = ['dense_2']

        temp_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
        )

        if len(layer_names) == 1:
            layer_outputs = [
                temp_model.predict(x_test, batch_size=batch_size, verbose=1)
            ]
        else:
            layer_outputs = temp_model.predict(
                x_test, batch_size=batch_size, verbose=1
            )

        print(len(layer_outputs[0].shape))
        print(layer_outputs[0].shape)
        print(layer_outputs[0][0])
        exit()

        y_pred = model.predict(x_test)
        print(y_pred)
        y_pred = np.around(y_pred)
        print(y_pred)
        exit()

