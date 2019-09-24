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

def load_models(path_folder, dataset):
    # return the model name depending on which dataset
    all_models = load_all_files(path_folder=path_folder)    
    data_models = [path_folder + model for model in all_models if dataset in model]
    return data_models


def get_best_models(models, x_test, y_test):
    accuracy = list()
    for path_model in models:
        model = load_model(path_model)
        score = model.evaluate(x_test, y_test, verbose=0)
        accuracy.append(score[1])
        print(path_model, score[1])
    max_index = accuracy.index(max(accuracy))
    return models[max_index]


def get_best_models_cifar(models, x_test, y_test):
    models = sorted(models)
    accuracy = list()
    for path_model in models:
        model = load_model(path_model)
        y_pred = model.predict(x_test)
        score = model.evaluate(x_test, y_test, verbose=0)
        accuracy.append(score[1])
        print(path_model, score[1])
    max_index = accuracy.index(max(accuracy))
    return models[max_index]


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

        # Load pre-trained model.
        path_models = load_models(path_folder='./model_tracking/', dataset=args.d)
        path_best_model = get_best_models(models=path_models, x_test=x_test, y_test=y_test)                
        model = load_model(path_best_model)
        score = model.evaluate(x_test, y_test, verbose=0)        

        y_pred = model.predict(x_test)
        y_pred = np.around(y_pred)

        correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred, y_true=y_test)        
        print('Best accuracy for %s: %.3f' %(args.d, score[1]))
        print(len(correct), len(incorrect))
        print(path_best_model)
        exit()

    if args.d == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = x_test.astype("float32")
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        # number of class
        num_class = 10
        y_train = utils.to_categorical(y_train, num_class)
        y_test = utils.to_categorical(y_test, num_class)

        # Load pre-trained model.
        path_models = load_models(path_folder='./model_tracking/', dataset=args.d)
        path_best_model = get_best_models(models=path_models, x_test=x_test, y_test=y_test)       
        model = load_model(path_best_model)
        score = model.evaluate(x_test, y_test, verbose=0)

        y_pred = model.predict(x_test)
        y_pred = np.around(y_pred)

        correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred, y_true=y_test)        
        print('Best accuracy for %s: %.3f' %(args.d, score[1]))
        print(len(correct), len(incorrect))
        print(path_best_model)
        exit()
