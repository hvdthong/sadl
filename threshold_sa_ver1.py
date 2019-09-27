import argparse
from threshold_sa import normalize_sa, get_correct_and_incorrect_instance, func_precision, func_recall
from utils import load_file
from keras.datasets import mnist, cifar10
from keras import utils
from keras.models import load_model
import numpy as np
import random
from sklearn.metrics import roc_auc_score


CLIP_MIN = -0.5
CLIP_MAX = 0.5


def random_selected(y, times):
    pos_index = [index for index in range(len(y)) if y[index] == 1]
    neg_index = [index for index in range(len(y)) if y[index] == 0]

    random_pos, random_neg = list(), list()
    for t in range(times):
        np.random.seed(t)
        pos = np.random.choice(pos_index, len(neg_index), replace=False)
        random_pos.append(list(pos))
        random_neg.append(list(neg_index))    
    return random_pos, random_neg


def select_best_result(y_score, y_true, thresholds):
    prcs, rcs, perfs, new_threshold = list(), list(), list(), list()
    for t in thresholds:
        y_pred = [0 if score >= t else 1 for score in y_score]  # note that we focus on the surprise adequacy score, higher surprise adequacy score => incorrect instance
        prc = func_precision(y_true=y_true, y_pred=y_pred)
        rc = func_recall(y_true=y_true, y_pred=y_pred)
        if (prc is not None) and (rc is not None):
            if (prc + rc) != 0:
                prcs.append(prc)
                rcs.append(rc)
                perfs.append(2 * prc * rc / (prc + rc))    
                new_threshold.append(t)
    return prcs, rcs, perfs, new_threshold


def find_best_threshold(y_score, y_pred, y_true, thresholds, times):
    true_score = get_correct_and_incorrect_instance(y_pred=y_pred, y_true=y_true)  # ground truth of correct and incorrect instances
    random_pos, random_neg = random_selected(y=true_score, times=times)
    values_threshold, values_perfs = list(), list()
    for t in range(times):
        pos_index, neg_index = random_pos[t], random_neg[t]
        all_index = pos_index + neg_index
        time_sa_score = [y_score[index] for index in all_index]
        time_true_score = [true_score[index] for index in all_index]
        prcs, rcs, perfs, new_threshold = select_best_result(y_score=time_sa_score, y_true=time_true_score, thresholds=thresholds)
        max_index = perfs.index(max(perfs))
        values_threshold.append(new_threshold[max_index])
        values_perfs.append(perfs[max_index])
    return values_threshold, values_perfs


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

    thresholds = [i / float(1000) for i in range(1, 1000, 1)]  # define the number of threshold
    times = 10  # define the repeated times 

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
        model = load_model('./model_tracking/model_improvement-04-0.99_mnist.h5')
        model.summary()
        y_pred = model.predict(x_test)

        if args.lsa:
            score_path_file = './sa/lsa_mnist.txt'
            type_sa = 'lsa'
        
        if args.dsa:
            score_path_file = './sa/dsa_mnist.txt'
            type_sa = 'dsa'

        sa_score = normalize_sa(load_file(score_path_file))
        values_threshold, values_perfs = find_best_threshold(y_score=sa_score, y_pred=y_pred, y_true=y_test, thresholds=thresholds, times=times)
        print(values_threshold)
        print(values_perfs)

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

        model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
        model.summary()
        y_pred = model.predict(x_test)

        if args.lsa:
            score_path_file = './sa/lsa_cifar.txt'
            type_sa = 'lsa'
        
        if args.dsa:
            score_path_file = './sa/dsa_cifar.txt'
            type_sa = 'dsa'

        sa_score = normalize_sa(load_file(score_path_file))
        values_threshold, values_perfs = find_best_threshold(y_score=sa_score, y_pred=y_pred, y_true=y_test, thresholds=thresholds, times=times)
        print(values_threshold)
        print(values_perfs)
