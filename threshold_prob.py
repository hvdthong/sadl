import argparse
from threshold_sa import normalize_sa, get_correct_and_incorrect_instance, func_precision, func_recall
from utils import load_file, write_file
from keras.datasets import mnist, cifar10
from keras import utils
from keras.models import load_model
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from threshold_sa_ver1 import random_selected, random_selected_adversarial

CLIP_MIN = -0.5
CLIP_MAX = 0.5

def select_best_result(y_score, y_true, thresholds):
    prcs, rcs, perfs, new_threshold = list(), list(), list(), list()
    for t in thresholds:
        y_pred = [1 if score >= t else 0 for score in y_score]  # note that we focus on the probability score, higher probability score => correct instance
        prc = func_precision(y_true=y_true, y_pred=y_pred)
        rc = func_recall(y_true=y_true, y_pred=y_pred)
        if (prc is not None) and (rc is not None):
            if (prc + rc) != 0:
                prcs.append(prc)
                rcs.append(rc)
                perfs.append(2 * prc * rc / (prc + rc))    
                new_threshold.append(t)
    return prcs, rcs, perfs, new_threshold

def find_best_threshold(y_pred, y_true, thresholds, times):
    true_score = get_correct_and_incorrect_instance(y_pred=y_pred, y_true=y_true)  # ground truth of correct and incorrect instances
    pred_score = list(np.amax(y_pred, axis=1))

    random_pos, random_neg = random_selected(y=true_score, times=times)
    values_threshold, values_perfs = list(), list()
    for t in range(times):
        pos_index, neg_index = random_pos[t], random_neg[t]
        all_index = pos_index + neg_index
        time_pred_score = [pred_score[index] for index in all_index]
        time_true_score = [true_score[index] for index in all_index]
        prcs, rcs, perfs, new_threshold = select_best_result(y_score=time_pred_score, y_true=time_true_score, thresholds=thresholds)
        max_index = perfs.index(max(perfs))
        values_threshold.append(new_threshold[max_index])
        values_perfs.append(perfs[max_index])
    return values_threshold, values_perfs

def find_best_threshold_advesarial(y_pred, y_true, thresholds, times, num_instances):
    true_score = get_correct_and_incorrect_instance(y_pred=y_pred, y_true=y_true)  # ground truth of correct and incorrect instances
    pred_score = list(np.amax(y_pred, axis=1))

    random_pos, random_neg = random_selected_adversarial(y=true_score, times=times, num_instances=num_instances)
    values_threshold, values_perfs = list(), list()
    for t in range(times):
        pos_index, neg_index = random_pos[t], random_neg[t]
        all_index = pos_index + neg_index
        time_pred_score = [pred_score[index] for index in all_index]
        time_true_score = [true_score[index] for index in all_index]
        prcs, rcs, perfs, new_threshold = select_best_result(y_score=time_pred_score, y_true=time_true_score, thresholds=thresholds)
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
    parser.add_argument(
        "--write", "-write", help="Write the Probability Score", action="store_true"
    )
    parser.add_argument(
        "--adv", "-adv", help="Write the Probability Score of adversarial examples", action="store_true"
    )

    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"

    thresholds = [i / float(1000) for i in range(1, 1000, 1)]  # define the number of threshold
    times = 10  # define the repeated times 

    if args.adv:
        num_instances = 500

    if args.d == "mnist":
        if args.adv: 
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            # number of class
            num_class = 10
            y_train = utils.to_categorical(y_train, num_class)
            y_test = utils.to_categorical(y_test, num_class)

            # load adversarial 
            x_adv = np.load('./adv/adv_mnist.npy')

            # Load pre-trained model.            
            model = load_model('./model_tracking/model_improvement-04-0.99_mnist.h5')
            model.summary()

            y_pred_adv = model.predict(x_adv)

            if args.write == True:
                print('Writing the probabilty score of adversarial examples of %s dataset' % (args.d))            
                max_prob = list(np.amax(y_pred_adv, axis=1))
                write_file(path_file='./sa/prob_adv_%s.txt' % (args.d), data=max_prob)
                exit()

            values_threshold, values_perfs = find_best_threshold_advesarial(y_pred=y_pred_adv, y_true=y_test, thresholds=thresholds, times=times, num_instances=num_instances)
            print(values_threshold)
            print(values_perfs)
        else:
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

            if args.write == True:
                print('Writing the probabilty score of %s dataset' % (args.d))            
                max_prob = list(np.amax(y_pred, axis=1))            
                write_file(path_file='./sa/prob_%s.txt' % (args.d), data=max_prob)
                exit()

            values_threshold, values_perfs = find_best_threshold(y_pred=y_pred, y_true=y_test, thresholds=thresholds, times=times)
            print(values_threshold)
            print(values_perfs)

    if args.d == 'cifar':
        if args.adv: 
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()            
            # number of class
            num_class = 10
            y_train = utils.to_categorical(y_train, num_class)
            y_test = utils.to_categorical(y_test, num_class)

            # load adversarial 
            x_adv = np.load('./adv/adv_cifar.npy')

            # Load pre-trained model.            
            model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
            model.summary()

            y_pred_adv = model.predict(x_adv)

            if args.write == True:
                print('Writing the probabilty score of adversarial examples of %s dataset' % (args.d))            
                max_prob = list(np.amax(y_pred_adv, axis=1))
                write_file(path_file='./sa/prob_adv_%s.txt' % (args.d), data=max_prob)
                exit()

            values_threshold, values_perfs = find_best_threshold_advesarial(y_pred=y_pred_adv, y_true=y_test, thresholds=thresholds, times=times, num_instances=num_instances)
            print(values_threshold)
            print(values_perfs)
        else:
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

            if args.write == True:
                print('Writing the probabilty score of %s dataset' % (args.d))            
                max_prob = list(np.amax(y_pred, axis=1))            
                write_file(path_file='./sa/prob_%s.txt' % (args.d), data=max_prob)
                exit()

            values_threshold, values_perfs = find_best_threshold(y_pred=y_pred, y_true=y_test, thresholds=thresholds, times=times)
            print(values_threshold)
            print(values_perfs)

