import argparse
from utils import load_file
from keras.datasets import mnist, cifar10
from keras import utils
from keras.models import load_model
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics

CLIP_MIN = -0.5
CLIP_MAX = 0.5


def func_precision(y_true, y_pred):
    tp, fp = 0, 0
    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            fp += 1    
    if tp + fp == 0:
        return None
    return tp / (tp + fp)


def func_recall(y_true, y_pred):
    tp, fn = 0, 0
    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
    if tp + fn == 0:
        return None
    return tp / (tp + fn)

def func_fp(y_true, y_pred):
    fp = 0
    for i in range(len(y_true)):        
        if y_pred[i] == 1 and y_true[i] == 0:
            fp += 1    
    if fp == 0:
        return None
    return fp

def func_fn(y_true, y_pred):
    fn = 0
    for i in range(len(y_true)):
        if y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
    if fn == 0:
        return None
    return fn


def get_correct_and_incorrect_instance(y_pred, y_true):
    # return the list of 0 and 1 values. 0: instance which incorrectly predict; 1: instance which correctly predict
    nrows, _ = y_pred.shape
    ground_truth = list()    
    for row in range(nrows):        
        if np.argmax(y_pred[row, :]) == np.argmax(y_true[row, :]):
            ground_truth.append(1)
        else:
            ground_truth.append(0)        
    return ground_truth

def normalize_sa(score):
    score = [float(s) for s in score]
    return [float(s)/max(score) for s in score]

def find_best_threshold(y_score, y_true, threshold):
    prcs, rcs, perfs, new_threshold = list(), list(), list(), list()
    for t in threshold:
        y_pred = [0 if score >= t else 1 for score in y_score]
        prc = func_precision(y_true=y_true, y_pred=y_pred)
        rc = func_recall(y_true=y_true, y_pred=y_pred)        
        if (prc is not None) and (rc is not None):
            prcs.append(prc)
            rcs.append(rc)
            perfs.append(2 * prc * rc / (prc + rc))    
            new_threshold.append(t)
    return prcs, rcs, perfs, new_threshold

def find_best_threshold_reliable(y_score, y_true, threshold):
    # use other evaluation metrics to find the best threshold (focus on reliable and unreliable instance)
    fps, fns, perfs, new_threshold = list(), list(), list(), list()
    for t in threshold:
        y_pred = [0 if score >= t else 1 for score in y_score]
        fp = func_fp(y_true=y_true, y_pred=y_pred)
        fn = func_fn(y_true=y_true, y_pred=y_pred)        
        if (prc is not None) and (rc is not None):
            fps.append(fp)
            fns.append(fn)
            perfs.append((fp + fn) / len(y_pred))    
            new_threshold.append(t)
    return fps, fns, perfs, new_threshold


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

    threshold = [i / float(1000) for i in range(1, 1000, 1)]

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
        true_score = get_correct_and_incorrect_instance(y_pred=y_pred, y_true=y_test)

        fpr, tpr, threshold = metrics.roc_curve(true_score, sa_score)
        roc_auc = metrics.auc(fpr, tpr)

        # method I: plt
        import matplotlib.pyplot as plt
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('example.jpg')


        # calculate AUC
        auc = roc_auc_score(true_score, sa_score)
        print('AUC score of %s dataset and %s surprise adequacy: %.3f' % (args.d, type_sa, auc))
        
        prcs, rcs, perfs, perfs_threshold = find_best_threshold(y_score=sa_score, y_true=true_score, threshold=threshold)
        max_index = perfs.index(max(perfs))        
        print(perfs_threshold[max_index], perfs[max_index], prcs[max_index], rcs[max_index])
        exit()
        # print(threshold)
        # exit()
        # normalized_sa = preprocessing.normalize([x_array])
        # normalize_sa(score)
        # print(sa_score.shape)
        print(sa_score)
        print(len(sa_score))
        # print(len(sa_score), len(true_score))
        # print(sa_score)
        # print(true_score)