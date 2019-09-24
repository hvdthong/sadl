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

def mnist_get_correct_and_incorrect_test_images(y_pred, y_true):
    # return the index of correct and incorrect test images 
    nrows, _ = y_pred.shape
    correct, incorrect = list(), list()    
    for row in range(nrows):        
        if np.argmax(y_pred[row, :]) == np.argmax(y_true[row, :]):
            correct.append(row)
        else:
            incorrect.append(row)        
    return correct, incorrect


def get_sa_for_correct_and_incorrect(sa_score, correct_index, incorrect_index, dataset):
    sa_score = [float(score) for score in sa_score]  # convert to float number
    # sa_score = [float(s)/max(sa_score) for s in sa_score]  # normalize list of numbers 
    if dataset == 'mnist':
        sa_correct = [sa_score[index] for index in correct_index if sa_score[index] < 5000]
        sa_incorrect = [sa_score[index] for index in incorrect_index if sa_score[index] < 5000]
    
    if dataset == 'cifar':
        sa_correct = [sa_score[index] for index in correct_index if sa_score[index] < 20000]
        sa_incorrect = [sa_score[index] for index in incorrect_index if sa_score[index] < 20000]
    return np.array(sa_correct), np.array(sa_incorrect)


def draw_box_plot(sa_correct, sa_incorrect, dataset, type_sa):    
    data = [sa_correct, sa_incorrect]
    fig, ax = plt.subplots()
    ax.set_title('Suprise Adequacy Score of {} for {} dataset'.format(type_sa, dataset))
    ax.boxplot(data)
    ax.set_xticklabels(['Correct', 'Incorrect'])
    plt.savefig('./figures/{}_{}.jpg'.format(type_sa, dataset))


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
        # model = load_model("./model/model_mnist.h5")
        # model = load_model("./model/best_model_mnist.h5")
        # model = load_model('./model_tracking_ver1/model_improvement-47-0.99_mnist.h5')
        # model = load_model('./model_tracking/model_improvement-10-0.99_mnist.h5')
        model = load_model('./model_tracking/model_improvement-04-0.99_mnist.h5')
        model.summary()

        y_pred = model.predict(x_test)
        # y_pred = np.around(y_pred)

        correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred, y_true=y_test)

        if args.lsa:
            score_path_file = './sa/lsa_mnist.txt'
            type_sa = 'lsa'
        
        if args.dsa:
            score_path_file = './sa/dsa_mnist.txt'
            type_sa = 'dsa'
        
        sa_score = load_file(score_path_file)
        sa_correct, sa_incorrect = get_sa_for_correct_and_incorrect(sa_score=sa_score, correct_index=correct, incorrect_index=incorrect, dataset=args.d)
        print('Dataset: %s' % (args.d))
        print('Mean and std of correct instances for %s score: %.3f, %.3f' % (type_sa, np.mean(sa_correct), np.std(sa_correct)))
        print('Mean and std of incorrect instances for %s score: %.3f, %.3f' % (type_sa, np.mean(sa_incorrect), np.std(sa_incorrect)))
        print('Accuracy of cifar dataset: %.3f' % (len(sa_correct) / (len(sa_correct) + len(sa_incorrect))))
        draw_box_plot(sa_correct=sa_correct, sa_incorrect=sa_incorrect, dataset=args.d, type_sa=type_sa)
    
    if args.d == "cifar":
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

        score = model.evaluate(x_test, y_test, verbose=1)        

        y_pred = model.predict(x_test)        
        # y_pred = np.around(y_pred)
        correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred, y_true=y_test)        

        if args.lsa:
            score_path_file = './sa/lsa_cifar.txt'
            type_sa = 'lsa'
        
        if args.dsa:
            score_path_file = './sa/dsa_cifar.txt'
            type_sa = 'dsa'
        
        sa_score = load_file(score_path_file)
        sa_correct, sa_incorrect = get_sa_for_correct_and_incorrect(sa_score=sa_score, correct_index=correct, incorrect_index=incorrect, dataset=args.d)
        print('Dataset: %s' % (args.d))
        print('Mean and std of correct instances for %s score: %.3f, %.3f' % (type_sa, np.mean(sa_correct), np.std(sa_correct)))
        print('Mean and std of incorrect instances for %s score: %.3f, %.3f' % (type_sa, np.mean(sa_incorrect), np.std(sa_incorrect)))
        print('Accuracy of cifar dataset: %.3f' % (len(sa_correct) / (len(sa_correct) + len(sa_incorrect))))
        draw_box_plot(sa_correct=sa_correct, sa_incorrect=sa_incorrect, dataset=args.d, type_sa=type_sa)