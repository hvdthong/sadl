from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
import argparse
from keras import utils
import numpy as np
from utils import load_file, load_all_files
import matplotlib.pyplot as plt
import numpy as np
from box_plot import mnist_get_correct_and_incorrect_test_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--all", "-all", help="Using all the adversarial examples", action="store_true"
    )
    parser.add_argument(
        "--adv_good", "-adv_good", help="Using the good adversarial examples (i.e., the advs that can make DL models wrongly predict)", action="store_true"
    )
    parser.add_argument(
        "--prob", "-prob", help="Confidence score", action="store_true"
    )

    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"

    if args.d == "mnist":
        if args.all:
            dsa_test_score = './sa/dsa_mnist.txt'
            dsa_adv_score = './sa/dsa_adversarial_mnist.txt'

            dsa_test_score = load_file(dsa_test_score)
            dsa_adv_score = load_file(dsa_adv_score)

            dsa_test_score = [float(i) for i in dsa_test_score]
            dsa_adv_score = [float(i) for i in dsa_adv_score]
            
            data = [dsa_test_score, dsa_adv_score]
            fig, ax = plt.subplots()
            ax.set_title('All-DSA: Test vs Adversarial Examples of {} dataset'.format(args.d))
            ax.boxplot(data)
            ax.set_xticklabels(['Test', 'Adv'])
            plt.savefig('./figures/dsa_test_vs_adversarial_{}.jpg'.format(args.d))

        if args.adv_good:
            dsa_test_score = './sa/dsa_mnist.txt'
            dsa_adv_score = './sa/dsa_adversarial_mnist.txt'

            dsa_test_score = load_file(dsa_test_score)
            dsa_adv_score = load_file(dsa_adv_score)

            dsa_test_score = [float(i) for i in dsa_test_score]
            dsa_adv_score = [float(i) for i in dsa_adv_score]

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

            correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred_adv, y_true=y_test)
            
            dsa_test_score = [dsa_test_score[index] for index in incorrect]
            dsa_adv_score = [dsa_adv_score[index] for index in incorrect]

            data = [dsa_test_score, dsa_adv_score]
            fig, ax = plt.subplots()
            ax.set_title('Incorrect-DSA: Test vs Adversarial Examples of {} dataset'.format(args.d))
            ax.boxplot(data)
            ax.set_xticklabels(['Test', 'Adv'])
            plt.savefig('./figures/dsa_incorrect_test_vs_adversarial_{}.jpg'.format(args.d))

        if args.adv_good and args.prob:
            dsa_test_score = './sa/prob_mnist.txt'
            dsa_adv_score = './sa/prob_adv_mnist.txt'

            dsa_test_score = load_file(dsa_test_score)
            dsa_adv_score = load_file(dsa_adv_score)

            dsa_test_score = [float(i) for i in dsa_test_score]
            dsa_adv_score = [float(i) for i in dsa_adv_score]

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

            correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred_adv, y_true=y_test)
            
            dsa_test_score = [dsa_test_score[index] for index in incorrect]
            dsa_adv_score = [dsa_adv_score[index] for index in incorrect]

            data = [dsa_test_score, dsa_adv_score]
            fig, ax = plt.subplots()
            ax.set_title('Incorrect-Confidence Score: Test vs Adversarial Examples of {} dataset'.format(args.d))
            ax.boxplot(data)
            ax.set_xticklabels(['Test', 'Adv'])
            plt.savefig('./figures/prob_incorrect_test_vs_adversarial_{}.jpg'.format(args.d))

    if args.d == "cifar":
        if args.all:
            dsa_test_score = './sa/dsa_cifar.txt'
            dsa_adv_score = './sa/dsa_adversarial_cifar.txt'

            dsa_test_score = load_file(dsa_test_score)
            dsa_adv_score = load_file(dsa_adv_score)

            dsa_test_score = [float(i) for i in dsa_test_score]
            dsa_adv_score = [float(i) for i in dsa_adv_score]
            
            data = [dsa_test_score, dsa_adv_score]
            fig, ax = plt.subplots()
            ax.set_title('All-DSA: Test vs Adversarial Examples of {} dataset'.format(args.d))
            ax.boxplot(data)
            ax.set_xticklabels(['Test', 'Adv'])
            plt.savefig('./figures/dsa_test_vs_adversarial_{}.jpg'.format(args.d))

        if args.adv_good:
            dsa_test_score = './sa/dsa_cifar.txt'
            dsa_adv_score = './sa/dsa_adversarial_cifar.txt'

            dsa_test_score = load_file(dsa_test_score)
            dsa_adv_score = load_file(dsa_adv_score)

            dsa_test_score = [float(i) for i in dsa_test_score]
            dsa_adv_score = [float(i) for i in dsa_adv_score]

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

            correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred_adv, y_true=y_test)
            
            dsa_test_score = [dsa_test_score[index] for index in incorrect]
            dsa_adv_score = [dsa_adv_score[index] for index in incorrect]

            data = [dsa_test_score, dsa_adv_score]
            fig, ax = plt.subplots()
            ax.set_title('Incorrect-DSA: Test vs Adversarial Examples of {} dataset'.format(args.d))
            ax.boxplot(data)
            ax.set_xticklabels(['Test', 'Adv'])
            plt.savefig('./figures/dsa_incorrect_test_vs_adversarial_{}.jpg'.format(args.d))

        if args.adv_good and args.prob:
            dsa_test_score = './sa/prob_cifar.txt'
            dsa_adv_score = './sa/prob_adv_cifar.txt'

            dsa_test_score = load_file(dsa_test_score)
            dsa_adv_score = load_file(dsa_adv_score)

            dsa_test_score = [float(i) for i in dsa_test_score]
            dsa_adv_score = [float(i) for i in dsa_adv_score]

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

            correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred_adv, y_true=y_test)
            
            dsa_test_score = [dsa_test_score[index] for index in incorrect]
            dsa_adv_score = [dsa_adv_score[index] for index in incorrect]

            data = [dsa_test_score, dsa_adv_score]
            fig, ax = plt.subplots()
            ax.set_title('Incorrect-Confidence Score: Test vs Adversarial Examples of {} dataset'.format(args.d))
            ax.boxplot(data)
            ax.set_xticklabels(['Test', 'Adv'])
            plt.savefig('./figures/prob_incorrect_test_vs_adversarial_{}.jpg'.format(args.d))