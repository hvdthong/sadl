import argparse
from keras.datasets import mnist, cifar10
from keras.models import load_model
from box_plot import mnist_get_correct_and_incorrect_test_images
from keras import utils
import numpy as np
import matplotlib.pyplot as plt
from utils import load_file, write_file

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")

    args = parser.parse_args()
    print(args)    

    if args.d == "mnist":
        lsa, dsa = load_file('./sa/lsa_%s.txt' % args.d), load_file('./sa/dsa_%s.txt' % args.d)                

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # number of class
        num_class = 10
        y_train = utils.to_categorical(y_train, num_class)
        y_test = utils.to_categorical(y_test, num_class)
        
        x_test_normalize = x_test.astype("float32")
        x_test_normalize = (x_test_normalize / 255.0) - (1.0 - CLIP_MAX)

        # Load pre-trained model.            
        model = load_model('./model_tracking/model_improvement-04-0.99_mnist.h5')
        model.summary()

        y_pred = model.predict(x_test_normalize)
        correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred, y_true=y_test)
        print(len(correct), len(incorrect), len(correct + incorrect))

        incorrect_info = list()

        for index in incorrect:
            image, pred, true = x_test[index], y_pred[index], y_test[index]
            image = image.reshape(28, 28)
            pred = np.argmax(pred)
            true = np.argmax(true)
            pred_score, lsa_score, dsa_score = np.amax(y_pred[index]), lsa[index], dsa[index]             

            # Plot
            plt.title('Label: %s -- Predict: %s' % (str(true), str(pred)))
            plt.imshow(image, cmap='gray')
            plt.savefig('./figures_QA/%s/%i_%.2f.jpg' % (args.d, index, pred_score))
            print('./figures_QA/%s/%i_%.2f.jpg' % (args.d, index, pred_score))
            
            incorrect_info.append('%s \t %s \t %s \t %s' % (index, pred_score, lsa_score, dsa_score))
        write_file('./figures_QA/incorrect_info_%s' % (args.d), data=incorrect_info)

    elif args.d == 'cifar':
        lsa, dsa = load_file('./sa/lsa_%s.txt' % args.d), load_file('./sa/dsa_%s.txt' % args.d)                

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test_normalize = x_test.astype("float32")
        x_test_normalize = (x_test_normalize / 255.0) - (1.0 - CLIP_MAX)

        # number of class
        num_class = 10
        y_train = utils.to_categorical(y_train, num_class)
        y_test = utils.to_categorical(y_test, num_class)                

        # Load pre-trained model.            
        model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
        model.summary()

        y_pred = model.predict(x_test_normalize)
        correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred, y_true=y_test)
        print(len(correct), len(incorrect), len(correct + incorrect))        
        incorrect_info = list()

        # for index in incorrect:
        #     image, pred, true = x_test[index], y_pred[index], y_test[index]            
        #     pred = np.argmax(pred)
        #     true = np.argmax(true)
        #     pred_score, lsa_score, dsa_score = np.amax(y_pred[index]), lsa[index], dsa[index]             

            
        #     # Plot
        #     plt.title('Label: %s -- Predict: %s' % (str(true), str(pred)))
        #     plt.imshow(image, cmap='gray')
        #     plt.savefig('./figures_QA/%s/%i_%.2f.jpg' % (args.d, index, pred_score))
        #     print('./figures_QA/%s/%i_%.2f.jpg' % (args.d, index, pred_score))
            
        #     incorrect_info.append('%s \t %s \t %s \t %s' % (index, pred_score, lsa_score, dsa_score))
        # write_file('./figures_QA/incorrect_info_%s' % (args.d), data=incorrect_info)

        correct_info = list()
        for index in correct:
            image, pred, true = x_test[index], y_pred[index], y_test[index]            
            pred = np.argmax(pred)
            true = np.argmax(true)
            pred_score, lsa_score, dsa_score = np.amax(y_pred[index]), lsa[index], dsa[index]             
            
            correct_info.append('%s \t %s \t %s \t %s' % (index, pred_score, lsa_score, dsa_score))
        write_file('./figures_QA/correct_info_%s' % (args.d), data=correct_info)
