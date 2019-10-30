import argparse
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from box_plot import mnist_get_correct_and_incorrect_test_images
from utils import write_file, load_file
from keras import utils
import matplotlib.pyplot as plt
import numpy as np 

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"    

    if args.d == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # x_train = x_train.astype("float32")
        # x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        # x_test = x_test.astype("float32")
        # x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        # # number of class
        # num_class = 10
        # y_train = utils.to_categorical(y_train, num_class)
        # y_test = utils.to_categorical(y_test, num_class)
        
        # model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
        # model.summary()

        # score = model.evaluate(x_test, y_test, verbose=1)  
        # y_pred = model.predict(x_test)

        # correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred, y_true=y_test)
        # results = list()
        # for index in range(0, 10000):
        #     if index in correct:
        #         results.append('TRUE')
        #     else:
        #         results.append('FALSE')
        # write_file('./sa/pred_results_%s.txt' % (args.d), results)


        # x_train = x_train.astype("float32")
        # x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        # x_test_normalize = x_test.astype("float32")
        # x_test_normalize = (x_test_normalize / 255.0) - (1.0 - CLIP_MAX)

        # # number of class
        # num_class = 10
        # y_train = utils.to_categorical(y_train, num_class)
        # y_test = utils.to_categorical(y_test, num_class)

        # model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
        # model.summary()

        # score = model.evaluate(x_test_normalize, y_test, verbose=1)  
        # y_pred = model.predict(x_test_normalize)

        # correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred, y_true=y_test)
        # print(len(correct), len(incorrect), len(correct + incorrect))     
        
        # preds, trues = list(), list()
        # for index in range(0, 10000):
        #     image, pred, true = x_test[index], y_pred[index], y_test[index]

        #     pred = np.argmax(pred)
        #     true = np.argmax(true)

        #     preds.append(pred)
        #     trues.append(true)

        # write_file('./figures_ripon/%s_pred.txt' % (args.d), preds)
        # write_file('./figures_ripon/%s_true.txt' % (args.d), trues)

        # if index >= 5320:
        #     image, pred, true = x_test[index], y_pred[index], y_test[index]

        #     pred = np.argmax(pred)
        #     true = np.argmax(true)

        pred = load_file('./figures_ripon/%s_pred.txt' % (args.d))
        true = load_file('./figures_ripon/%s_true.txt' % (args.d))

        dict_label = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

        for index in range(0, 10000):
            image, p, t = x_test[index], pred[index], true[index]

            label_p, label_t = dict_label[int(p)], dict_label[int(t)]

            # Plot
            plt.title('Label: %s -- Predict: %s' % (label_t, label_p))
            plt.imshow(image, cmap='gray')
            plt.savefig('./figures_ripon/%s/%i.jpg' % (args.d, index))
            print('./figures_ripon/%s/%i.jpg' % (args.d, index))
            plt.close()            

