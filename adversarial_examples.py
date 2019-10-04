import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from sa import fetch_dsa, fetch_lsa, get_sc
from utils import *

from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend

import keras
import tensorflow as tf

from cleverhans.dataset import MNIST
from utils import write_file

adv_mnist_test = np.load('./adv/adv_mnist_fgsm.npy')
print(adv_mnist_test.shape)
exit()

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
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"    

    # Force TensorFlow to use single thread to improve reproducibility
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    if keras.backend.image_data_format() != 'channels_last':
        raise NotImplementedError("this tutorial requires keras to be configured to channels_last format") 

    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    
    if args.d == "mnist":        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Load pre-trained model.        
        model = load_model('./model_tracking/model_improvement-04-0.99_mnist.h5')
        model.summary()

        wrap = KerasModelWrapper(model)

        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        y = tf.placeholder(tf.float32, shape=(None, 10))

        fgsm = FastGradientMethod(wrap, sess=sess)        
        fgsm_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.}        
        x_adv = fgsm.generate(x_test, **fgsm_params)        
        x_adv = tf.stop_gradient(x_adv)
        x_adv_prob = model(x_adv)
        fetches = [x_adv_prob]
        fetches.append(x_adv)
        outputs = session.run(fetches=fetches, feed_dict={x:x_test})

        adv_prob = outputs[0]
        adv_examples = outputs[1]

        write_file(path_file='./adv/adv_{}.txt'.format(args.d), data=adv_examples)

        adv_predicted = adv_prob.argmax(1)

        adv_accuracy = np.mean(adv_predicted == y_test)

        print("Adversarial accuracy: %.5f" % adv_accuracy)
        # print(len(adv_examples))
        # print(adv_examples.shape)

        # preds_adv = model(x_adv)
        # print(keras.metrics.categorical_accuracy(y_test, preds_adv))
        exit()

        # with sess.as_default():
        #     t = type(adv_x).eval()
        #     print(type(adv_x.eval()))
        #     exit()

        print(adv_x.shape)
        print(type(adv_x))
        print('hello')
        exit()

