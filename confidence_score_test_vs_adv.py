from box_plot import mnist_get_correct_and_incorrect_test_images
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
import argparse
from keras import utils
import numpy as np
from utils import load_file, load_all_files
import matplotlib.pyplot as plt
import numpy as np
from utils import write_file

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--prob", "-prob", help="Confidence score", action="store_true"
    )   
    parser.add_argument(
        "--adv", "-adv", help="Calculate the confidence score of adversarial examples ", action="store_true"
    )    

    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"

    batch_size = 64

    if args.d == "mnist":
        if args.adv:
            # load adversarial 
            x_adv = np.load('./adv/adv_mnist.npy')            

            model = load_model('./model_tracking/model_improvement-04-0.99_mnist.h5')
            model.summary()

            layer_names = ['dense_2']

            temp_model = Model(
                inputs=model.input,
                outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
            )

            if len(layer_names) == 1:
                layer_outputs = [
                    temp_model.predict(x_adv, batch_size=batch_size, verbose=1)
                ]
            else:
                layer_outputs = temp_model.predict(
                    x_adv, batch_size=batch_size, verbose=1
                )

            output = layer_outputs[0]                
            output = list(np.amax(output, axis=1))
            write_file('./sa/prob_adv_no_normalize_%s.txt' % (args.d), output)
        
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

            output = layer_outputs[0]                
            output = list(np.amax(output, axis=1))
            write_file('./sa/prob_no_normalize_%s.txt' % (args.d), output)

    if args.d == "cifar":
        if args.adv:
            # load adversarial 
            x_adv = np.load('./adv/adv_cifar.npy')

            x_train = x_train.astype("float32")
            x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
            x_test = x_test.astype("float32")
            x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)       

            # Load pre-trained model.            
            model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
            model.summary()

            layer_names = ['dense_2']

            temp_model = Model(
                inputs=model.input,
                outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
            )

            if len(layer_names) == 1:
                layer_outputs = [
                    temp_model.predict(x_adv, batch_size=batch_size, verbose=1)
                ]
            else:
                layer_outputs = temp_model.predict(
                    x_adv, batch_size=batch_size, verbose=1
                )

            output = layer_outputs[0]                
            output = list(np.amax(output, axis=1))
            write_file('./sa/prob_adv_no_normalize_%s.txt' % (args.d), output)

        else:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()           

            x_train = x_train.astype("float32")
            x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
            x_test = x_test.astype("float32")
            x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

            # Load pre-trained model.            
            model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
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

            output = layer_outputs[0]                
            output = list(np.amax(output, axis=1))
            write_file('./sa/prob_no_normalize_%s.txt' % (args.d), output)

