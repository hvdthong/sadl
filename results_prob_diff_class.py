import argparse
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from box_plot import mnist_get_correct_and_incorrect_test_images
from keras import utils
from utils import *
import numpy as np

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--pred", "-pred", help="Prediction Score of different classes", action="store_true"
    )
    parser.add_argument(
        "--pred_no_norm", "-pred_no_norm", help="Prediction Score -- No normalization", action="store_true"
    )
    parser.add_argument(
        "--label", "-label", help="Print the Predicted label and True label", action="store_true"
    )
    parser.add_argument(
        "--train", "-train", help="Prediction Score of all classes for Training Dataset", action="store_true"
    )
    parser.add_argument(
        "--test", "-test", help="Prediction Score of all classes for Testing Dataset", action="store_true"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=64
    )

    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"

    if args.d == 'cifar':
        dict_label = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
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

        if args.train == True:
            if args.pred:
                y_pred = model.predict(x_train)
                np.savetxt('./results_pred/train_pred_classes_{}.txt'.format(args.d), y_pred, delimiter='\t')                
            if args.pred_no_norm:
                layer_names = ['dense_2']

                temp_model = Model(
                    inputs=model.input,
                    outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
                )

                if len(layer_names) == 1:
                    layer_outputs = [
                        temp_model.predict(x_train, batch_size=args.batch_size, verbose=1)
                    ]
                else:
                    layer_outputs = temp_model.predict(
                        x_train, batch_size=args.batch_size, verbose=1
                    )

                y_pred_no_norm = layer_outputs[0]
                np.savetxt('./results_pred/train_pred_no_norm_classes_{}.txt'.format(args.d), y_pred_no_norm, delimiter='\t')
            if args.label:
                y_pred = model.predict(x_train)
                pred_label = list(np.argmax(y_pred, axis=1))
                true_label = list(np.argmax(y_train, axis=1))

                pred_label = [dict_label[p] for p in pred_label]
                true_label = [dict_label[p] for p in true_label]
                
                write_file(path_file='./results_pred/train_pred_label_{}.txt'.format(args.d), data=pred_label)
                write_file(path_file='./results_pred/train_true_label_{}.txt'.format(args.d), data=true_label)

        if args.test == True:
            if args.pred:
                y_pred = model.predict(x_test)
                np.savetxt('./results_pred/test_pred_classes_{}.txt'.format(args.d), y_pred, delimiter='\t')                
            if args.pred_no_norm:
                layer_names = ['dense_2']

                temp_model = Model(
                    inputs=model.input,
                    outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
                )

                if len(layer_names) == 1:
                    layer_outputs = [temp_model.predict(x_test, batch_size=args.batch_size, verbose=1)]
                else:
                    layer_outputs = temp_model.predict(x_test, batch_size=args.batch_size, verbose=1)

                y_pred_no_norm = layer_outputs[0]
                np.savetxt('./results_pred/test_pred_no_norm_classes_{}.txt'.format(args.d), y_pred_no_norm, delimiter='\t')
            if args.label:
                y_pred = model.predict(x_test)
                pred_label = list(np.argmax(y_pred, axis=1))
                true_label = list(np.argmax(y_test, axis=1))

                pred_label = [dict_label[p] for p in pred_label]
                true_label = [dict_label[p] for p in true_label]
                
                write_file(path_file='./results_pred/test_pred_label_{}.txt'.format(args.d), data=pred_label)
                write_file(path_file='./results_pred/test_true_label_{}.txt'.format(args.d), data=true_label)
