import argparse
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from box_plot import mnist_get_correct_and_incorrect_test_images
from keras import utils
from sa import fetch_dsa, fetch_lsa
from utils import *

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
        "--conf", "-conf", help="Confidence Score", action="store_true"
    )
    parser.add_argument(
        "--conf_no_norm", "-conf_no_norm", help="Confidence Score -- No normalization", action="store_true"
    )
    parser.add_argument(
        "--result", "-result", help="Print True/False of all instances", action="store_true"
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./results_training_data/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=64
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=float,
        default=1e-5,
        # default=0.01,
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
    parser.add_argument(
        "--layer",
        "-layer",
        help="Layer name",
        type=str,
    )

    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"

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

        y_pred_train = model.predict(x_train)

        correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred_train, y_true=y_train)

        if args.lsa == True:
            layer_names = ["activation_11"]
            lsa = fetch_lsa(model, x_train, x_train, "train", layer_names, args)
            write_file(path_file='./results_training_data/lsa_{}.txt'.format(args.d), data=lsa)

        if args.dsa == True:
            layer_names = ["activation_11"]
            dsa = fetch_dsa(model, x_train, x_train, "train", layer_names, args)
            write_file(path_file='./results_training_data/dsa_{}.txt'.format(args.d), data=dsa)
        
        if args.conf == True:            
            conf_score = list(np.amax(y_pred_train, axis=1))
            write_file(path_file='./results_training_data/conf_{}.txt'.format(args.d), data=conf_score)
        
        if args.conf_no_norm == True:
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

            conf_score_no_norm = layer_outputs[0]                
            conf_score_no_norm = list(np.amax(conf_score_no_norm, axis=1))
            write_file(path_file='./results_training_data/conf_no_norm_{}.txt'.format(args.d), data=conf_score_no_norm)
        
        if args.result == True:
            row, _ = y_train.shape            
            results = list()
            for i in range(0, row):
                if i in incorrect:
                    results.append('FALSE')
                else:
                    results.append('TRUE')
            write_file(path_file='./results_training_data/results_{}.txt'.format(args.d), data=results)

        print(len(correct), len(incorrect))
        print('Accuracy of %s dataset is: ', len(correct) / (len(correct) + len(incorrect)))