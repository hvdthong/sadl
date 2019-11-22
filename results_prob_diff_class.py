import argparse
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from box_plot import mnist_get_correct_and_incorrect_test_images
from keras import utils
from utils import *
import numpy as np
from sa import fetch_dsa, fetch_lsa

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--conf", "-conf", help="Confidence score", action="store_true"
    )
    parser.add_argument(
        "--conf_no_norm", "-conf_no_norm", help="Confidence score -- No normalization", action="store_true"
    )
    parser.add_argument(
        "--result", "-result", help="Correct and incorrect instances", action="store_true"
    )
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
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./results_pred_ver2/tmp/"
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=float,
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
    parser.add_argument(
        "--layer",
        "-layer",
        help="Layer name",
        type=str,
    )
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
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

        # model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
        model = load_model('./model_tracking_ver2/cifar_model_improvement-494-0.81.h5')
        model.summary()

        # evaluation
        # scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
        # print('Evaluation result -- Test: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
        # scores = model.evaluate(x_train, y_train, batch_size=128, verbose=1)
        # print('Evaluation result -- Train: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
        # exit()

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

            if args.conf:
                y_pred = model.predict(x_test)
                y_pred = list(np.amax(y_pred, axis=1))
                write_file(path_file='./results_pred_ver2/test_conf_{}.txt'.format(args.d), data=y_pred)
            
            if args.conf_no_norm:
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
                y_pred_no_norm = list(np.amax(y_pred_no_norm, axis=1))
                write_file(path_file='./results_pred_ver2/test_conf_no_norm_{}.txt'.format(args.d), data=y_pred_no_norm)


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
                
                # write_file(path_file='./results_pred/test_pred_label_{}.txt'.format(args.d), data=pred_label)
                # write_file(path_file='./results_pred/test_true_label_{}.txt'.format(args.d), data=true_label)

                write_file(path_file='./results_pred_ver2/test_pred_label_{}.txt'.format(args.d), data=pred_label)
                write_file(path_file='./results_pred_ver2/test_true_label_{}.txt'.format(args.d), data=true_label)

            if args.result:
                y_pred = model.predict(x_test)
                pred_label = list(np.argmax(y_pred, axis=1))
                true_label = list(np.argmax(y_test, axis=1))

                pred_label = [dict_label[p] for p in pred_label]
                true_label = [dict_label[p] for p in true_label]

                results = list()
                for p, t in zip(pred_label, true_label):
                    if p == t:
                        results.append('TRUE')
                    else:
                        results.append('FALSE')    
                write_file(path_file='./results_pred_ver2/test_result_{}.txt'.format(args.d), data=results)

            if args.lsa:
                layer_names = ["activation_11"]
                test_lsa = fetch_lsa(model, x_train, x_test, "test", layer_names, args)
                write_file(path_file='./results_pred_ver2/test_lsa_{}.txt'.format(args.d), data=test_lsa) 

            if args.dsa:
                layer_names = ["activation_11"]
                test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args) 
                write_file(path_file='./results_pred_ver2/test_dsa_{}.txt'.format(args.d), data=test_dsa)