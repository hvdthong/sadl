import argparse
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from box_plot import mnist_get_correct_and_incorrect_test_images
from keras import utils
import numpy as np
from sklearn.manifold import TSNE
from utils import load_file
import matplotlib.pyplot as plt
import matplotlib

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
    dict_color_correct = {0: 'bisque', 1: 'silver', 2: 'lightcoral', 3: 'sandybrown', 
        4: 'khaki', 5: 'darkkhaki', 6: 'palegreen', 7: 'lightcyan', 8: 'skyblue', 9: 'cornflowerblue'}

    dict_incolor_correct = {0: 'tab:orange', 1: 'gray', 2: 'brown', 3: 'peru', 
        4: 'gold', 5: 'olive', 6: 'forestgreen', 7: 'lightseagreen', 8: 'steelblue', 9: 'royalblue'}
    dict_label = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
    if args.d == 'cifar':
        # (x_train, y_train), (x_test, y_test) = cifar10.load_data()

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

        # layer_names = ["activation_11"]

        # score = model.evaluate(x_test, y_test, verbose=1)  
        # y_pred = model.predict(x_test)

        # correct, incorrect = mnist_get_correct_and_incorrect_test_images(y_pred=y_pred, y_true=y_test)

        # temp_model = Model(
        #     inputs=model.input,
        #     outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
        # )

        # outputs = temp_model.predict(x_test, batch_size=args.batch_size, verbose=1)
        # # print(args)
        # # print(outputs.shape)
        # # print(type(outputs))
        # np.save('./feature_based_analysis/ftr_based', outputs)


        # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # ftr_based = np.load('./feature_based_analysis/ftr_based.npy')
        # ftr_based_embedded = TSNE(n_components=2).fit_transform(ftr_based)
        # np.save('./feature_based_analysis/%s_ftr_based_embedded' % (args.d), ftr_based_embedded)
        # print('Saving embedding...')
        # exit()

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        ftr_based_embedded = np.load('./feature_based_analysis/%s_ftr_based_embedded.npy' % (args.d))

        correct = load_file('./figures_QA/correct_info_%s' % (args.d))
        correct = [int(s.split('\t')[0]) for s in correct]
        
        incorrect = load_file('./figures_QA/incorrect_info_%s' % (args.d))
        incorrect = [int(s.split('\t')[0]) for s in incorrect]
        
        y_test = y_test.ravel()
        label = sorted(list(set(list(y_test))))

        # Shrink current axis's height by 10% on the bottom
        # fig = plt.figure()
        # ax = plt.subplot()
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])        

        for l in label:
            for l_ in label:
                l_y_test = list()
                for i, y in enumerate(y_test):
                    if y == l_:
                        l_y_test.append(i)
                embeds = list()
                for i in l_y_test:
                    if i in correct:
                        embeds.append(ftr_based_embedded[i])
                embeds = np.array(embeds)
                color = dict_color_correct[l_]
                plt.scatter(embeds[:, 0], embeds[:, 1], c=color, marker='o', s=7.5, label='Correct: %s' % (dict_label[l_]))

            l_y_test = list()
            for i, y in enumerate(y_test):
                if y == l:
                    l_y_test.append(i)
            embeds = list()
            for i in l_y_test:
                if i in incorrect:
                    embeds.append(ftr_based_embedded[i])
            embeds = np.array(embeds)
            color = dict_incolor_correct[l]
            plt.scatter(embeds[:, 0], embeds[:, 1], c=color, marker='o', label='Incorrect: %s' % (dict_label[l]))
            plt.subplots_adjust(right=1)
            plt.tight_layout(rect=[0,0,0.75,1])
            plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(8, 7.5)
            plt.title('Visual incorrect instances: %s vs. all correct instances' % dict_label[l])
            plt.savefig('./feature_based_analysis/%s_%s.jpg' % (args.d, str(l)))
            plt.close()
            print('Visual incorrect instances: %s vs. all correct instances' % dict_label[l])