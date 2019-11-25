import argparse
from keras.datasets import mnist, cifar10
from utils import write_file
from keras import utils
from keras.models import load_model, Model
import numpy as np
from utils import load_file
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from threshold_sa import get_correct_and_incorrect_instance, normalize_sa 

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")

    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"    
    print(args)
    
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

        # # # # evaluate
        # # scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
        # # print('\nEvaluation result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
        # # exit()

        y_pred = model.predict(x_test)
        true_score = get_correct_and_incorrect_instance(y_pred=y_pred, y_true=y_test)   
        
        y_conf = np.amax(y_pred, axis=1)
        fpr_conf, tpr_conf, _ = roc_curve(true_score, y_conf)
        roc_auc_conf = auc(fpr_conf, tpr_conf)

        y_lsa = load_file('./sa/lsa_%s.txt' % (args.d))        
        y_lsa = [-float(s) for s in y_lsa]
        fpr_lsa, tpr_lsa, _ = roc_curve(true_score, y_lsa)
        roc_auc_lsa = auc(fpr_lsa, tpr_lsa)

        y_dsa = load_file('./sa/dsa_%s.txt' % (args.d))        
        y_dsa = [-float(s) for s in y_dsa]
        fpr_dsa, tpr_dsa, _ = roc_curve(true_score, y_dsa)
        roc_auc_dsa = auc(fpr_dsa, tpr_dsa)

        # method I: plt
        import matplotlib.pyplot as plt
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr_conf, tpr_conf, 'b', label = 'AUC_conf = %0.2f' % roc_auc_conf)
        plt.plot(fpr_lsa, tpr_lsa, 'c', label = 'AUC_lsa = %0.2f' % roc_auc_lsa)
        plt.plot(fpr_dsa, tpr_dsa, 'g', label = 'AUC_dsa = %0.2f' % roc_auc_dsa)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('roc_curve_%s.jpg' % (args.d))


        # # calculate AUC
        # auc = roc_auc_score(true_score, y_score)
        # print('AUC score of %s dataset' % (args.d))
        
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(num_class):
        #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])

        # # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # plt.figure()
        # lw = 2
        # plt.plot(fpr[2], tpr[2], color='darkorange',
        #         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic of confidence score')
        # plt.legend(loc="lower right")
        # plt.savefig('roc_curve_%s.jpg' % (args.d))        
        # plt.close()

        

        