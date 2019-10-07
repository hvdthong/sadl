from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import keras
from keras import backend
from keras.models import load_model
import tensorflow as tf
# from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper
from matplotlib import pyplot as plt
import imageio
from keras.datasets import mnist, cifar10
import argparse
from keras import utils
from attacks import fast_gradient_sign_method
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod


CLIP_MIN = -0.5
CLIP_MAX = 0.5

def stitch_images(images, y_img_count, x_img_count, margin = 2):
    
    # Dimensions of the images
    img_width = images[0].shape[0]
    img_height = images[0].shape[1]
    
    width = y_img_count * img_width + (y_img_count - 1) * margin
    height = x_img_count * img_height + (x_img_count - 1) * margin
    stitched_images = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(y_img_count):
        for j in range(x_img_count):
            img = images[i * x_img_count + j]
            if len(img.shape) == 2:
                img = np.dstack([img] * 3)
            stitched_images[(img_width + margin) * i: (img_width + margin) * i + img_width,
                            (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    return stitched_images

def stitch_images_cifar(images, y_img_count, x_img_count, margin = 2):
    # Dimensions of the images
    img_width = images[0].shape[0]
    img_height = images[0].shape[1]
    
    width = y_img_count * img_width + (y_img_count - 1) * margin
    height = x_img_count * img_height + (x_img_count - 1) * margin
    stitched_images_cifar = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(y_img_count):
        for j in range(x_img_count):
            img = images[i * x_img_count + j]
            if len(img.shape) == 2:
                img = np.dstack([img] * 3)
            stitched_images_cifar[(img_width + margin) * i: (img_width + margin) * i + img_width, (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    return stitched_images

def cifar10_plot(image, title, path_save):    
    img = image

    # image = image.reshape(-1)
    # im = image

    # im_r = im[0:1024].reshape(32, 32)
    # im_g = im[1024:2048].reshape(32, 32)
    # im_b = im[2048:].reshape(32, 32)

    # img = np.dstack((im_r, im_g, im_b))
    
    plt.imshow(img)     
    plt.suptitle(title)
    plt.savefig(path_save)
    print(path_save)
    print('===> Saving image')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")

    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"

    if args.d == "mnist":
        # USING cleverhans library
        #####################################################################################
        #####################################################################################
        # # Set the learning phase to false, the model is pre-trained.
        # backend.set_learning_phase(False)
        # keras_model = load_model('./model_tracking/model_improvement-04-0.99_mnist.h5')

        # (x_train, y_train), (x_test, y_test) = mnist.load_data()        
        # x_train = x_train.reshape(-1, 28, 28, 1)
        # x_validation = x_test.reshape(-1, 28, 28, 1)
        # y_validation = y_test

        # if not hasattr(backend, "tf"):
        #     raise RuntimeError("This tutorial requires keras to be configured"
        #                     " to use the TensorFlow backend.")

        # if keras.backend.image_dim_ordering() != 'tf':
        #     keras.backend.set_image_dim_ordering('tf')
        #     print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
        #         "'th', temporarily setting to 'tf'")

        # # Retrieve the tensorflow session
        # sess =  backend.get_session()

        # # Evaluate the model's accuracy on the validation data used in training
        # x_validation = x_validation.astype('float32')
        # x_validation /= 255

        # pred = np.argmax(keras_model.predict(x_validation), axis = 1)
        # acc =  np.mean(np.equal(pred, y_validation))

        # print("The normal validation accuracy is: {}".format(acc))

        # # Initialize the Fast Gradient Sign Method (FGSM) attack object and 
        # # use it to create adversarial examples as numpy arrays.
        # wrap = KerasModelWrapper(keras_model)
        # fgsm = FastGradientMethod(wrap, sess=sess)
        # fgsm_params = {'eps': 0.3,
        #             'clip_min': 0.,
        #             'clip_max': 1.}
        # adv_x = fgsm.generate_np(x_validation, **fgsm_params)

        # adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
        # adv_acc =  np.mean(np.equal(adv_pred, y_validation))

        # print("The adversarial validation accuracy is: {}".format(adv_acc))
        # print(adv_x.shape)

        # np.save('./adv/adv_{}.npy'.format(args.d), adv_x)

        # x_sample = x_validation[0].reshape(28, 28)
        # adv_x_sample = adv_x[0].reshape(28, 28)

        # adv_comparison = stitch_images([x_sample, adv_x_sample], 1, 2)
        # plt.suptitle('TestPredicted: %s -- AdvPredicted: %s' % (str(pred[0]), str(adv_pred[0])))
        # plt.imshow(adv_comparison)
        # plt.savefig('./adv/adv_{}.jpg'.format(args.d))
        #####################################################################################
        #####################################################################################

        # Using https://github.com/IBM/adversarial-robustness-toolbox to create adversarial examples
        #####################################################################################
        #####################################################################################
        keras_model = load_model('./model_tracking/model_improvement-04-0.99_mnist.h5')
        classifier = KerasClassifier(model=keras_model, clip_values=(0, 255), use_logits=False)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(-1, 28, 28, 1)        

        # Evaluate the model's accuracy on the validation data used in training        
        x_test = x_test.astype("float32")
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        # Accuracy of our test data
        pred = np.argmax(classifier.predict(x_test), axis = 1)
        acc =  np.mean(np.equal(pred, y_test.reshape(-1)))

        print("The normal validation accuracy is: {}".format(acc))            

        attack = FastGradientMethod(classifier=classifier, eps=0.3, batch_size=64)
        x_test_adv = attack.generate(x=x_test)

        adv_pred = np.argmax(keras_model.predict(x_test_adv), axis = 1)
        adv_acc =  np.mean(np.equal(adv_pred, y_test))

        print("The adversarial validation accuracy is: {}".format(adv_acc))
        print(x_test_adv.shape)
        np.save('./adv/adv_{}.npy'.format(args.d), x_test_adv)
        
        # find the incorrrect instance
        incorrect_index = None
        for i in range(adv_pred.shape[0]):
            if pred[i] != adv_pred[i]:
                incorrect_index = i
                break

        x_sample = x_test[incorrect_index].reshape(28, 28)
        adv_x_sample = x_test_adv[incorrect_index].reshape(28, 28)

        adv_comparison = stitch_images([x_sample, adv_x_sample], 1, 2)
        plt.suptitle('TestPredicted: %s -- AdvPredicted: %s' % (str(pred[incorrect_index]), str(adv_pred[incorrect_index])))
        plt.imshow(adv_comparison)
        plt.savefig('./adv/adv_{}.jpg'.format(args.d))
        #####################################################################################
        #####################################################################################

    if args.d == 'cifar':
        # Set the learning phase to false, the model is pre-trained.
        # backend.set_learning_phase(False)        
        keras_model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
        classifier = KerasClassifier(model=keras_model, clip_values=(0, 255), use_logits=False)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()           
        x_org = x_test / 255
        # Evaluate the model's accuracy on the validation data used in training        
        x_test = x_test.astype("float32")
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        # Accuracy of our test data
        pred = np.argmax(classifier.predict(x_test), axis = 1)
        acc =  np.mean(np.equal(pred, y_test.reshape(-1)))

        print("The normal validation accuracy is: {}".format(acc))            

        attack = FastGradientMethod(classifier=classifier, eps=0.3, batch_size=64)
        x_test_adv = attack.generate(x=x_test)

        adv_pred = np.argmax(keras_model.predict(x_test_adv), axis = 1)
        adv_acc =  np.mean(np.equal(adv_pred, y_test))

        print("The adversarial validation accuracy is: {}".format(adv_acc))
        print(x_test_adv.shape)
        np.save('./adv/adv_{}.npy'.format(args.d), x_test_adv)
        
        # find the incorrrect instance
        incorrect_index, cnt = None, 0
        for i in range(adv_pred.shape[0]):
            if pred[i] != adv_pred[i]:
                incorrect_index = i
                if cnt == 10:
                    break

        x_sample = x_org[incorrect_index].reshape(32, 32, 3)
        adv_x_sample = x_test_adv[incorrect_index].reshape(32, 32, 3)

        # cifar10_plot(image=x_sample, title='TestPredicted: %s' % (str(pred[incorrect_index])), path_save='./adv/adv_%s_test.jpg' % (args.d))
        # cifar10_plot(image=adv_x_sample, title='AdvPredicted: %s' % (str(adv_pred[incorrect_index])), path_save='./adv/adv_%s_adv.jpg' % (args.d))        

        adv_comparison = stitch_images([x_sample, adv_x_sample], 1, 2)
        plt.suptitle('TestPredicted: %s -- AdvPredicted: %s' % (str(pred[incorrect_index]), str(adv_pred[incorrect_index])))
        plt.imshow(adv_comparison)
        plt.savefig('./adv/adv_{}.jpg'.format(args.d))
