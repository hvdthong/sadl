from keras import utils
from keras.models import load_model, Model
from keras.datasets import mnist, cifar10
import argparse
import matplotlib.pyplot as plt
import numpy as np
from spacial_transformation import *
import random
import math
from utils import write_file
import math

class Perturbator:
    def __init__(self):
        self.trans_functions = dict()
        self.trans_functions["rotate"] = rotate_image
        self.trans_functions["translate"] = image_translation_cropped
        self.trans_functions["shear"] = image_shear_cropped

        # the function of filters
        self.trans_functions["zoom"] = image_zoom
        self.trans_functions["blur"] = image_blur
        self.trans_functions["contrast"] = image_contrast
        self.trans_functions["brightness"] = image_brightness

    def set_rotation_range(self, _range=30):
        self.rotation_range = range(_range*-1, _range)

    def random_rotate_perturb(self, x=None, y=None):
        for i in range(len(x)):
            angle = random.choice(self.rotation_range)
            x[i] = self.trans_functions["rotate"](x[i], angle)
        return x, y

    def random_perturb(self, x=None, y=None):
        for i in range(len(x)):
            x[i] = self.random_perturb_image(x[i])
        return x, y

    def random_perturb_image(self, img=None):
        """randomly perturb one image"""
        # spacial transformation
        angle = random.choice(self.rotation_range)
        translation = random.choice(self.translate_range)
        translation_v = random.choice(self.translate_range)
        shear = random.choice(self.shear_range)
        zoom = 1
        blur = 0
        brightness = 0
        contrast = 1
        # transformation based on filter
        if self.enable_filters:
            zoom = random.choice(self.zoom_range)
            blur = random.choice(self.blur_range)
            brightness = random.choice(self.brightness_range)
            contrast = random.choice(self.contrast_range)
        img = self.fix_perturb_img(img, angle, translation, translation_v, shear, zoom, blur, brightness, contrast)
        return img

    def fix_perturb(self, x=None, y=None, angle=15, translation=2, translation_v=0, shear=0.1,
                    zoom=1, blur=0,  brightness=0, contrast=1):
        length = range(len(x))
        for i in length:
            x[i] = self.fix_perturb_img(x[i], angle, translation, translation_v,
                                        shear, zoom, blur, brightness, contrast)
        return x, y

    def fix_perturb_img(self, img, angle=15, translation=0, translation_v=0, shear=0.1,
                        zoom=1, blur=0, brightness=0, contrast=1):
        img = self.trans_functions["rotate"](img, angle)
        img = self.trans_functions["translate"](img, translation, translation_v)
        img = self.trans_functions["shear"](img, shear)
        if self.enable_filters:
            img = self.trans_functions["zoom"](img, zoom)
            img = self.trans_functions["blur"](img, blur)
            img = self.trans_functions["brightness"](img, brightness)
            img = self.trans_functions["contrast"](img, contrast)        
        return img

CLIP_MIN = -0.5
CLIP_MAX = 0.5

def add_noise_to_image(image, actions, values):
    n_images = list()    

    for act in actions:
        pt = Perturbator()
        if act == 'rotate':
            for v in values[0]:
                # print(image.shape)
                img = pt.trans_functions['rotate'](image, v)
                img = img[:32, :32, :]
                n_images.append(img)
        
        if act == 'translate_h':            
            for v in values[1]:
                img = pt.trans_functions['translate'](image, v, 0)
                # print(img.shape)
                # exit()
                n_images.append(img)

        if act == 'translate_v':            
            for v in values[2]:
                img = pt.trans_functions['translate'](image, 0, v)                
                n_images.append(img)
    return n_images

def correctness(pred, label):
    row, _ = pred.shape
    correct = 0
    for i in range(row):        
        if np.argmax(pred[i]) == np.argmax(label):
            correct += 1
    return float(correct / row)

def diversity(pred, label):
    row, _ = pred.shape
    # dv = list()
    dv = dict()
    for i in range(row):
        # if np.argmax(pred[i]) != np.argmax(label):
        #     dv.append(np.argmax(pred[i]))
        pred_label = np.argmax(pred[i])
        if pred_label not in dv.keys():
            dv[pred_label] = 1
        else:
            dv[pred_label] = dv[pred_label] + 1
    total = 0
    
    for k in dv.keys():                
        total += math.pow(dv[k] / row, 2)    
    # return len(list(set(dv)))
    return total

def majority(pred, label, class_name):
    row, _ = pred.shape
    dv = dict()
    for i in range(row):        
        pred_label = np.argmax(pred[i])
        if pred_label not in dv.keys():
            dv[pred_label] = 1
        else:
            dv[pred_label] = dv[pred_label] + 1    
    max_key = max(dv, key=dv.get)
    max_value = dv[max_key]

    name = ''
    for k in dv.keys():
        if dv[k] == max_value:
            name = name + class_name[k] + '_'
    name = name[:-1]
    return name


def result_of_noise(noise_image, model, label, class_name):
    noise_image = np.array(noise_image)
    noise_image = (noise_image / 255.0) - (1.0 - CLIP_MAX)
    y_pred = model.predict(noise_image)
    return diversity(y_pred, label), correctness(y_pred, label), majority(y_pred, label, class_name)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"    
    actions = ['rotate', 'translate_h', 'translate_v']
    values = [[-10, -8, -6 ,-4, -2, 2, 4, 6, 8, 10], [-3, -2, -1, 1 , 2, 3], [-3, -2, -1, 1, 2, 3]]    
    if args.d == 'cifar':
        dict_label = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # x_train = x_train.astype("float32")
        # x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        # x_test = x_test.astype("float32")
        # x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        # # number of class
        # num_class = 10        
        # y_test = utils.to_categorical(y_test, num_class)

        # model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
        # y_pred = model.predict(x_test)

        # pred_label = list(np.argmax(y_pred, axis=1))
        # true_label = list(np.argmax(y_test, axis=1))

        # pred_label = [dict_label[p] for p in pred_label]
        # true_label = [dict_label[p] for p in true_label]

        # cnt = 0
        # for p, t in zip(pred_label, true_label):
        #     if p == t:
        #         cnt += 1
        # print(cnt)
        # write_file('./sa/pred_label_cifar.txt', pred_label)
        # write_file('./sa/true_label_cifar.txt', true_label)

        # number of class
        num_class = 10        
        y_test = utils.to_categorical(y_test, num_class)
        model = load_model('./model_tracking/cifar_model_improvement-496-0.87.h5')
        divs, cors, majors = list(), list(), list()
        for i in range(x_test.shape[0]):            
            div, cor, major = result_of_noise(noise_image=add_noise_to_image(image=x_test[i], actions=actions, values=values), model=model, label=y_test[i], class_name=dict_label)
            print(i, div, cor, major)
            divs.append(div)
            cors.append(cor)
            majors.append(major)

        # write_file('./sa/diversity_cifar.txt', divs)
        # write_file('./sa/correctness_cifar.txt', cors)
        write_file('./sa/majors_cifar.txt', majors)
        print(len(divs), len(cors), len(majors))



