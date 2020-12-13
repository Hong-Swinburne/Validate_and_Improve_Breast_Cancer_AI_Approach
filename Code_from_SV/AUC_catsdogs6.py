"""
This script goes along my blog post:
'Keras Cats Dogs Tutorial' (https://jkjung-avt.github.io/keras-tutorial/)
"""


import os
import sys
import glob
import argparse

import numpy as np

from keras import backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import load_model
from tensorflow.python.keras.preprocessing import image


from sklearn import metrics
from scipy import misc
import scipy
from PIL import Image
# import nibabel
import matplotlib.pyplot as plt

IMAGE_SIZE    = (1024, 1024)
# WEIGHTS_FINAL = 'model-resnet50-ltd_2_April30_epoch20.h5'
# WEIGHTS_FINAL = 'model-inception_resnet_v2-input-size_1024.h5'
WEIGHTS_FINAL = 'model-inception_resnet_v2-input-size_1024_Jan2020.h5'
Threshold=0.95

def parse_args():
    """Parse input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    files = [f for f in files if f.endswith('JPG') or f.endswith('jpg')]

    if not len(files):
        sys.exit('No images found by the given path!')

    return files


if __name__ == '__main__':
    args = parse_args()
    cls_list = ['cats/', 'dogs/']
    # cls_list = ['whole_4_jpgs/', 'normal_4_jpgs/']

    pos=[]
    neg=[]
    TP=0
    FP=0
    TN=0
    FN=0    

    Base_Dir='inception_tensorb_thresh0.95_1024/'

    try:
        os.mkdir(Base_Dir)
    except:
        print('directory present')

    try:
        os.mkdir(Base_Dir+'TP')
    except:
        print('directory present')

    try:
        os.mkdir(Base_Dir+'FN')
    except:
        print('directory present')
    try:
        os.mkdir(Base_Dir+'TN')
    except:
        print('directory present')
    try:
        os.mkdir(Base_Dir+'FP')
    except:
        print('directory present')


    net = load_model(WEIGHTS_FINAL)

    for category in cls_list:
        print(args.path+category)
        
        files = get_files(args.path+category)

        # load the trained model

        # loop through all files and make predictions
        for f in files:
            img = image.load_img(f, target_size=IMAGE_SIZE)
            if img is None:
                continue
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)
            pred = net.predict(x)[0]
            top_inds = pred.argsort()[::-1][:5]
            # print(f)
            # for i in top_inds:
            # print(pred[0],pred[1])
            split_name=os.path.split(f)
            file_name=split_name[1][:-4]

            if category==cls_list[0]:
                print('cats: prediction = ', pred[0])
                if pred[0]>=Threshold:
                    TP+=1
                    # img.save(Base_Dir+'TP'+'/'+file_name+'.jpg')
                else:
                    FN+=1
                    # img.save(Base_Dir+'FN'+'/'+file_name+'.jpg')
                pos.append(pred[0])
            else:
                print('dogs: prediction = ', pred[0])
                if pred[0]<Threshold:
                    TN+=1
                    # img.save(Base_Dir+'TN'+'/'+file_name+'.jpg')
                else:
                    FP+=1
                    # img.save(Base_Dir+'FP'+'/'+file_name+'.jpg')
                neg.append(pred[0])

score_pos=pos
score_neg=neg
y_neg = np.zeros(len(score_neg))
y_pos = np.ones(len(score_pos))
y = np.concatenate((y_pos,y_neg),axis=0)
y = y.astype(int)
y = y+1
n_classes=2
scores=np.concatenate((score_pos,score_neg),axis=0)

fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
roc_auc = metrics.auc(fpr, tpr)

print('TP FN TN FP = ', TP, FN, TN, FP)
# print('Sensitivity Specificity Accuracy = ', 100*TP/(TP+FN), 100*TN/(TN+FP), 100*(TP+TN)/(TP+TN+FP+FN))

print('roc_auc = ', roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (AUC =  %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Classification')
plt.legend(loc="lower right")


# create the axis of thresholds (scores)
ax2 = plt.gca().twinx()
ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
ax2.set_ylabel('Threshold',color='r')
ax2.set_ylim([thresholds[-1],thresholds[0]])
ax2.set_xlim([fpr[0],fpr[-1]])


plt.savefig('roc_and_threshold_batch_Data_Jan_2020.png')
plt.show()

plt.close()


#