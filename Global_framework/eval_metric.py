# evaluate models on testing data and calculate metrics,draw ROC
import os
import argparse
import sys
import glob
import numpy as np
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing import image


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    files = [f for f in files if f.endswith('PNG') or f.endswith('png') or f.endswith('JPG') or f.endswith('jpg')]

    if not len(files):
        sys.exit('No images found by the given path!')

    return files

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--type", type=str, default="patch",
    help="classifier type (patch or mammo)")
ap.add_argument("--model_dir", type=str, default="model",
    help="path of models")
ap.add_argument("--datapath", type=str, default="../sample/valid",
    help="path of dataset")
ap.add_argument("--cropped", type=bool, default=False,
    help="using cropped mammo")
ap.add_argument("--fold", type=int, default=0,
    help="data fold")
ap.add_argument("--epochs", type=int, default=50,
    help="epoch numbers")
ap.add_argument("--lr", type=float, default=1e-5,
    help="base learning rate")
ap.add_argument("--bs", type=int, default=6,
    help="batch size")
ap.add_argument("--is", type=int, default=224,
    help="image size")
ap.add_argument("--fl", type=int, default=2,
    help="freeze layers")
ap.add_argument("--pretrained", type=bool, default=True,
    help="use pretrained model")
ap.add_argument("--model", type=str, default='resnet-50',
    help="CNN model")
ap.add_argument("--thresh", type=float, default=0.95,
    help="threshold for ROC")
args = vars(ap.parse_args())

CLASSIFIER_TYPE = args['type']
if args["cropped"] == True:
    CROP = 'crop'
else:
    CROP = 'nocrop'
NUM_EPOCHS = args["epochs"]
LEARNING_RATE = args["lr"]
BATCH_SIZE = args["bs"]
IMAGE_SIZE = (args["is"], args["is"])
FREEZE_LAYERS = args["fl"]
MODEL = args["model"]
PRETRAIN = args["pretrained"]
DATASET_PATH  = args["datapath"]
MODEL_PATH = args["model_dir"]
FOLD = args["fold"]
Threshold = args['thresh']

if PRETRAIN:
    model_weights = 'imagenet'
else:
    model_weights = None

WEIGHTS_FINAL = 'mammo-inception_resnet_v2-None_bs2_lr1e-05_ep50.h5'
WEIGHTS_FINAL = '{}-mammo-{}-{}_bs{}_lr{}_ep{}_is{}_fl{}_fold{}_bestauc.h5'.format(CROP, MODEL, model_weights, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, \
    IMAGE_SIZE[0], FREEZE_LAYERS, FOLD)

if MODEL == 'resnet50':
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
elif MODEL == 'inception_resnet_v2':
    preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
elif MODEL == 'NASNet':
    preprocess_input = tf.keras.applications.nasnet.preprocess_input
elif MODEL == 'EfficientNet':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

cls_list = ['cancer/', 'normal/']
pos=[]
neg=[]
TP=0
FP=0
TN=0
FN=0

print('loading model.....')
net = load_model(os.path.join(MODEL_PATH, WEIGHTS_FINAL))
print('model loaded.....')

for category in cls_list:
    data_path = os.path.join(DATASET_PATH, category)
    print(data_path)

    files = get_files(data_path)
    print('find {} images'.format(len(files)))

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
        print(f)
        for i in top_inds:
            print(pred[0],pred[1])

        if category==cls_list[0]:
            print('cancer: prediction = ', pred[0])
            if pred[0]>=Threshold:
                TP+=1
            else:
                FN+=1
            pos.append(pred[0])
        else:
            print('normal: prediction = ', pred[0])
            if pred[0]<Threshold:
                TN+=1
            else:
                FP+=1
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
print('Sensitivity Specificity Accuracy = ', 100*TP/(TP+FN), 100*TN/(TN+FP), 100*(TP+TN)/(TP+TN+FP+FN))

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

plotname = '{}-roc-thresh{}-{}-{}_bs{}_lr{}_ep{}.jpg'.format(CLASSIFIER_TYPE, Threshold, MODEL, model_weights, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS)
plt.savefig(os.path.join(MODEL_PATH, plotname))

plt.close()