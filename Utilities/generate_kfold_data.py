# This file generates data for k-fold cross-validation

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--type", type=str, default="patch",
    help="patch or mammo")
ap.add_argument("--k", type=int, default=5,
    help="fold number")
args = vars(ap.parse_args())

data_type = args['type']
num_fold = args["k"]

if data_type == 'patch':
    datapath = ['../data/sample/cancer', '../data/sample/normal']
    dstpath = '../data/sample'
elif data_type == 'mammo':
    datapath = ['../data/mammo/cancer', '../data/mammo/normal']
    dstpath = '../data/mammo'

imagenames = []
labels = []

for path in datapath:
    category = os.path.split(path)[-1]
    images = os.listdir(path)
    for image in images:
        imagenames.append(image)
        labels.append(category)

imagenames = np.array(imagenames)
labels = np.array(labels)

skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=10)
k = 1
for train_index, test_index in skf.split(imagenames, labels):
    print("fold {}".format(k), "TRAIN:", train_index.shape[0], "TEST:", test_index.shape[0])
    X_train, X_test = imagenames[train_index], imagenames[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    split_path = os.path.join(dstpath, 'fold_{}'.format(k))
    train_path = os.path.join(split_path, 'train')
    valid_path = os.path.join(split_path, 'valid')
    train_cancer_path = os.path.join(train_path, 'cancer')
    train_normal_path = os.path.join(train_path, 'normal')
    valid_cancer_path = os.path.join(valid_path, 'cancer')
    valid_normal_path = os.path.join(valid_path, 'normal')
    if not os.path.exists(split_path):
        os.mkdir(split_path)
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(valid_path):
        os.mkdir(valid_path)
    if not os.path.exists(train_cancer_path):
        os.mkdir(train_cancer_path)
    if not os.path.exists(train_normal_path):
        os.mkdir(train_normal_path)
    if not os.path.exists(valid_cancer_path):
        os.mkdir(valid_cancer_path)
    if not os.path.exists(valid_normal_path):
        os.mkdir(valid_normal_path)
    num_train_cancer = 0
    num_train_normal = 0
    num_test_cancer = 0
    num_test_normal = 0

    for image, label in zip(X_train, y_train):
        if label == 'cancer':
            shutil.copyfile(os.path.join(datapath[0], image), os.path.join(train_cancer_path, image))
            num_train_cancer+=1
        elif label == 'normal':
            shutil.copyfile(os.path.join(datapath[1], image), os.path.join(train_normal_path, image))
            num_train_normal+=1
    for image, label in zip(X_test, y_test):
        if label == 'cancer':
            shutil.copyfile(os.path.join(datapath[0], image), os.path.join(valid_cancer_path, image))
            num_test_cancer+=1
        elif label == 'normal':
            shutil.copyfile(os.path.join(datapath[1], image), os.path.join(valid_normal_path, image))
            num_test_normal+=1

    print('finished split {}/{} fold data generation, train cancer normal#:{} {}, test cancer normal#:{} {}'.format(k, num_fold, num_train_cancer, num_train_normal,\
         num_test_cancer, num_test_normal))
    k+=1
        
