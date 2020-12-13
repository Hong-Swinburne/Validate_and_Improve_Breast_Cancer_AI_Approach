# Introduction

This folder contains the code of global+local framework_1 which implements the method proposed in [Deep Learning to Improve Breast Cancer Detection on Screening Mammography](https://www.nature.com/articles/s41598-019-48995-4). All the code is cloned from the public Github repo [https://github.com/lishen/end2end-all-conv](https://github.com/lishen/end2end-all-conv) with a slight modification.

The following files are included in this folder

**train/image_clf_train.py**
* Train the image-level classifier with early stopping strategy
* Save the image-level classifier with the best performance on validation set
* Evaluate the performance of saved image-level classifier on test data

**train/model_eval.py.py**
* Calculate classification accuracy and AUC of any saved model(specified by the input parameter) on a particular data set

**train/patch_clf_train.py**
* Train the patch classifier with early stopping strategy
* Save the patch classifier with the best performance on validation set
* Evaluate the performance of saved patch classifier on test data

**dm_image.py**
* Perform image normalisation and augmentation
* Implement some basic image processing operations

**dm_keras_ext.py**
* Implement the network architecture of the whole image classifier  
* Implement the network architecture of the patch classifier
* Implement evaluation metrics such as accuracy, sensitivity, specificity etc.

**dm_multi_gpu.py**
* Implementation of training the DNN model on multi-GPU platform

**dm_resnet.py**
* Implement Resnet block and vgg block for image-level classifier and patch classifier

**model_eval.sh**
* Example script to evaluate the performance of image classifier

**retrain_mammo_classifier_featurewise-normalise.sh**
* Script to re-train the pre-trained image-level classifiers on SV data