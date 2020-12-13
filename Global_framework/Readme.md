# Introduction

This folder contains the code for training and fine-tuning whole mammo and patch classifiers using the existing DNN models implemented by Keras

The following files are included in this folder

**train_mammo_classifier.py**
* Train the whole mammogram classifiers on SV's mammo dataset collected by Peter 
using InceptionResetV2, EfficientNetB6 and NASNetLarge models implemented by Keras
* The final model and training curves are saved
* It also evaluates the classification accuracy and roc_auc when training finishes

**train_patch_classifier.py**
* Train the patch classifiers on SV's sample dataset collected by Peter using Resnet50, EfficientNetB6 and NASNetLarge models implemented by Keras
* The final model and training curves are saved at the end of training

**fine_tuning_mammo_classifier.py**
* Perform fine-tuning on the whole mammogram classifiers on SV's mammo dataset
using InceptionResetV2, EfficientNetB6 and NASNetLarge models implemented by Keras
* Hyper-parameters like training epoch, batch size, image size, optimiser, learning rate and freeze layers are tuned on validation set to seek the optimal classification performance
* The final model and training curves of each tried parmater combinations are saved at the end of training
* Evaluates the classification accuracy and roc_auc on training and validation sets

**eval_metric.py**
* Evaluate perfromance metrics (incl. accuracy, auc, FP, FN, TP, TN) of a saved model on images from a specified folder
* Draw the ROC curve

**generate_finetuning_mammo_classifier_script.py**
* Automatically generate scripts for fine-tuning of mammogram classifiers

**mammo_inception_resnet_v2-kfold_training.sh**
* Script for training and testing the performance of InceptionResetV2-based whole mammo classifier under in a k-fold cross-validation setting

**patch_resnet50-kfold_training.sh**
* Script for training and testing the performance of ResNet50-based patch classifier under a k-fold cross-validation setting

**finetuning_inception_resnet_v2.sh**
* Example script for fine-tuning the whole mammo classifier using InceptionResetV2 model

**finetuning_EfficientNet.sh**
* Example script for fine-tuning the whole mammo classifier using EfficientNetB6 model

**finetuning_NASNet.sh**
* Example script for fine-tuning the whole mammo classifier using NASNetLarge model