import os, argparse
import numpy as np
import keras
from keras.models import load_model
from dm_image_1 import DMImageDataGenerator
from dm_keras_ext_1 import DMAucModelCheckpoint, DMMetrics
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description="DM image evaluation")
parser.add_argument("image_path", type=str)
parser.add_argument("resume_from", type=str)
parser.add_argument("rescale_factor", type=float, default=1.0)
parser.add_argument("featurewise_mean", type=float, default=37.40)
parser.add_argument("img_size", nargs=2, type=int, default=[1152, 896])
args = parser.parse_args()

model_path = args["resume_from"]
image_path = args["image_path"]
rescale_factor = args["rescale_factor"]
feature_mean = args["featurewise_mean"]
img_size = args["img_size"]

# load models
model = load_model(model_path, compile=False)

test_imgen = DMImageDataGenerator(featurewise_center=True)
test_imgen.mean = feature_mean
imageset_mean = 31.61 #sv_train_mean
imageset_std = 45.50 #sv_train_std
expected_std = 67.56 #dsm_test_std

test_generator = test_imgen.flow_from_directory(
    image_path, target_size=img_size, target_scale=None,
    rescale_factor=rescale_factor,
    equalize_hist=False, dup_3_channels=True, 
    classes=['neg', 'pos'], class_mode='categorical', batch_size=4, 
    shuffle=False, imageset_mean=imageset_mean, imageset_std=imageset_std, expected_mean=test_imgen.mean, expected_std=expected_std)

resnet_auc, resnet_y_true, resnet_y_pred, sensitivity, specificity, accuracy, TP, FN, TN, FP = DMAucModelCheckpoint.calc_test_auc(
    test_generator, model, test_samples=test_generator.nb_sample, return_y_res=True)
print 'resent model auc:', resnet_auc

print 'resent model senitivity, sepcificity, accuracy:', sensitivity, specificity, accuracy
print 'resent model TP, FN, TN, FP:', TP, FN, TN, FP
