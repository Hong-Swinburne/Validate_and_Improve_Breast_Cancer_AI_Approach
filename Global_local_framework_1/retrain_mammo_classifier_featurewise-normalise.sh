#!/bin/bash

cd "Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_1/"

export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_1/"

TRAIN_DIR="data/mdpp/Combined_full_images/train"
VAL_DIR="data/mdpp/Combined_full_images/val"
TEST_DIR="data/mdpp/Combined_full_images/test"
RESUME_FROM="Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_1/pretrained_models/ddsm_vgg16_s10_512x1.h5"
BEST_MODEL="saved_model/mdpp_vgg16_s10_512x1_featurewise-normalisation.h5"
FINAL_MODEL="NOSAVE"

python ddsm_train/image_clf_train.py \
    --no-patch-model-state \
    --resume-from $RESUME_FROM \
    --img-size 1152 896 \
    --no-img-scale \
    --rescale-factor 1.00 \
    --featurewise-center \
    --featurewise-mean 33.59 \
    --no-equalize-hist \
    --batch-size 4 \
    --train-bs-multiplier 0.5 \
    --augmentation \
    --class-list neg pos \
    --nb-epoch 0 \
    --all-layer-epochs 80 \
    --no-load-val-ram \
    --no-load-train-ram \
    --optimizer adam \
    --weight-decay 0.001 \
    --hidden-dropout 0.0 \
    --weight-decay2 0.01 \
    --hidden-dropout2 0.0 \
    --init-learningrate 0.0001 \
    --all-layer-multiplier 0.01 \
    --es-patience 10 \
    --auto-batch-balance \
    --best-model $BEST_MODEL \
    --final-model $FINAL_MODEL \
    $TRAIN_DIR $VAL_DIR $TEST_DIR