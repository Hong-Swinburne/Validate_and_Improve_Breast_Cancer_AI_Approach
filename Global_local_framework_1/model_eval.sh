#!/bin/bash

cd "Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_1"

export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_1/"

IMAGE_PATH="data/mdpp/Combined_full_images/test"
MODEL_PATH="pretrained_models/ddsm_vgg16_s10_512x1.h5"

python train/predict_ddsm_full1152.py \
    --image_path $IMAGE_PATH \
    --resume_from $MODEL_PATH \
    --rescale_factor 1.0 \
    --featurewise_mean 37.40 \
    --img_size 1152 896
