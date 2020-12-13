#!/bin/bash

cd "Validate_and_Improve_Breast_Cancer_AI_Approach/Global_framework"

python fine_tuning_mammo_classifier.py\
    --datapath="../data/mammo"\
    --fold=0\
    --epochs=50\
    --lr=0.001\
    --bs=4\
    --is=512\
    --fl=2\
    --val_size=0.1\
    --pretrained= True\
    --model="inception_resnet_v2"\
    --loss='categorical_crossentropy'\
    --class_mode='categorical'\
    --optimiser="adam"

