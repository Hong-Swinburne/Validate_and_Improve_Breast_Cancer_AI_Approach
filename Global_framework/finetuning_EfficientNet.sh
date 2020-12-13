#!/bin/bash

cd "Validate_and_Improve_Breast_Cancer_AI_Approach/Global_framework"

python "fine_tuning_mammo_classifier.py"
    --datapath="../data/mammo"\
    --fold=0\
    --epochs=50\
    --lr=0.001\
    --bs=2\
    --is=768\
    --fl=4\
    --val_size 0.1\
    --pretrained True\
    --model="EfficientNet"\
    --optimiser="rmsprop"

