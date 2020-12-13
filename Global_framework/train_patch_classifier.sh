#!/bin/bash

cd "Validate_and_Improve_Breast_Cancer_AI_Approach/Global_framework"

python train_patch_classifier.py
    --datapath="../data/sample"\
    --epochs=100\
    --lr=1e-5\
    --bs=6\
    --is=224\
    --pretrained=True\
    --model="resnet50"

