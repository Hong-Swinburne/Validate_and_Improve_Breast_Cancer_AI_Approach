#!/bin/bash

cd "Validate_and_Improve_Breast_Cancer_AI_Approach/Global_framework"

python eval_metric.py
 --type='mammo'\
 --model_dir='model'\
 --datapath='../data/mammo/valid'\
 --epochs=50\
 --lr=1e-5\
 --bs=2\
 --is=1024\
 --fl=2\
 --pretrained=True\
 --model="inception_resnet_v2"\
 --thresh=0.50