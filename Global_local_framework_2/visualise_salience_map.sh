#!/bin/bash

cd "Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2"

export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"

python src/scripts/visualise_model.py \
--model_path="checkpoints/model_1-16/2-channel_val_best_model.pth" \
--data_csv_path="data/cropped_mammo-16/test.csv" \
--data_path="data/cropped_mammo-16/test" \
--mask_dir="data/mammo-test-samples/segmentation" \
--output_path="retrained_visualization" \
--bs=6 \
--percent_t=0.02 \