# This code generates batch scripts for fine tuning of mammo classifier in "Global_framework/fine_tune_scripts" folder

import os
import subprocess
import time


template_script = r"""#!/bin/bash

python fine_tuning_mammo_classifier.py
    --datapath="../data/mammo"\
    --fold={fold}\
    --epochs={epoch}\
    --lr={lr}\
    --bs={bs}\
    --is={image_size}\
    --fl={freeze_layer}\
    --val_size 0.2\
    --pretrained True\
    --model={model}\
    --loss 'categorical_crossentropy'\
    --class_mode 'categorical'\
    --optimiser={opt}
"""

if not os.path.exists("Global_framework/fine_tune_scripts"):
    os.makedirs("Global_framework/fine_tune_scripts")

k = 0
epochs = [30, 40, 50]
learning_rate = [1e-3, 1e-4, 1e-5]
batch_size = [1, 2, 3, 4]
image_size = [512, 896, 1024]
freeze_layer = [2, 4]
classifiers = ['inception_resnet_v2', 'EfficientNet', 'NASNet']
optimiser = ['adam', 'rmsprop']

# Write and create srcipt files
for epoch in epochs:
    for lr in learning_rate:
        for bs in batch_size:
            for img_size in image_size:
                for fl in freeze_layer:
                    for clf in classifiers:
                        for opt in optimiser:
                            model = clf
                            task_name = "mammo_{}".format(clf)
                            task_name += "-ep"+str(epoch)+"-lr"+str(lr)+"-bs"+str(bs)+"-is"+str(img_size)+"-fl"+str(fl)+"-opt_"+str(opt)
                            if k == 0:
                                task_name += "-fold"+str(0)
                                fold = -1
                                script_path = os.path.join("Global_framework/fine_tune_scripts", task_name+".sh")

                                with open(script_path, "w+") as slurm_file:
                                    print(
                                        template_script.format(
                                            task_name=task_name,
                                            fold=fold+1,
                                            epoch=epoch,
                                            lr=lr,
                                            bs=bs,
                                            image_size=img_size,
                                            freeze_layer=fl,
                                            model=model,
                                            opt=opt,
                                        ),
                                        file=slurm_file,
                                        flush=True,
                                    )

                            else:
                                for fold in range(k):
                                    task_name += "-fold"+str(fold+1)
                                    script_path = os.path.join("Global_framework/fine_tune_scripts", task_name+".sh")

                                    with open(script_path, "w+") as slurm_file:
                                        print(
                                            template_script.format(
                                                task_name=task_name,
                                                fold=fold+1,
                                                epoch=epoch,
                                                lr=lr,
                                                bs=bs,
                                                image_size=img_size,
                                                freeze_layer=fl,
                                                model=model,
                                                opt=opt,
                                            ),
                                            file=slurm_file,
                                            flush=True,
                                        )