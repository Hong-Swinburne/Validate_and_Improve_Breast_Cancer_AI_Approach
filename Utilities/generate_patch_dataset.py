# This file generates the original dataset and file list for image patches from 'sample' folder sent by Peter
import os
import shutil

patch_dir = '../data/sample'
cancer_dir = '../data/sample/cancer'
normal_dir = '../data/sample/normal'

if not os.path.exists(patch_dir):
    os.mkdir(patch_dir)
if not os.path.exists(cancer_dir):
    os.mkdir(cancer_dir)
if not os.path.exists(normal_dir):
    os.mkdir(normal_dir)

classes = ['cancer', 'normal']
src_cancer_imagedir = ['../data/sample/train/cancer', '../data/sample/valid/cancer']
src_normal_imagedir = ['../data/sample/train/normal', '../data/sample/valid/normal']

k = 0
with open('patch_cancer.txt', 'w') as file:
    for dir in src_cancer_imagedir[:10]:
        imagelists = os.listdir(dir)
        for image in imagelists:
            shutil.copyfile(os.path.join(dir, image), os.path.join(cancer_dir, image))
            file.write(image + '\n')
            k+=1
print('copy {} images to cancer folder'.format(k))

k = 0
with open('patch_normal.txt', 'w') as file:
    for dir in src_normal_imagedir[:10]:
        imagelists = os.listdir(dir)
        for image in imagelists:
            shutil.copyfile(os.path.join(dir, image), os.path.join(normal_dir, image))
            file.write(image + '\n')
            k+=1
print('copy {} images to normal folder'.format(k))