# This file generates the original dataset for training mammogram classifier using training and test file lists sent from Peter
import os
import shutil

mammo_dir = '../data/mammo'
mammo_train_dir = '../data/mammo/train'
mammo_valid_dir = '../data/mammo/valid'
classes = ['cancer', 'normal']
src_imagedir = '../data/File2-Cancer-image/'
imagelists = os.listdir(src_imagedir)

if not os.path.exists(mammo_dir):
    os.mkdir(mammo_dir)
if not os.path.exists(mammo_train_dir):
    os.mkdir(mammo_train_dir)
    for cls in classes:
        if not os.path.exists(mammo_train_dir + '/' + cls):
            os.mkdir(mammo_train_dir + '/' + cls)
if not os.path.exists(mammo_valid_dir):
    os.mkdir(mammo_valid_dir)
    for cls in classes:
        if not os.path.exists(mammo_valid_dir + '/' + cls):
            os.mkdir(mammo_valid_dir + '/' + cls)


train_cancer = []
train_normal = []
val_cancer = []
val_normal = []
with open('train_normal.txt', 'r') as file:
    train_normal = file.readlines()
k = 0
for i in train_normal:
    i = i.strip('\n')
    exam = i.split('_')[0]
    image_id = i[i.find('_')+1:i.find('.')]
    image_name = exam + '-' + image_id + '-nocancer-image.png'
    image_path = src_imagedir + image_name
    if not os.path.exists(image_path):
        print('{} not exist\n'.format(image_name))
    else:
        shutil.copyfile(src_imagedir + image_name, '../data/mammo/train/normal/'+ i.split('.')[0] + '.png')
        k+=1
print('copy {} images to train/normal'.format(k))

with open('train_cancer.txt', 'r') as file:
    train_cancer = file.readlines()
k = 0
for i in train_cancer:
    i = i.strip('\n')
    exam = i.split('_')[0]
    image_id = i[i.find('_')+1:i.find('.')]
    image_name = exam + '-' + image_id + '-cancer-image.png'
    image_path = src_imagedir + image_name
    if not os.path.exists(image_path):
        print('{} not exist\n'.format(image_name))
    else:
        shutil.copyfile(src_imagedir + image_name, '../data/mammo/train/cancer/'+ i.split('.')[0] + '.png')
        k+=1
print('copy {} images to train/cancer'.format(k))

with open('valid_normal.txt', 'r') as file:
    val_normal = file.readlines()
k = 0
for i in val_normal:
    i = i.strip('\n')
    exam = i.split('_')[0]
    image_id = i[i.find('_')+1:i.find('.')]
    image_name = exam + '-' + image_id + '-nocancer-image.png'
    image_path = src_imagedir + image_name
    if not os.path.exists(image_path):
        print('{} not exist\n'.format(image_name))
    else:
        shutil.copyfile(src_imagedir + image_name, '../data/mammo/valid/normal/'+ i.split('.')[0] + '.png')
        k+=1
print('copy {} images to valid/normal'.format(k))

with open('valid_cancer.txt', 'r') as file:
    val_cancer = file.readlines()
k = 0
for i in val_cancer:
    i = i.strip('\n')
    exam = i.split('_')[0]
    image_id = i[i.find('_')+1:i.find('.')]
    image_name = exam + '-' + image_id + '-cancer-image.png'
    image_path = src_imagedir + image_name
    if not os.path.exists(image_path):
        print('{} not exist\n'.format(image_name))
    else:
        shutil.copyfile(src_imagedir + image_name, '../data/mammo/valid/cancer/'+ i.split('.')[0] + '.png')
        k+=1
print('copy {} images to valid/cancer'.format(k))