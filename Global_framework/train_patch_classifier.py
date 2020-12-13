# This code trains mammogram classifier using resnet50, efficientnetB6 and nasnetlarge models implemented in Keras
# It loads pretrained models on Imagenet, switch the last layer to classify malignant and non-malignant patches
# retrains the Keras models on SV dataset
# draws training curve

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger

import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--datapath", type=str, default='../data/sample/',
    help="data path")
ap.add_argument("--fold", type=int, default=0,
    help="data fold")
ap.add_argument("--epochs", type=int, default=50,
    help="epoch numbers")
ap.add_argument("--lr", type=float, default=1e-5,
    help="base learning rate")
ap.add_argument("--bs", type=int, default=6,
    help="batch size")
ap.add_argument("--is", type=int, default=224,
    help="image size")
ap.add_argument("--fl", type=int, default=2,
    help="freeze layers")
ap.add_argument("--pretrained", type=bool, default=False,
    help="use pretrained model")
ap.add_argument("--model", type=str, default='resnet50',
    help="CNN model")
args = vars(ap.parse_args())

NUM_EPOCHS = args["epochs"]
LEARNING_RATE = args["lr"]
BATCH_SIZE = args["bs"]
MODEL = args["model"]
if MODEL == 'NASNet':
    IMAGE_SIZE = (331, 331)
else:
    IMAGE_SIZE = (args["is"], args["is"])
FREEZE_LAYERS = args["fl"]

PRETRAIN = args["pretrained"]
DATASET_PATH  = args["datapath"]
FOLD = args["fold"]

NUM_CLASSES   = 2
if PRETRAIN:
    model_weights = 'imagenet'
else:
    model_weights = None
WEIGHTS_FINAL = 'patch-{}-{}_bs{}_lr{}_ep{}_is{}_fl{}_fold{}.h5'.format(MODEL, model_weights, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, \
    IMAGE_SIZE[0], FREEZE_LAYERS, FOLD)
OUTPUT_DIR = 'model'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

print('training info: patch-classifier \nmodel:{}, model-weights:{}, batch-size:{}, learning rate:{}, epochs:{}, image-size:{}, freeze-layer:{}, fold:{}\n'.format(MODEL, model_weights, \
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, args["is"], FREEZE_LAYERS, FOLD))

if MODEL == 'resnet50':
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
elif MODEL == 'NASNet':
    preprocess_input = tf.keras.applications.nasnet.preprocess_input
elif MODEL == 'EfficientNet':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# show class indices
print('****************')
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')

if MODEL == 'EfficientNet':
    net = tf.keras.applications.EfficientNetB6(include_top=False,
                        weights=model_weights,
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
elif MODEL == 'NASNet':
    net = tf.keras.applications.NASNetLarge(include_top=False,
                        weights=model_weights,
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
elif MODEL == 'resnet50':
    net = ResNet50(include_top=False,
                weights=model_weights,
                input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))

x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

net_final.compile(optimizer=Adam(lr=LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=[
                    'accuracy',
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.TruePositives(),
                    tf.keras.metrics.FalseNegatives(),
                    tf.keras.metrics.TrueNegatives(),
                    tf.keras.metrics.FalsePositives()
                    ])
print(net_final.summary())

# train the model
logger_name = 'patch-{}-{}_bs{}_lr{}_ep{}_is{}_fl{}_fold{}.csv'.format(MODEL, model_weights, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, \
    IMAGE_SIZE[0], FREEZE_LAYERS, FOLD)
csv_logger = CSVLogger(os.path.join(OUTPUT_DIR, logger_name))
H = net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS, callbacks=[csv_logger])

# save trained weights
net_final.save(os.path.join(OUTPUT_DIR, WEIGHTS_FINAL))

# train report
loss_hist = H.history['val_loss']
acc_hist = H.history['val_accuracy']
if len(loss_hist) > 0:
    print(loss_hist)
    print (min(loss_hist))
    min_loss_locs = np.argmin(np.array(loss_hist))
    print(min_loss_locs)
    best_val_loss = loss_hist[min_loss_locs]
    best_val_accuracy = acc_hist[min_loss_locs]
    print('===========================================================================')
    print('                          train report                                     ')
    print('===========================================================================')
    print("Minimum val loss achieved at epoch:", min_loss_locs + 1)
    print("Best val loss:", best_val_loss)
    print("Best val accuracy:", best_val_accuracy)

# plot training curve
N = NUM_EPOCHS
# summarize history for loss
fig = plt.figure()
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('Training loss curve of patch classifier')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['training', 'valid'], loc='upper left')
plt.grid(True)
plt.tight_layout()
plotname = 'patch-train_loss_curve-{}-{}_bs{}_lr{}_ep{}_is{}_fl{}_fold{}.jpg'.format(MODEL, model_weights, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, \
    IMAGE_SIZE[0], FREEZE_LAYERS, FOLD)
plt.savefig(os.path.join(OUTPUT_DIR, plotname))
plt.close(fig)

fig = plt.figure()
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title("Training accuracy curve of patch classifier")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['training', 'valid'], loc='upper left')
plt.grid(True)
plt.tight_layout()
plotname = 'patch-train_acc_curve-{}-{}_bs{}_lr{}_ep{}_is{}_fl{}_fold{}.jpg'.format(MODEL, model_weights, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, \
    IMAGE_SIZE[0], FREEZE_LAYERS, FOLD)
plt.savefig(os.path.join(OUTPUT_DIR, plotname))
plt.close(fig)

plt.style.use("ggplot")
fig = plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy of patch classifier")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.grid(True)
plt.legend(loc="lower left")
plotname = 'patch-train_curve-{}-{}_bs{}_lr{}_ep{}_is{}_fl{}_fold{}.jpg'.format(MODEL, model_weights, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, \
    IMAGE_SIZE[0], FREEZE_LAYERS, FOLD)
plt.savefig(os.path.join(OUTPUT_DIR, plotname))
plt.close(fig)