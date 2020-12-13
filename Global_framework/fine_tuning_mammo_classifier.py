#split training data into training and validation sets for fine-tuning hyperparameters
from cv2 import data
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os, sys, glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn import metrics
from tensorflow.python.keras.preprocessing import image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--datapath", type=str, default='../data/mammo/',
    help="data path")
ap.add_argument("--fold", type=int, default=0,
    help="data fold")
ap.add_argument("--epochs", type=int, default=30,
    help="epoch numbers")
ap.add_argument("--lr", type=float, default=1e-5,
    help="base learning rate")
ap.add_argument("--bs", type=int, default=2,
    help="batch size")
ap.add_argument("--is", type=int, default=1024,
    help="image size")
ap.add_argument("--fl", type=int, default=2,
    help="freeze layers")
ap.add_argument("--val_size", type=float, default=0.2,
    help="validation size")
ap.add_argument("--pretrained", type=bool, default=True,
    help="use pretrained model")
ap.add_argument("--model", type=str, default='inception_resnet_v2',
    help="CNN model")
ap.add_argument("--loss", type=str, default='categorical_crossentropy',
    help="loss function")
ap.add_argument("--class_mode", type=str, default='categorical',
    help="class mode")
ap.add_argument("--optimiser", type=str, default='adam',
    help="optimiser")
args = vars(ap.parse_args())

NUM_EPOCHS = args["epochs"]
LEARNING_RATE = args["lr"]
BATCH_SIZE = args["bs"]
MODEL = args["model"]
IMAGE_SIZE = (args["is"], args["is"])
FREEZE_LAYERS = args["fl"]

PRETRAIN = args["pretrained"]
DATASET_PATH  = args["datapath"]
FOLD = args["fold"]
VALID_SIZE = args["val_size"]
LOSS = args["loss"]
CLS_MODE = args["class_mode"]
OPTIMISER = args["optimiser"]

NUM_CLASSES = 2
if PRETRAIN:
    model_weights = 'imagenet'
else:
    model_weights = None
WEIGHTS_FINAL = 'mammo-{}-{}-{}-{}_bs{}_lr{}_ep{}_is{}_fl{}_fold{}.h5'.format(MODEL, model_weights, LOSS, OPTIMISER, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, \
    IMAGE_SIZE[0], FREEZE_LAYERS, FOLD)
  
OUTPUT_DIR = 'model'
os.makedirs(OUTPUT_DIR, exist_ok=True)
  
print('training info: mammo-classifier \nmodel:{}, model-weights:{}, batch-size:{}, learning rate:{}, epochs:{}, image-size:{}, freeze-layer:{}, optimizer:{}, fold:{}\n'.format(MODEL, \
    model_weights, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, IMAGE_SIZE[0], FREEZE_LAYERS, OPTIMISER, FOLD))

if MODEL == 'inception_resnet_v2':
    preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
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
                                   fill_mode='nearest', validation_split=VALID_SIZE)
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode=CLS_MODE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset='training')

valid_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode=CLS_MODE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset='validation')
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_batches = test_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode=CLS_MODE,
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)


# show class indices
print('****************')
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')

if MODEL == 'inception_resnet_v2':
    net = InceptionResNetV2(include_top=False,
                            weights=model_weights,
                            input_tensor=None,
                            input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
elif MODEL == 'EfficientNet':
        net = tf.keras.applications.EfficientNetB6(include_top=False,
                            weights=model_weights,
                            input_tensor=None,
                            input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
elif MODEL == 'NASNet':
    net = tf.keras.applications.NASNetLarge(include_top=False,
                    weights=model_weights,
                    # weights=None,
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

if OPTIMISER == 'adam':
    optimizer = Adam(lr=LEARNING_RATE)
elif OPTIMISER == 'rmsprop':
    optimizer = RMSprop(lr=LEARNING_RATE)
elif OPTIMISER == 'adadelta':
    optimizer = Adadelta(lr=LEARNING_RATE)

net_final.compile(optimizer=optimizer,
                loss=LOSS, metrics=[
                    'accuracy',
                    tf.keras.metrics.AUC()
                    ])
print(net_final.summary())

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
# save the best model so far
model_name = 'mammo-{}-{}-{}-{}_bs{}_lr{}_ep{}_is{}_fl{}_fold{}'.format(MODEL, model_weights, LOSS, OPTIMISER, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, \
    IMAGE_SIZE[0], FREEZE_LAYERS, FOLD)
checkpoint_filepath = 'fine_tune'
if not os.path.exists(checkpoint_filepath):
    os.mkdir(checkpoint_filepath)
checkpoint_filepath = checkpoint_filepath + '/' + model_name
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_auc',
    mode='max',
    save_best_only=True)

# train the model
logger_name = 'mammo-{}-{}-{}-{}_bs{}_lr{}_ep{}_is{}_fl{}_fold{}_training.csv'.format(MODEL, model_weights, \
    LOSS, OPTIMISER, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, IMAGE_SIZE[0], FREEZE_LAYERS, FOLD)
csv_logger = CSVLogger(os.path.join(OUTPUT_DIR, logger_name))

# train the model
H = net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS, callbacks=[earlyStopping, model_checkpoint_callback, csv_logger], verbose=2)


# save trained weights
net_final.save(os.path.join(OUTPUT_DIR, WEIGHTS_FINAL))

# train report
auc_hist = H.history['val_auc']
acc_hist = H.history['val_accuracy']
if len(auc_hist) > 0:
    print (max(auc_hist))
    max_auc_locs = np.argmax(np.array(auc_hist))
    best_val_auc = auc_hist[max_auc_locs]
    best_val_accuracy = acc_hist[max_auc_locs]
    print('===========================================================================')
    print('                          train report                                     ')
    print('===========================================================================')
    print("Maximum val auc achieved at epoch:", max_auc_locs + 1)
    print("Best val auc:", best_val_auc)
    print("Best val accuracy:", best_val_accuracy)


#==============================================================================================================
# load the best model, test on validation set
def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    files = [f for f in files if f.endswith('PNG') or f.endswith('png') or f.endswith('JPG') or f.endswith('jpg')]

    if not len(files):
        sys.exit('No images found by the given path!')

    return files

print('evaluate best model on valid set....')
cls_list = ['cancer/', 'normal/']
pos=[]
neg=[]
TP=0
FP=0
TN=0
FN=0
Threshold=0.5

print('loading best model.....')
net = load_model(checkpoint_filepath)
print('model loaded.....')

datasets=['valid', 'train']
for dataset in datasets:
    for category in cls_list:
        data_path = os.path.join(DATASET_PATH+'/'+dataset, category)
        print(data_path)

        files = get_files(data_path)
        print('find {} images in {} set'.format(len(files), dataset))

        # loop through all files and make predictions
        for f in files:
            img = image.load_img(f, target_size=IMAGE_SIZE)
            if img is None:
                continue
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)
            pred = net.predict(x)[0]
            # print(f)
            # print(pred[0],pred[1])

            if category==cls_list[0]:
                # print('cancer: prediction = ', pred[0])
                if pred[0]>=Threshold:
                    TP+=1
                else:
                    FN+=1
                pos.append(pred[0])
            else:
                # print('normal: prediction = ', pred[0])
                if pred[0]<Threshold:
                    TN+=1
                else:
                    FP+=1
                neg.append(pred[0])

    score_pos=pos
    score_neg=neg
    y_neg = np.zeros(len(score_neg))
    y_pos = np.ones(len(score_pos))
    y = np.concatenate((y_pos,y_neg),axis=0)
    y = y.astype(int)
    y = y+1
    n_classes=2
    scores=np.concatenate((score_pos,score_neg),axis=0)

    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    roc_auc = metrics.auc(fpr, tpr)

    print('dataset TP FN TN FP = ', dataset, TP, FN, TN, FP)
    print('Sensitivity Specificity Accuracy = ', 100*TP/(TP+FN), 100*TN/(TN+FP), 100*(TP+TN)/(TP+TN+FP+FN))

    print('roc_auc = ', roc_auc)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (AUC =  %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - %s Classification' % dataset)
    plt.legend(loc="lower right")


    # create the axis of thresholds (scores)
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold',color='r')
    ax2.set_ylim([thresholds[-1],thresholds[0]])
    ax2.set_xlim([fpr[0],fpr[-1]])

    plotname = '{}-roc-thresh{}-{}-{}_bs{}_lr{}_ep{}_{}.jpg'.format('mammo', Threshold, MODEL, model_weights, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, dataset)
    plt.savefig(os.path.join(OUTPUT_DIR, plotname))

    plt.close()

# plot training curve
N = NUM_EPOCHS
fig = plt.figure()
# plt.plot(range(0, N), H.history['loss'])
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('Training loss curve of mammo classifier')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(['train', 'valid'], loc='upper left')
plt.tight_layout()
plotname = 'mammo-train_loss_curve-{}-{}-{}-{}_bs{}_lr{}_ep{}_is{}_fl{}_fold{}.jpg'.format(MODEL, model_weights, LOSS, OPTIMISER, BATCH_SIZE, \
    LEARNING_RATE, NUM_EPOCHS, IMAGE_SIZE[0], FREEZE_LAYERS, FOLD)
plt.savefig(os.path.join(OUTPUT_DIR, plotname))
plt.close(fig)

fig = plt.figure()
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title("Training accuracy curve of mammo classifier")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(['train', 'valid'], loc='upper left')
plt.tight_layout()
plotname = 'mammo-train_acc_curve-{}-{}-{}-{}_bs{}_lr{}_ep{}_is{}_fl{}_fold{}.jpg'.format(MODEL, model_weights, LOSS, OPTIMISER, BATCH_SIZE, \
    LEARNING_RATE, NUM_EPOCHS, IMAGE_SIZE[0], FREEZE_LAYERS, FOLD)
plt.savefig(os.path.join(OUTPUT_DIR, plotname))
plt.close(fig)

fig = plt.figure()
plt.plot(H.history['auc'])
plt.plot(H.history['val_auc'])
plt.title("Training auc curve of mammo classifier")
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(['train', 'valid'], loc='upper left')
plt.tight_layout()
plotname = 'auc-mammo-{}-{}_bs{}_lr{}_ep{}_is{}_fl{}_fold{}.jpg'.format(MODEL, model_weights, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, \
    IMAGE_SIZE[0], FREEZE_LAYERS, FOLD)
plt.savefig(os.path.join(OUTPUT_DIR, plotname))
plt.close(fig)
