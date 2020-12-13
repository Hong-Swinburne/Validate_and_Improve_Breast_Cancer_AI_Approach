"""
This script goes along my blog post:
Keras InceptionResetV2 (https://jkjung-avt.github.io/keras-inceptionresnetv2/)
"""


from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.callbacks import ModelCheckpoint, TensorBoard
import datetime


DATASET_PATH  = '/home/peter/ML/Projects/Mammo/Mamm_ML/Data_reviewed/whole_mammo_dir/'
IMAGE_SIZE    = (1024, 1024)
NUM_CLASSES   = 2
BATCH_SIZE    = 2  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 25
WEIGHTS_FINAL = 'model-inception_resnet_v2-input-size_1024_tensorboard.h5'

try:
  os.mkdir('checkpoints')
except:
  print('dir present')

try:
  os.mkdir('tensorboard')
except:
  print('dir present')



# class NewCallback:

#     def __init__(self):

#         self.use_existing = None # './weights-ModelD_0.92.hdf5'  # '/path/2/your/weights.hdf5'

#         self.checkpoint_path = './checkpoints/'
#         self.tensorboard_path = './tensorboard/'

#     def save(self):

#         # Creat checkpoint name from date and time
#         now = datetime.datetime.now()
#         year = now.year
#         month = now.month
#         day = now.day
#         hour = now.hour
#         minute = now.minute
#         name = str(year) + '_' + str(month) + '_' + str(day) + '_' + str(hour) + '_' + str(minute)
#         os.mkdir(os.path.join(self.checkpoint_path, name))

#         # Create checkpoint callback
#         checkpoint_name = os.path.join(self.checkpoint_path,
#                                        name,
#                                        'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5')

#         self.checkpoint = ModelCheckpoint(checkpoint_name,
#                                           monitor='val_loss',
#                                           save_best_only=False,
#                                           save_weights_only=False)

#         # Create tensorboard callback
#         self.tensorboard = TensorBoard(log_dir=self.tensorboard_path,
#                                        histogram_freq=0,
#                                        write_graph=True,
#                                        write_images=True)


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

# build our classifier model based on pre-trained InceptionResNetV2:
# 1. we don't include the top (fully connected) layers of InceptionResNetV2
# 2. we add a DropOut layer followed by a Dense (fully connected)
#    layer which generates softmax class score for each class
# 3. we compile the final model using an Adam optimizer, with a
#    low learning rate (since we are 'fine-tuning')
net = InceptionResNetV2(include_top=False,
                        weights='imagenet',
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
net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
print(net_final.summary())

model_path='checkpoints_wholem/'
try:
  os.mkdir(model_path)
except:
  print('dir present')

tensorboard_path='tensorboard_wholem/'
try:
  os.mkdir(tensorboard_path)
except:
  print('dir present')


callbacks = [
    ModelCheckpoint(filepath=model_path+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)+'.h5', monitor="val_acc", save_best_only=False, save_weights_only=False),
    TensorBoard(log_dir=tensorboard_path,
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True)
,
    # ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2)
  ]


# train the model
net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
                        callbacks=callbacks)

# save trained weights
net_final.save(WEIGHTS_FINAL)
