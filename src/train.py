import sys
import os
code_path = 'home/tranle/Desktop/CSC395/question-answering/src/model'
data_path = 'home/tranle/Desktop/CSC395/question-answering/data/merged'
if not code_path in sys.path:
    sys.path.append(code_path)
import time
import tensorflow as tf
from DMN import *
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, \
    LearningRateScheduler, TensorBoard, RemoteMonitor, History, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import pickle

os.environ['KMP_DUPLICATE_LIB_OK']='True'

##################################
#           Hyper params         #
##################################

n_class = 0
embeddings = None
MASK_ZERO = True
LEARNING_RATE = 0.001
OPTIMIZER = 'rmsprop'
BATCH_SIZE = 32

###################################
#       Loading dataset           #
###################################

# Get tokenizer
with open(os.path.join(data_path, 'tokenizer.p'), 'rb') as f:
    tokenizer = pickle.loads(f)



###################################
#          Model                  #
###################################

model = DMN(n_class, embeddings, mask_zero=MASK_ZERO, trainable=True)

if OPTIMIZER == 'rmsprop':
    op = RMSprop(learning_rate=LEARNING_RATE)
else:
    op = Adam(learning_rate=LEARNING_RATE)

print("Compiling the model ... ")

model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])

path = os.getcwd()
checkpoint_dir = os.path.join(path, 'checkpoints')
log_dir = os.path.join(path, 'logs')

if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

# Initialize Keras callbacks
checkpoints = ModelCheckpoint(filepath='/weights.epoch{epoch:02d}-val_loss{val_loss:.2f}.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='min',
                             period=1)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=15,
                               verbose=1)

csv_logger = CSVLogger(filename='training_log.csv',
                       separator=',',
                       append=True)

remote = RemoteMonitor()

tensorboard = TensorBoard(log_dir="{}/{}".format(log_dir, time.time()),
                          histogram_freq=0,
                          batch_size=BATCH_SIZE,
                          write_graph=True,
                          write_grads=True,
                          write_images=True,
                          update_freq='batch')

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=15,
                                         min_lr=0.001)

callbacks = [checkpoints,
            early_stopping,
            csv_logger,
            tensorboard,
            reduce_lr,
            remote]

validation_split = 0.2

history = model.fit()