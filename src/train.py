import sys
import os
import pickle
import time
import numpy as np


import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, \
    LearningRateScheduler, TensorBoard, RemoteMonitor, History, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

code_path = ['home/tranle/Desktop/CSC395/question-answering/src/model',
             'home/tranle/Desktop/CSC395/question-answering/src/preprocessing']
data_path = '~/Desktop/GrinnellCollege/CSC395/question-answering/data/merged'
if not code_path in sys.path:
    sys.path.append(code_path)

from model.DMN import *
from preprocessing.preprocessing import transform

os.environ['KMP_DUPLICATE_LIB_OK']='True'

##################################
#           Hyper params         #
##################################

MASK_ZERO = True
LEARNING_RATE = 0.001
OPTIMIZER = 'rmsprop'
BATCH_SIZE = 32
NUM_EPOCHS = 50
MAX_CONTEXT = 30
MAX_QUESTION = 10
n_answer = 0
###################################
#       Loading dataset           #
###################################

# Get tokenizer
with open(os.path.join(data_path, 'special/tokenizer.p'), 'rb') as f:
    tokenizer = pickle.loads(f)

with open(os.path.join(data_path, 'special/embedding_matrix.npy'), 'rb') as f:
    embeddings = np.load(f)

with open(os.path.join(data_path, 'Context_Train.txt'), 'r') as f:
    context = f.read().strip().split('\n')
with open(os.path.join(data_path, 'Question_Train.txt'), 'r') as f:
    question = f.read().strip().split('\n')
with open(os.path.join(data_path, 'Answer_Train.txt'), 'r') as f:
    answer = f.read().strip().split('\n')
n_answer = len(answer)
context = transform(context, max_len=MAX_CONTEXT, tokenizer=tokenizer)
question = transform(question, max_len=MAX_QUESTION, tokenizer=tokenizer)
answer = transform(answer, max_len=1, tokenizer=tokenizer)

###################################
#          Model                  #
###################################

model = DMN(n_answer, embeddings, mask_zero=MASK_ZERO, trainable=True)

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

history = model.fit(x={'in_context': context, 'in_question': question},
                    y=answer,
                    batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    verbose=2,
                    callbacks=callbacks,
                    validation_split=validation_split)
