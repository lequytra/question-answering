import sys
sys.path.append('../')
from preprocessing.preprocessing import transform
import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, \
     TensorBoard, RemoteMonitor, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, RepeatVector, Embedding, LSTM
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np
import time

data_path = '/home/tranle/Desktop/GrinnellCollege/CSC395/question-answering/data/merged'

BATCH_SIZE = 32 # batch size for training
NBR_EPOCHS = 200
MAX_CONTEXT = 50
MAX_QUESTION = 20

###################################
#       Loading dataset           #
###################################

# Get tokenizer
with open(os.path.join(data_path, 'special/tokenizer.p'), 'rb') as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.word_index) + 1

# Get embedding matrix
with open(os.path.join(data_path, 'special/embedding_matrix.npy'), 'rb') as f:
    embedding_matrix = np.load(f, allow_pickle=True)

# Get Training data
with open(os.path.join(data_path, '../merged10/Context_Train_2.txt'), 'r') as f:
    context = f.read().strip().split('\n')
with open(os.path.join(data_path, '../merged10/Question_Train_2.txt'), 'r') as f:
    question = f.read().strip().split('\n')
with open(os.path.join(data_path, '../merged10/Answer_Train_2.txt'), 'r') as f:
    answer = f.read().strip().split('\n')

context = transform(context, max_len=MAX_CONTEXT, tokenizer=tokenizer)
question = transform(question, max_len=MAX_QUESTION, tokenizer=tokenizer)
answer = transform(answer, max_len=1, tokenizer=tokenizer)

# generate one hot encode
encoded_answer = to_categorical(tf.squeeze(answer, axis=1), num_classes=vocab_size)

###################################
#       Model                     #
###################################
context_model = Sequential()
context_model.add(Embedding(vocab_size, output_dim=300, weights = [embedding_matrix],
                            input_length=MAX_CONTEXT, trainable=False))
context_model.add(Dropout(0.3))
# summarize the model
print(context_model.summary())

# generate embeddings for question and make adaptable to story
question_model = Sequential()
question_model.add(Embedding(vocab_size, output_dim=300, weights = [embedding_matrix],
                             input_length=MAX_QUESTION, trainable=False))
question_model.add(Dropout(0.3))
question_model.add(LSTM(300, return_sequences=False))
question_model.add(RepeatVector(MAX_CONTEXT))

print(question_model.summary())

# merge the two
merged_model = add([context_model.output, question_model.output])

model_combined = Sequential()
model_combined.add(LSTM(300, return_sequences=False))
model_combined.add(Dropout(0.3))
model_combined.add(Dense(vocab_size, activation="softmax"))

# combine models
final_model = Model([context_model.input, question_model.input], model_combined(merged_model))
print(final_model.summary())
final_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy", "mae"])

print("Training...")

path = os.getcwd()
checkpoint_dir = os.path.join(path, 'checkpoints')
log_dir = os.path.join(path, 'logs')

if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

checkpoints = ModelCheckpoint(filepath='weights.epoch{epoch:02d}-val_loss{val_loss:.2f}.hdf5',
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
callbacks = [tensorboard, reduce_lr]

final_model.fit(x = [context, question],  y=encoded_answer,
          batch_size=BATCH_SIZE, epochs=NBR_EPOCHS, validation_split=0.2, callbacks= callbacks)
