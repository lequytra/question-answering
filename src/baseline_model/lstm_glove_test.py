from __future__ import print_function

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import os
import pickle
import sys
sys.path.append('../')
from preprocessing.preprocessing import transform

data_path = '/home/stellasylee/Documents/CSC395/question-answering/data/merged'

BATCH_SIZE = 32 # batch size for training
NBR_EPOCHS = 100
MAX_CONTEXT = 50
MAX_QUESTION = 30
TASK_NBR = 5 # choose the task number

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

# Restore the model and construct the encoder and decoder.
model = load_model('lstm_glove_train{}.h5'.format(TASK_NBR))

# Get Test data
with open(os.path.join(data_path, 'Context_Test_{}.txt'.format(TASK_NBR)), 'r') as f:
    context = f.read().strip().split('\n')
with open(os.path.join(data_path, 'Question_Test_{}.txt'.format(TASK_NBR)), 'r') as f:
    question = f.read().strip().split('\n')
with open(os.path.join(data_path, 'Answer_Test_{}.txt'.format(TASK_NBR)), 'r') as f:
    answer = f.read().strip().split('\n')

context = transform(context, max_len=MAX_CONTEXT, tokenizer=tokenizer)
question = transform(question, max_len=MAX_QUESTION, tokenizer=tokenizer)
answer = transform(answer, max_len=1, tokenizer=tokenizer)
encoded_answer = to_categorical(tf.squeeze(answer, axis=1), num_classes=vocab_size)
loss, acc = model.evaluate([context, question], encoded_answer, batch_size=BATCH_SIZE)
print("Test Task {}: loss/accuracy = {:.4f}, {:.4f}".format(TASK_NBR, loss, acc))
# Test Task 1: loss/accuracy = 0.5450, 0.9040
# Test Task 2: loss/accuracy = 3.9351, 0.2050
# Test Task 3: loss/accuracy = 4.6930, 0.1910
# Test Task 4: loss/accuracy = 1.7926, 0.1710
# Test Task 5: loss/accuracy = 2.8622, 0.3950