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
NBR_EPOCHS = 200
MAX_CONTEXT = 50
MAX_QUESTION = 30
TASK_NBR = 3 # choose the task number
EMBED_HIDDEN_SIZE = 300 # size of the vector space (word embedding)

###################################
#       Loading dataset           #
###################################
# Get tokenizer
with open(os.path.join(data_path, 'special/tokenizer.p'), 'rb') as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.word_index) + 1

# Restore the model and construct the encoder and decoder.
model = load_model('lstm_train{}.h5'.format(TASK_NBR))

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
# Test Task 1: loss/accuracy = 3.9904, 0.4360
# Test Task 2: loss/accuracy = 3.6211, 0.2090
# Test Task 3: loss/accuracy = 3.1279, 0.1610
