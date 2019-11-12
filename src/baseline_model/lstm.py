# Modified from
# https://github.com/sujitpal/dl-models-for-qa/blob/master/src/babi-lstm.py
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.layers import Dense, Dropout, RepeatVector, merge
import keras.layers
#import keras.layers
# Merge need to import subclass directly
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
import os

import getbAbi

BABI_DIR = "/home/stellasylee/Documents/CSC395/question-answering/script/data/en"
TASK_NBR = 1 # choose the task number
EMBED_HIDDEN_SIZE = 50
BATCH_SIZE = 32
NBR_EPOCHS = 40

train_file, test_file = getbAbi.get_files_for_task(TASK_NBR, BABI_DIR)

data_train = getbAbi.get_stories(os.path.join(BABI_DIR, train_file))
data_test = getbAbi.get_stories(os.path.join(BABI_DIR, test_file))

word2idx = getbAbi.build_vocab([data_train, data_test])
vocab_size = len(word2idx) + 1
print("vocab_size=", vocab_size)

story_maxlen, question_maxlen = getbAbi.get_maxlens([data_train, data_test])
print("story_maxlen=", story_maxlen)
print("question_maxlen=", question_maxlen)

Xs_train, Xq_train, Y_train = getbAbi.vectorize(data_train, word2idx, story_maxlen, question_maxlen)
Xs_test, Xq_test, Y_test = getbAbi.vectorize(data_test, word2idx, story_maxlen, question_maxlen)
print(Xs_train.shape, Xq_train.shape, Y_train.shape)
print(Xs_test.shape, Xq_test.shape, Y_test.shape)

# define model
# generate embeddings for stories
story_rnn = Sequential()
story_rnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=story_maxlen))
story_rnn.add(Dropout(0.3))

print("Generate embeddings for story")

# generate embeddings for question and make adaptable to story
question_rnn = Sequential()
question_rnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=question_maxlen))
question_rnn.add(Dropout(0.3))
question_rnn.add(LSTM(EMBED_HIDDEN_SIZE, return_sequences=False))
question_rnn.add(RepeatVector(story_maxlen))

print("Generate embeddings for questions")

# merge the two
merged_model = keras.layers.add([story_rnn.output, question_rnn.output])

model_combined = Sequential()
model_combined.add(LSTM(EMBED_HIDDEN_SIZE, return_sequences=False))
model_combined.add(Dropout(0.3))
model_combined.add(Dense(vocab_size, activation="softmax"))

# combine models 
final_model = Model([story_rnn.input, question_rnn.input], model_combined(merged_model))

final_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("Training...")

final_model.fit([Xs_train, Xq_train], Y_train, 
          batch_size=BATCH_SIZE, nb_epoch=NBR_EPOCHS, validation_split=0.05)
loss, acc = final_model.evaluate([Xs_test, Xq_test], Y_test, batch_size=BATCH_SIZE)
print("Test loss/accuracy = {:.4f}, {:.4f}".format(loss, acc))

