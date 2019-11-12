# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.layers import Dense, Dropout, RepeatVector
# Merge need to import subclass directly
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import os

import getbAbi

BABI_DIR = "/home/stellasylee/Documents/CSC395/question-answering/script/data/en"
TASK_NBR = 1
EMBED_HIDDEN_SIZE = 50
BATCH_SIZE = 32
NBR_EPOCHS = 40

train_file, test_file = getbAbi.get_files_for_task(TASK_NBR, BABI_DIR)

data_train = getbAbi.get_stories(os.path.join(BABI_DIR, train_file))
data_test = getbAbi.get_stories(os.path.join(BABI_DIR, test_file))

word2idx = getbAbi.build_vocab([data_train, data_test])
vocab_size = len(word2idx) + 1
print("vocab_size=", vocab_size)

