import sys
sys.path.append('../')
import preprocessing.preprocessing as Preprocess
import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, RepeatVector, Embedding, LSTM
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer as T
import numpy as np

data_path = '/home/stellasylee/Documents/CSC395/question-answering/script/data/merged'

BATCH_SIZE = 32 # batch size for training
NBR_EPOCHS = 50
MAX_CONTEXT = 30
MAX_QUESTION = 10

###################################
#       Loading dataset           #
###################################

# Get tokenizer
with open(os.path.join(data_path, 'special/tokenizer.p'), 'rb') as f:
    tokenizer = pickle.loads(f)
    # TODO: TypeError: a bytes-like object is required, not '_io.BufferedReader'
vocab_size = len(tokenizer.word_index) + 1

# Get embedding matrix
with open(os.path.join(data_path, 'special/embedding_matrix.npy'), 'rb') as f:
    embedding_matrix = np.load(f, allow_pickle=True)

# Get Training data
with open(os.path.join(data_path, 'Context_Train.txt'), 'r') as f:
    context = f.read().strip().split('\n')
with open(os.path.join(data_path, 'Question_Train.txt'), 'r') as f:
    question = f.read().strip().split('\n')
with open(os.path.join(data_path, 'Answer_Train.txt'), 'r') as f:
    answer = f.read().strip().split('\n')

context = Preprocess.transform(context, max_len=MAX_CONTEXT, tokenizer=tokenizer)
question = Preprocess.transform(question, max_len=MAX_QUESTION, tokenizer=tokenizer)
answer = Preprocess.transform(answer, max_len=1, tokenizer=tokenizer)

###################################
#       Model                     #
###################################
context_model = Sequential()
context_model.add(Embedding(vocab_size, output_dim=300, weights = [embedding_matrix]))
#story_rnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=story_maxlen))
context_model.add(Dropout(0.3))

# summarize the model
print(context_model.summary())