import sys
import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, RepeatVector, Embedding, LSTM
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer as T

data_path = 'home/tranle/Desktop/CSC395/question-answering/data/merged'

BATCH_SIZE = 32 # batch size for training
NBR_EPOCHS = 50

###################################
#       Loading dataset           #
###################################

# Get tokenizer
with open(os.path.join(data_path, '/special/tokenizer.p'), 'rb') as f:
    tokenizer = pickle.loads(f)
vocab_size = len(tokenizer.word_index) + 1

# Get embedding matrix
with open(os.path.join(data_path, '/special/embedding_matrix.npy'), 'rb') as f:
    embedding_matrix = pickle.loads(f)

# Get Training data
with open(os.path.join(data_path, '/Answer_Train.txt'), 'rb') as f:
    train_answer = pickle.loads(f)

context_model = Sequential()
context_model.add(Embedding(vocab_size, output_dim=300, weights = [embedding_matrix]))
#story_rnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=story_maxlen))
context_model.add(Dropout(0.3))

# summarize the model
print(context_model.summary())