import os

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence as seq
from tensorflow.keras.preprocessing.text import Tokenizer as T
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import sequence_mask

import get_glove

MAX_CONTENT_LENGTH = 10
MAX_QUESTION_LENGTH = 30


# def pad_input(tokens_embeddings, max_len=MAX_CONTENT_LENGTH):
#     """
#         Turn a 2D array of token embeddings to have uniform length. Short sequences will
#         be padded and long sequences will be dropped to meet the length requirement.
#     :param tokens_embeddings: is a list of embeddings for sequences
#                         each sequence embeddings is a 2D array
#                         shape = (n_sequences, None, embedding_dim)
#     :param max_len: the maximum length of a sequence
#     :param paddings:
#         - 'zero': pad the remaining with 0 values.
#         - 'random': pad the remaining with random values based on Gaussian distribution.
#     :return: uniform: the padded 2D array
#     """
#     sequence_len = [len(sequence) for sequence in tokens_embeddings]
#     input_mask = sequence_mask(lengths=sequence_len, maxlen=max_len)
#
#     padded = seq.pad_sequences(sequences=tokens_embeddings,
#                                maxlen=max_len,
#                                padding='post')
#
#     return padded, input_mask
#
# def text_to_numeric_sequences(context, question, embedding_dict, dims=300, unknown_word=None):
#     """
#
#     :param context: list of strings
#     :param question: list of strings
#     :param embedding_dict:
#     :param dims: dimension of embeddings
#     :param unknown_word: a vector to represent unknown word.
#                         If None is pass, then initialized to all zeros.
#     :return:
#     """
#     text = context + question
#     t = Tokenizer()
#     t.fit_on_texts(text)
#     if not unknown_word:
#         unknown_word = np.random.rand(1, dims)
#     embedding_matrix = np.zeros(shape=(len(t.word_index) + 1, dims))
#     for word, index in t.word_index.items():
#         if word in embedding_dict:
#             embedding_matrix[index] = embedding_dict[word]
#         else:
#             embedding_matrix[index] = unknown_word
#
#     return t, embedding_matrix


def transform(input, max_len, tokenizer):
    if not isinstance(input, list):
        input = [input]
    res = tokenizer.fit_on_texts(input)
    return pad_sequences(res, maxlen=max_len, padding='post', truncating='post')

def get_embeddings(tokenizer, embed_dict, dim):
    embeddings_matrix = np.zeros(shape=(len(tokenizer.word_index) + 1, dim))
    for key, item in tokenizer.word_index.items():
        if key in embed_dict:
            embeddings_matrix[item] = embed_dict[key]

    return embeddings_matrix

def make_tokenizer(file_list):
    list_ = [os.path.join(file_list, i) for i in os.listdir(file_list) if i.endswith('.txt') ]

    t = T()
    for file in list_:
        with open(file, 'r') as f:
            lines = f.read().strip()
            t.fit_on_texts([lines])

    return t

def main(dim, embedding_folder, data_folder):
    # Load the embeddings vectors
    embedding_path = embedding_folder + 'glove.6B.{}d.txt'.format(dim)
    tokenizer = make_tokenizer(data_folder)
    # Reading embedding matrix
    print("Start loading embedding matrix")
    embed_dict = get_glove.load_vectors(embedding_path)
    embedding_matrix = get_embeddings(tokenizer, embed_dict, dim=dim)

    path = os.path.join(data_folder, 'special')

    if not os.path.isdir(path):
        os.mkdir(path)

    with open(os.path.join(path, 'embedding_matrix.npy'), 'wb') as f:
        embedding_matrix.dump(f)

    with open(os.path.join(path, 'tokenizer.p'), 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Finish creating embedding matrix and tokenizer")

    return

if __name__ == '__main__':
    embedding_path = "/home/stellasylee/Documents/CSC395/question-answering/script/data/glove/"
    data_path = "/home/stellasylee/Documents/CSC395/question-answering/script/data/merged/"
    main(300, embedding_path, data_path)
