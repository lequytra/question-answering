import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence as seq
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import sequence_mask

MAX_CONTENT_LENGTH = 10
MAX_QUESTION_LENGTH = 30


def pad_input(tokens_embeddings, max_len=MAX_CONTENT_LENGTH):
    """
        Turn a 2D array of token embeddings to have uniform length. Short sequences will
        be padded and long sequences will be dropped to meet the length requirement.
    :param tokens_embeddings: is a list of embeddings for sequences
                        each sequence embeddings is a 2D array
                        shape = (n_sequences, None, embedding_dim)
    :param max_len: the maximum length of a sequence
    :param paddings:
        - 'zero': pad the remaining with 0 values.
        - 'random': pad the remaining with random values based on Gaussian distribution.
    :return: uniform: the padded 2D array
    """
    sequence_len = [len(sequence) for sequence in tokens_embeddings]
    input_mask = sequence_mask(lengths=sequence_len, maxlen=max_len)

    padded = seq.pad_sequences(sequences=tokens_embeddings,
                               maxlen=max_len,
                               padding='post')

    return padded, input_mask

def text_to_numeric_sequences(context, question, embedding_dict, dims=300, unknown_word=None):
    """

    :param context: list of strings
    :param question: list of strings
    :param embedding_dict:
    :param dims: dimension of embeddings
    :param unknown_word: a vector to represent unknown word.
                        If None is pass, then initialized to all zeros.
    :return:
    """
    text = context + question
    t = Tokenizer()
    t.fit_on_texts(text)
    if not unknown_word:
        unknown_word = np.random.rand(1, dims)
    embedding_matrix = np.zeros(shape=(len(t.word_index) + 1, dims))
    for word, index in t.word_index.items():
        if word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]
        else:
            embedding_matrix[index] = unknown_word

    return t, embedding_matrix
