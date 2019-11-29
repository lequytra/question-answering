import csv
import os
import numpy as np

NUM_TOKENS = 100000


def load_vectors(filepath):
    """
        Load Glove text file to dictionary of {word: embedding}
    :return:
    """

    file = csv.reader(open(filepath), delimiter=' ', quoting=csv.QUOTE_NONE)

    glove = {line[0]: np.asarray(list(map(float, line[1:]))) for line in file}

    return glove


def load_embedding_matrix(data_folder, dims=100):
    """
        Load glove embedding vectors/dictionary to an embeddings matrix and create word-index mapping
    :param data_folder: path to folder contain Glove files
    :param dims: the dim of the embedding vector
    :return: the word-to-indices mapping
                embedding matrix
    """
    path = os.path.join(data_folder, 'glove.6B.' + str(dims) + 'd.txt')
    file = csv.reader(open(path), delimiter=' ', quoting=csv.QUOTE_NONE)

    mapping = {}
    # Initialize the embeddings
    embeddings = np.zeros(shape=(NUM_TOKENS + 1, dims))

    for c, line in enumerate(file):
        mapping[line[0]] = c
        vec = np.asarray(list(map(float, line[1:])))
        embeddings[c] = vec

    # Create a random vector for unknown words
    unk_vec = np.random.rand(dims)
    end = len(file)
    embeddings[end] = unk_vec
    mapping['<UNK>'] = end

    return mapping, embeddings
