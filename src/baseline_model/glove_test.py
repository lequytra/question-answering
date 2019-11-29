import tensorflow
import os
import numpy as np
GLOVE_DIR = "/home/stellasylee/Documents/CSC395/question-answering/script/data/glove.6B"

def get_gloVe_embedding(gloVe_dir):
    """
    Load GloVe word embedding file to memory as dictionary of word to embedding array
    :param gloVe_dir: path of parent directory where glove embedding is located
    :return:  embeddings_index
    """
    embeddings_index = {}
    glove = open(os.path.join(GLOVE_DIR, "glove.6B.300d.txt"), encoding='utf-8')
    for line in glove:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    glove.close()
    print("found %s word vectors." % len(embeddings_index))
    print(len(embeddings_index[list(embeddings_index.keys())[0]]))
    return embeddings_index

def get_embedding_matrix(embedding_index, word_index, dim=300):
    """
    Create embedding matrix
    :param embedding_index:
    :param word_index:
    :param dim: dimension of GloVe embedding vectors
    :return:
    """
    # TODO: Build the embedding matrix for gloVe embedding layer
    return True

if __name__ == '__main__':
    embed = get_gloVe_embedding(GLOVE_DIR)
    print(embed[list(embed.keys())[0]])
    print(embed[list(embed.keys())[1]])




