import numpy as np

MAX_CONTENT_LENGTH = 10
MAX_QUESTION_LENGTH = 30


def uniform_size(tokens_embeddings, max_len=MAX_CONTENT_LENGTH, paddings='zero'):
    """
        Turn a 2D array of token embeddings to have uniform length. Short sequences will
        be padded and long sequences will be dropped to meet the length requirement.
    :param tokens_embeddings: a 2D numpy array of tokens embeddings
    :param max_len: the maximum length of a sequence
    :param paddings:
        - 'zero': pad the remaining with 0 values.
        - 'random': pad the remaining with random values based on Gaussian distribution.
    :return: uniform: the padded 2D array
    """
    if paddings != 'zero' and paddings != 'random':
        print("Argument for paddings is not recognized.")
        return

    dims = tokens_embeddings.shape[1]

    if paddings == 'zero':
        uniform = np.zeros(shape=(max_len, dims))
    else:
        uniform = np.random.rand(max_len, dims)

    len_ = min(tokens_embeddings.shape[0], max_len)
    uniform = tokens_embeddings[:len_]
    return uniform


def pad_question(q_tokens, paddings='zero'):
    """
        Pad question tokens
    :param q_tokens: a 2D numpy array of question tokens' embeddings
    :param paddings:
        - 'zero': pad the remaining with 0 values.
        - 'random': pad the remaining with random values based on Gaussian distribution.
    :return: uniform: the padded 2D question
    """
    return uniform_size(q_tokens, MAX_QUESTION_LENGTH, paddings)


def pad_context(c_tokens, paddings='zero'):
    """
        Pad question tokens
    :param c_tokens: a 2D numpy array of context tokens' embeddings
    :param paddings:
        - 'zero': pad the remaining with 0 values.
        - 'random': pad the remaining with random values based on Gaussian distribution.
    :return: uniform: the padded 2D question
    """
    return uniform_size(c_tokens, MAX_CONTENT_LENGTH, paddings)
