import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout

class PretrainedEmbedding(Layer):
    """Non-trainable embedding layer"""

    def __init__(self, embeddings, mask_zero=True, rate=0.1):
        self.embeddings = tf.constant(embeddings)
        self.mask_zero = mask_zero
        self.dropout = Dropout(rate=rate)

    def call(self, input, training=None):
        output = tf.nn.embedding_lookup(self.embeddings, input)
        return self.dropout(output, training=training)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        output_mask = K.not_equal(inputs, 0)
        return output_mask