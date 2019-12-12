import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, GRU

class PretrainedEmbedding(Layer):
    """Non-trainable embedding layer"""

    def __init__(self, embeddings, mask_zero=True, rate=0.1, **kwargs):
        self.embeddings = tf.constant(embeddings)
        self.mask_zero = mask_zero
        super(PretrainedEmbedding, self).__init__(**kwargs)

    def call(self, input, training=None):
         return tf.nn.embedding_lookup(self.embeddings, input)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        output_mask = K.not_equal(inputs, 0)
        return output_mask

    def get_config(self):
        config = super(PretrainedEmbedding, self).get_config()
        config.update({'embeddings': self.embeddings,
                       'mask_zero': self.mask_zero
                       })
        return config

class InputModule(Layer):

    def __init__(self,
                 units,
                 return_sequences=True,
                 **kwargs):
        super(InputModule, self).__init__(**kwargs)
        self.gru = GRU(units=units, return_sequences=return_sequences, **kwargs)

    def __call__(self, input, mask=None):
        return self.call(input=input, mask=mask)

    def call(self, input, mask=None):
        return self.gru(input, mask=mask)
