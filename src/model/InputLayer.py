import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, GRU
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# use tf.keras.preprocessing to

class TokenizerLayer(Layer):

    def __init__(self, tokenizer, max_len=30, **kwargs):
        self.tokenizer = tokenizer
        self.max_len = max_len
        super(TokenizerLayer, self).__init__(**kwargs)

    def __call__(self, input):
        return self.call(input)

    def call(self, input):
        if not isinstance(input, list):
            input = [input]
        res = self.tokenizer.fit_on_texts(input)
        return pad_sequences(res, maxlen=self.max_len, padding='post', truncating='post')

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
