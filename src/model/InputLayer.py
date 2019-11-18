import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, GRU
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PretrainedEmbedding(Layer):
    """Non-trainable embedding layer"""

    def __init__(self, embeddings, mask_zero=True, rate=0.1, **kwargs):
        self.embeddings = tf.constant(embeddings)
        self.mask_zero = mask_zero
        self.dropout = Dropout(rate=rate)
        super(PretrainedEmbedding, self).__init__(**kwargs)

    def __call__(self, input, training=None):
        return self.call(input, training)

    def call(self, input, training=None):
        output = tf.nn.embedding_lookup(self.embeddings, input)
        return self.dropout(output, training=training)

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
                 tokenizer,
                 embedding,
                 max_len=30,
                 mask_zero=True,
                 rate=0.1,
                 return_sequences=True,
                 **kwargs):
        super(InputModule, self).__init__(**kwargs)
        self.embedding = PretrainedEmbedding(embeddings=embedding,
                                             mask_zero=mask_zero,
                                             rate=rate)
        self.tokenizer = TokenizerLayer(tokenizer=tokenizer, max_len=max_len)
        self.gru = GRU(units=units, return_sequences=return_sequences, **kwargs)

    def __call__(self, input, mask=None):
        return self.call(input=input, mask=mask)

    def call(self, input, mask=None):
        tokenized = self.tokenizer(input)
        embedding = self.embedding(tokenized)

        if mask is None:
            mask = self.embedding.compute_mask(inputs=input)

        return self.gru(embedding, mask=mask)
