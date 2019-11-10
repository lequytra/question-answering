import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras import regularizers, activations, initializers, constraints


class EpisodicMemoryCell(tf.keras.layers.GRUCell):

    def __init__(self, units, iter=3, **kwargs):
        self.iter = iter
        super(EpisodicMemoryCell, self).__init__(units=units, **kwargs)

    def build(self, input_shape):
        """
            Create a trainable weight variable for the layer
        :param input_shape:
        :return:
        """
        # TODO: implement this with multiple RNN cell
        pass

    def call(self, x):
        """
            Forward pass for the layer.
        :param x: a list of [c_t, h_{t - 1}, q]
        :return:
        """
        # TODO: Implement this
        pass

    def compute_output_shape(self, input_shape):
        # TODO: Implement this
        pass


class AttentionLayer:

    def __init__(self,
                 hidden_size,
                 attention_hidden_size=4,
                 reg_scale=0.001):
        self.wb = tf.get_variable("w_b",
                                  shape=[hidden_size, hidden_size],
                                  regularizers=regularizers.l2())
        self.w1 = tf.get_variable("w_1",
                                  shape=[7 * hidden_size + 1, attention_hidden_size],
                                  regularizer=regularizers.l2())
        self.b1 = tf.get_variable("b_1",
                                  shape=[1, attention_hidden_size])
        self.w2 = tf.get_variable("w_2",
                                  shape=[hidden_size, 1],
                                  regularizers=regularizers.l2())
        self.b2 = tf.get_variable("b_2",
                                  shape=[1, 1])

    def _calculate_scores(self, c, m, q):
        batch_size = c.shape[0]
        z = [c, m, q, tf.math.multiply(c, q),
             tf.math.multiply(c, m), tf.abs(tf.subtract(c, q)), tf.abs(tf.subtract(c, m)),
             tf.reshape(tf.reduce_sum(tf.multiply(tf.matmul(c, self.wb), q)), (batch_size, 1)),
             tf.reshape(tf.reduce_sum(tf.multiply(tf.matmul(c, self.wb), m)), (batch_size, 1))]

        z = tf.concat(values=z, axis=1)

        scores = tf.tanh(tf.add(tf.matmul(self.w1, z), self.b1))
        scores = tf.sigmoid(tf.add(tf.matmul(self.w2, scores), self.b2))

        return scores
