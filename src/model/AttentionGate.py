import tensorflow as tf
from tensorflow.keras.layers import Layer, GRUCell, RNN
from tensorflow.keras import regularizers
from tensorflow.python.util import nest


class EpisodicMemoryCell(GRUCell):

    def __init__(self, units,
                 question,
                 m,
                 attention_units=4,
                 reg_scale=0.001,
                 trainable=True,
                 **kwargs):
        self.g = AttentionLayer(units=attention_units,
                                reg_scale=reg_scale,
                                trainable=trainable)
        self.question = question
        self.m = m
        super(EpisodicMemoryCell, self).__init__(units=units, **kwargs)

    def __call__(self, inputs, states, training):
        return self.call(inputs, states, training)

    def call(self, inputs, states, training=None):
        """
            Forward pass for the layer.
        :param question: The question' final state.
        :param training: Python boolean indicating whether the layer should behave in training mode or
                        inference mode, only relevant when `dropout` or `recurrent_dropout` is used.
        :param state: List of state tensors corresponding to the previous timestep
        :param inputs: a list of states for inputs until the current timestep.
        :return:
        """
        # Get scores
        scores = self.g((inputs, self.m, self.question))
        output, _ = super(EpisodicMemoryCell, self).call(inputs, states, training=training)
        h_t = tf.add(tf.multiply(scores, output), tf.multiply(tf.subtract(1, scores), states))
        return h_t, [h_t]

    def reset_dropout_mask(self):
        return super(EpisodicMemoryCell, self).reset_dropout_mask()

    def reset_recurrent_dropout_mask(self):
        return super(EpisodicMemoryCell, self).reset_recurrent_dropout_mask()

    def compute_output_shape(self, input_shape):
        # TODO: Implement this
        pass

class EpisodicModule(RNN):

    def __init__(self, units,
                 question,
                 m,
                 attention_layer_units,
                 reg_scale=0.001,
                 trainable=True,
                 initial_state=None,
                 **kwargs):
        self.cell = EpisodicMemoryCell(units=units,
                                       question=question,
                                       m=m,
                                      attention_layer_units=attention_layer_units,
                                      reg_scale=reg_scale,
                                      trainable=trainable)
        self.initial_states = initial_state
        self.trainable = trainable
        super(EpisodicModule, self).__init__(self.cell, **kwargs)

    def __call__(self,
             inputs,
             initial_state=None,
             masks=None,
             training=None,
             constants=None):
        return self.call(inputs, initial_state, masks, training, constants)

    def call(self,
             inputs,
             initial_state=None,
             masks=None,
             training=None,
             constants=None,
             **kwargs):

        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()
        return super(EpisodicModule, self).call(inputs,
                                                mask=masks,
                                                training=training,
                                                initial_state=initial_state,
                                                constants=constants,
                                                **kwargs)



class AttentionLayer(Layer):

    def __init__(self,
                 units=4,
                 reg_scale=0.001,
                 trainable=True,
                 **kwargs):
        self.attention_hidden_size = units
        self.reg_scale = reg_scale
        self.trainable = trainable
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        self.wb = self.add_weight(name="w_b",
                                  shape=[hidden_size, hidden_size],
                                  initializers='zeros',
                                  trainable=self.trainable,
                                  regularizers=regularizers.l2(self.reg_scale))
        self.w1 = self.add_weight(name="w_1",
                                  shape=[7 * hidden_size + 1, self.attention_hidden_size],
                                  trainable=self.trainable,
                                  regularizer=regularizers.l2(self.reg_scale))
        self.b1 = self.add_weight(name="b_1",
                                  trainable=self.trainable,
                                  shape=[1, self.attention_hidden_size])
        self.w2 = self.add_weight(name="w_2",
                                  shape=[hidden_size, 1],
                                  trainable=self.trainable,
                                  regularizers=regularizers.l2(self.reg_scale))
        self.b2 = self.add_weight(name="b_2",
                                  trainable=self.trainable,
                                  shape=[1, 1])

    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        c, m, q = inputs
        batch_size = c.shape[0]
        z = [c, m, q, tf.multiply(c, q),
             tf.multiply(c, m), tf.abs(tf.subtract(c, q)), tf.abs(tf.subtract(c, m)),
             tf.reshape(tf.reduce_sum(tf.multiply(tf.matmul(c, self.wb), q)), (batch_size, 1)),
             tf.reshape(tf.reduce_sum(tf.multiply(tf.matmul(c, self.wb), m)), (batch_size, 1))]

        z = tf.concat(values=z, axis=1)

        scores = tf.tanh(tf.add(tf.matmul(self.w1, z), self.b1))
        scores = tf.sigmoid(tf.add(tf.matmul(self.w2, scores), self.b2))

        return scores

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        hidden_size = input_shape[-1]
        return (batch_size, hidden_size)
