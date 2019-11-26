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
        super(EpisodicMemoryCell, self).__init__(units=units, **kwargs)
        self.g = AttentionLayer(units=attention_units,
                                reg_scale=reg_scale,
                                trainable=trainable)
        self.question = question
        self.m = m

    # def __call__(self, inputs, states, training):
    #     return self.call(inputs, states, training)

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
        output, last_output = super(EpisodicMemoryCell, self).call(inputs, states, training=training)
        h_t = tf.add(tf.multiply(scores, output), tf.multiply(tf.subtract(1.0, scores), states[0]))
        return h_t, [h_t]

    def reset_dropout_mask(self):
        return super(EpisodicMemoryCell, self).reset_dropout_mask()

    def reset_recurrent_dropout_mask(self):
        return super(EpisodicMemoryCell, self).reset_recurrent_dropout_mask()

    def compute_output_shape(self, input_shape):
        return super(EpisodicMemoryCell, self).compute_output_shape(input_shape=input_shape)

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
                                      attention_units=attention_layer_units,
                                      reg_scale=reg_scale,
                                      trainable=trainable)
        self.initial_states = initial_state
        self.trainable = trainable
        super(EpisodicModule, self).__init__(self.cell, **kwargs)

    # def __call__(self,
    #          inputs,
    #          initial_state=None,
    #          mask=None,
    #          training=None):
    #     return self.call(inputs, initial_state, mask, training)

    def call(self,
             inputs,
             initial_state=None,
             mask=None,
             training=None,
             **kwargs):

        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()
        print(inputs)
        return super(EpisodicModule, self).call(inputs,
                                                mask=mask,
                                                training=training,
                                                initial_state=initial_state,
                                                **kwargs)

    def compute_output_shape(self, input_shape):
        return super(EpisodicModule, self).compute_output_shape(input_shape)

class AttentionLayer(Layer):

    def __init__(self,
                 units=4,
                 reg_scale=0.001,
                 trainable=True,
                 **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_hidden_size = units
        self.reg_scale = reg_scale
        self.trainable = trainable

    def build(self, input_shape):
        input_shape = input_shape[0]
        hidden_size = input_shape[-1]

        self.wb = self.add_weight(name="wb",
                                  shape=(hidden_size, hidden_size),
                                  initializer='zeros',
                                  trainable=self.trainable,
                                  regularizer=regularizers.l2(self.reg_scale))

        self.w1 = self.add_weight(name="w1",
                                  shape=(7 * hidden_size + 2, self.attention_hidden_size),
                                  trainable=self.trainable,
                                  regularizer=regularizers.l2(self.reg_scale))

        self.b1 = self.add_weight(name="b1",
                                  trainable=self.trainable,
                                  shape=(1, self.attention_hidden_size))

        self.w2 = self.add_weight(name="w2",
                                  shape=(self.attention_hidden_size, 1),
                                  trainable=self.trainable,
                                  regularizer=regularizers.l2(self.reg_scale))
        self.b2 = self.add_weight(name="b2",
                                  trainable=self.trainable,
                                  shape=(1, 1))

        super(AttentionLayer, self).build(input_shape)

    def call(self, input):
        c, m, q = input
        # Find the number of time steps in input
        n_timestep = c.shape[1]
        # Tile the context and question states for broadcasting
        # tf.print(tf.rank(c))
        # if tf.rank(c) > 2:
        #     m = tf.tile(input=m, multiples=[1, n_timestep, 1])
        #     q = tf.tile(input=q, multiples=[1, n_timestep, 1])

        wb_q = tf.expand_dims(tf.reduce_sum(tf.multiply(tf.matmul(c, self.wb), q), axis=-1), axis=-1)
        wb_m = tf.expand_dims(tf.reduce_sum(tf.multiply(tf.matmul(c, self.wb), m), axis=-1), axis=-1)

        z = [c, m, q, tf.multiply(c, q),
             tf.multiply(c, m), tf.abs(tf.subtract(c, q)), tf.abs(tf.subtract(c, m)), wb_q, wb_m]

        z = tf.concat(values=z, axis=-1)

        scores = tf.tanh(tf.add(tf.matmul(z, self.w1), self.b1))
        scores = tf.sigmoid(tf.add(tf.matmul(scores, self.w2), self.b2))

        return scores

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return (batch_size, 1)
