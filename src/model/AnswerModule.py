from tensorflow.keras.layers import Layer, GRUCell, RNN
import tensorflow as tf
from tensorflow.keras import regularizers

class AnswerCell(GRUCell):

    def __init__(self, units,
                 question,
                 trainable=True,
                 **kwargs):
        super(AnswerCell, self).__init__(units=units, **kwargs)
        self.units = units
        self.question = question
        self.trainable = trainable

    def build(self, input_shape):
        # Get dim of question
        q_dim = self.question.shape[-1]
        new_dim = input_shape[-1] + q_dim
        input_shape = (input_shape[0], new_dim)
        super(AnswerCell, self).build(input_shape)

    def call(self, inputs, states, training=None):
        input = tf.concat([inputs, self.question], axis=1)
        output, new_states = super(AnswerCell, self).call(input, states, training)
        return output, [new_states]


class AnswerModule(RNN):
    # TODO: initial state is initialized to the last memory
    def __init__(self, units, question, **kwargs):
        self.cell = AnswerCell(units=units, question=question)
        super(AnswerModule, self).__init__(cell=self.cell, **kwargs)

    # def __call__(self, inputs, masks=None, training=None, initial_state=None, constants=None):
    #     return self.call(inputs=inputs,
    #                      masks=masks,
    #                      training=training,
    #                      initial_state=initial_state,
    #                      constants=constants)
    
    def call(self, inputs, mask=None, training=None, initial_state=None, constants=None):
        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()
        return super(AnswerModule, self).call(inputs,
                                              mask=mask,
                                              training=training,
                                              initial_state=initial_state,
                                              constants=constants)


class LinearRegression(Layer):

    def __init__(self,
                 units=4,
                 reg_scale=0.001,
                 trainable=True,
                 **kwargs):
        self.units = units
        self.reg_scale = reg_scale
        self.trainable = trainable
        super(LinearRegression, self).__init__(**kwargs)

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        self.wa = self.add_weight(name="w_a",
                                  shape=[hidden_size, self.units],
                                  initializer='zeros',
                                  trainable=self.trainable,
                                  regularizer=regularizers.l2(self.reg_scale))

        super(LinearRegression, self).build(input_shape)

    def call(self, inputs):
        return tf.nn.softmax(tf.matmul(inputs, self.wa))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)