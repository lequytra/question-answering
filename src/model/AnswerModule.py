from tensorflow.keras.layers import Layer, GRUCell, RNN
import tensorflow as tf
from tensorflow.keras import regularizers

class AnswerCell(GRUCell):

    def __init__(self, units,
                 n_class,
                 question,
                 trainable=True,
                 reg_scale=0.001,
                 **kwargs):
        super(AnswerCell, self).__init__(units=units, **kwargs)
        self.linear = LinearRegression(n_class, reg_scale)
        self.units = units
        self.question = question
        self.trainable = trainable

    def build(self, input_shape):
        super(AnswerCell, self).build(input_shape)

    def __call__(self, inputs, states, training=None):
        return self.call(inputs, states, training)

    def call(self, inputs, states, training=None):
        input = tf.concat([inputs, self.question], axis=1)
        output, new_states = super(AnswerCell, self).call(input, states, training)
        output = self.linear(output)
        return output, [new_states]


class AnswerModule(RNN):
    # TODO: initial state is initialized to the last memory
    def __init__(self, units, question, **kwargs):
        self.cell = AnswerCell(units=units, question=question)
        super(AnswerModule, self).__init__(**kwargs)

    def __call__(self, inputs, masks=None, training=None, initial_state=None, constants=None):
        return self.call(inputs=inputs,
                         masks=masks,
                         training=training,
                         initial_state=initial_state,
                         constants=constants)
    
    def call(self, inputs, masks=None, training=None, initial_state=None, constants=None):
        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()
        return super(AnswerModule, self).call(inputs,
                                              mask=masks,
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
                                  initializers='zeros',
                                  trainable=self.trainable,
                                  regularizers=regularizers.l2(self.reg_scale))

    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        return tf.nn.softmax(tf.matmul(inputs, self.wa))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)