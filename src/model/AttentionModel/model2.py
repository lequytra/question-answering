import tensorflow as tf
from tensorflow.keras.layers import GRU, Attention, Input, Dense, Bidirectional, LSTM, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

from model.InputLayer import PretrainedEmbedding

INPUT_LAYER_UNIT = 4
DROPOUT_RATE = 0.1
CONTEXTUAL_UNITS = 32
ATTENTION_LAYER_UNITS = 32
REG_SCALE = 0.001
MAX_CONTEXT_LENGTH = 50
MAX_QUESTION_LENGTH = 30
MAX_ANSWER_LENGTH = 3
EMBEDDING_DIMS = 300

def AttentionModel2(n_answer,
                    embedding,
                    mask_zero=True,
                    trainable=True):
    context_shape = (MAX_CONTEXT_LENGTH,)
    question_shape = (MAX_QUESTION_LENGTH,)

    in_context = Input(shape=context_shape, dtype='int32')
    in_question = Input(shape=question_shape, dtype='int32')

    embedding_layer = PretrainedEmbedding(embeddings=embedding, mask_zero=mask_zero, rate=DROPOUT_RATE)
    context, context_mask = embedding_layer(in_context), embedding_layer.compute_mask(in_context)
    question, question_mask = embedding_layer(in_question), embedding_layer.compute_mask(in_question)

    context_fw = GRU(units=CONTEXTUAL_UNITS,
                       return_sequences=True,
                       return_state=False)(context, mask=context_mask)
    context_bw = GRU(units=CONTEXTUAL_UNITS,
                     return_sequences=True,
                     return_state=False,
                     go_backwards=True)(context, mask=context_mask)
    context_bw = tf.reverse(context_bw, axis=[1])

    question_output = GRU(units=CONTEXTUAL_UNITS,
                          return_sequences=True,
                          return_state=True)(question, mask=question_mask)


    # fw, bw = context_output[0], context_output[1]

    attention_fw = Attention(use_scale=True)([
        context_fw, question_output
    ])
    attention_bw = Attention(use_scale=True)([
        context_bw, question_output
    ])

    attention = tf.concat([attention_fw, attention_bw], axis=-1)

    answer = Dense(units=1, activation='relu')(attention)
    answer = tf.squeeze(answer, axis=-1)
    answer = Dense(units=n_answer, activation='softmax')(answer)

    model = Model([in_context, in_question], answer)
    return model


def AttentionModel3(n_answer,
                    embedding,
                    mask_zero=True,
                    trainable=True):
    context_shape = (MAX_CONTEXT_LENGTH,)
    question_shape = (MAX_QUESTION_LENGTH,)

    in_context = Input(shape=context_shape, dtype='int32')
    in_question = Input(shape=question_shape, dtype='int32')

    embedding_layer = PretrainedEmbedding(embeddings=embedding, mask_zero=mask_zero, rate=DROPOUT_RATE)
    context, context_mask = embedding_layer(in_context), embedding_layer.compute_mask(in_context)
    question, question_mask = embedding_layer(in_question), embedding_layer.compute_mask(in_question)

    context_fw = GRU(units=CONTEXTUAL_UNITS,
                       return_sequences=True,
                       return_state=False)(context, mask=context_mask)
    context_bw = GRU(units=CONTEXTUAL_UNITS,
                     return_sequences=True,
                     return_state=False,
                     go_backwards=True)(context, mask=context_mask)
    context_bw = tf.reverse(context_bw, axis=[1])

    question_output, question_state = GRU(units=CONTEXTUAL_UNITS,
                                          return_sequences=True,
                                          return_state=True)(question, mask=question_mask)


    # fw, bw = context_output[0], context_output[1]

    attention_fw = Attention(use_scale=True)([
        context_fw, question_output
    ])
    attention_bw = Attention(use_scale=True)([
        context_bw, question_output
    ])

    attention = Weighted_Sum()([attention_fw, attention_bw])

    context_weighted = GRU(units=CONTEXTUAL_UNITS,
                           return_sequences=True,
                           return_state=False)(context, mask=context_mask)

    c = tf.expand_dims(tf.reduce_sum(tf.multiply(context_weighted, attention), axis=-2), axis=-1)
    pred = Output_Layer(units=n_answer)([c, tf.expand_dims(question_state, axis=-1)])

    model = Model([in_context, in_question], pred)
    return model

class Weighted_Sum(Layer):

    def __init__(self):
        super(Weighted_Sum, self).__init__()

    def build(self, input_shape):
        input_shape = input_shape[0]
        self.W = self.add_weight(name="w",
                                  shape=(input_shape[-1], input_shape[-1]),
                                 initializer='random_uniform',
                                  trainable=self.trainable,
                                  regularizer=regularizers.l2(0.001))
        self.built = True

    def call(self, inputs, **kwargs):
        return tf.expand_dims(tf.reduce_sum(tf.multiply(tf.matmul(inputs[0], self.W), inputs[1]), axis=-1), axis=-1)

    def get_config(self):
        config = {
            'W': self.W
        }
        base_config = super(Weighted_Sum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Output_Layer(Layer):

    def __init__(self, units):
        super(Output_Layer, self).__init__()
        self.units = units

    def build(self, input_shape):
        input_shape = input_shape[0]
        self.W = self.add_weight(name="w",
                                  shape=(self.units, input_shape[-2]),
                                 initializer='random_uniform',
                                  trainable=self.trainable,
                                  regularizer=regularizers.l2(0.001))

        self.built = True

    def call(self, inputs, **kwargs):
        tot = tf.math.add(inputs[0], inputs[1])
        res = tf.matmul(self.W, tot)
        return tf.nn.softmax(res, axis=-1)