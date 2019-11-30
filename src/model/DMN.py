import tensorflow as tf
from tensorflow.keras.layers import Input, GRU
from tensorflow.keras.backend import constant
from tensorflow.keras.models import Model
try:
    from model.AnswerModule import *
    from model.AttentionGate import *
    from model.InputLayer import *
except ModuleNotFoundError:
    from src.model.AnswerModule import *
    from src.model.AttentionGate import *
    from src.model.InputLayer import *

INPUT_LAYER_UNIT = 4
DROPOUT_RATE = 0.1
EPISODIC_LAYER_UNITS = 4
ATTENTION_LAYER_UNITS = 3
REG_SCALE = 0.001
MAX_CONTEXT_LENGTH = 30
MAX_QUESTION_LENGTH = 10
MAX_ANSWER_LENGTH = 3
EMBEDDING_DIMS = 300

def DMN(n_answer,
        embedding,
        mask_zero=True,
        trainable=True):

    context_shape = (MAX_CONTEXT_LENGTH,)
    question_shape = (MAX_QUESTION_LENGTH,)
    answer_shape = (MAX_ANSWER_LENGTH,)
    in_context = Input(shape=context_shape, dtype='int32')
    in_question = Input(shape=question_shape, dtype='int32')
    in_answer = Input(shape=answer_shape, dtype='int32')

    embedding_layer = PretrainedEmbedding(embeddings=embedding, mask_zero=mask_zero, rate=DROPOUT_RATE)
    context, context_mask = embedding_layer(in_context), embedding_layer.compute_mask(in_context)
    question, question_mask = embedding_layer(in_question), embedding_layer.compute_mask(in_question)

    context, c_lst = GRU(units=INPUT_LAYER_UNIT,
                                  return_sequences=True,
                                  return_state=True)(context, mask=context_mask)

    question = GRU(units=INPUT_LAYER_UNIT,
                    return_sequences=False,
                    return_state=False)(question, mask=question_mask)
    # question = tf.expand_dims(question, axis=1)

    initial_states = [question, question, question]

    # TODO: this looks problematic. Maybe we should not pass m from the beginning to EPISODIC CELL?
    m1, states = EpisodicModule(units=EPISODIC_LAYER_UNITS,
                      attention_layer_units=ATTENTION_LAYER_UNITS,
                      reg_scale=REG_SCALE,
                      trainable=trainable,
                      return_sequences=False,
                      return_state=True,
                      zero_output_for_mask=True,
                      unroll=True)(context, initial_state=initial_states, mask=context_mask)
    # m1 = tf.expand_dims(m1, axis=1)
    initial_states = [states, m1, question]

    m2, states2 = EpisodicModule(units=EPISODIC_LAYER_UNITS,
                      attention_layer_units=ATTENTION_LAYER_UNITS,
                      reg_scale=REG_SCALE,
                      trainable=trainable,
                      return_sequences=False,
                      return_state=True,
                      zero_output_for_mask=True,
                      unroll=True)(context, initial_state=initial_states, mask=context_mask)
    # m2 = tf.expand_dims(m2, axis=1)
    initial_states = [states2, m2, question]
    m = EpisodicModule(units=EPISODIC_LAYER_UNITS,
                      attention_layer_units=ATTENTION_LAYER_UNITS,
                      reg_scale=REG_SCALE,
                      trainable=trainable,
                      return_sequences=False,
                      return_state=False,
                      zero_output_for_mask=True,
                      unroll=True)(context, initial_state=initial_states, mask=context_mask)

    input = tf.concat([m, question], axis=-1)
    # Decode the predicted answer out
    answer_outputs = LinearRegression(units=n_answer,
                                      reg_scale=REG_SCALE,
                                      trainable=trainable)(input)

    model = Model([in_context, in_question], answer_outputs)
    return model



