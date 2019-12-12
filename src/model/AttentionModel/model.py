import tensorflow as tf
from tensorflow.keras.layers import GRU, Attention, Input, Dense, Bidirectional, LSTM
from tensorflow.keras.models import Model

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

def AttentionModel(n_answer,
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

    context_output, h = GRU(units=CONTEXTUAL_UNITS,
                               return_sequences=True,
                               return_state=True)(context, mask=context_mask)

    question_output = GRU(units=CONTEXTUAL_UNITS,
                          return_sequences=True,
                          return_state=False)(question, mask=question_mask)

    context_out = Bidirectional(GRU(units=CONTEXTUAL_UNITS,
                                   return_sequences=True,
                                   return_state=False),
                                merge_mode=None)(context_output,
                                               mask=context_mask,
                                               initial_state=[h, h])

    fw, bw = context_out[0], context_out[1]

    attention_fw = Attention(use_scale=True)([
        question_output, fw
    ])
    attention_bw = Attention(use_scale=True)([
        question_output, bw
    ])

    attention = tf.concat([attention_fw, attention_bw], axis=-1)

    answer = Dense(units=1, activation='relu')(attention)
    answer = tf.squeeze(answer, axis=-1)
    answer = Dense(units=n_answer, activation='softmax')(answer)

    model = Model([in_context, in_question], answer)
    return model

