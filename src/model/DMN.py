import tensorflow as tf
from tensorflow.keras.layers import Input, GRU
from tensorflow.keras.models import Model
try:
    from AnswerModule import *
    from AttentionGate import *
    from InputLayer import *
except ModuleNotFoundError:
    from src.model.AnswerModule import *
    from src.model.AttentionGate import *
    from src.model.InputLayer import *

INPUT_LAYER_UNIT = 4
DROPOUT_RATE = 0.1
EPISODIC_LAYER_UNITS = 4
ATTENTION_LAYER_UNITS = 3
REG_SCALE = 0.001
MAX_CONTEXT_LENGTH = 20
MAX_QUESTION_LENGTH = 15
EMBEDDING_DIMS = 300
def DMN(n_answer,
        embedding,
        mask_zero=True,
        trainable=True):

    context_shape = (MAX_CONTEXT_LENGTH,)
    question_shape = (MAX_QUESTION_LENGTH,)

    context = Input(shape=context_shape, dtype='int32')
    question = Input(shape=question_shape, dtype='int32')

    embedding_layer = PretrainedEmbedding(embeddings=embedding, mask_zero=mask_zero, rate=DROPOUT_RATE)
    context, context_mask = embedding_layer(context), embedding_layer.compute_mask(context)
    question, question_mask = embedding_layer(question), embedding_layer.compute_mask(question)
    print(context_mask.shape)
    context, c_lst = GRU(units=INPUT_LAYER_UNIT,
                                  return_sequences=True,
                                  return_state=True)(context, mask=context_mask)
    print(context)

    question, q_lst = GRU(units=INPUT_LAYER_UNIT,
                                    return_sequences=False,
                                    return_state=True)(question, mask=question_mask)
    print(context)
    # TODO: this looks problematic. Maybe we should not pass m from the beginning to EPISODIC CELL?
    attention, m = EpisodicModule(units=EPISODIC_LAYER_UNITS,
                                  question=question,
                                  attention_layer_units=ATTENTION_LAYER_UNITS,
                                  m=question,
                                  reg_scale=REG_SCALE,
                                  trainable=trainable,
                                  return_sequences=False,
                                  return_state=True,
                                  zero_output_for_mask=True)(context, mask=context_mask)

    answer = AnswerModule(units=n_answer, question=question)(m)

    model = Model([context, question], answer)
    return model



