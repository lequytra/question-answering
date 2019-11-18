from tensorflow.keras.layers import Input
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

def DMN(max_question_length,
        max_context_length,
        n_answer,
        tokenizer,
        embedding,
        mask_zero=True,
        context_shape=None,
        question_shape=None,
        trainable=True):

    if context_shape is None:
        context_shape = (None,)
    if question_shape is None:
        question_shape = (None,)
    context = Input(shape=context_shape)
    question = Input(shape=question_shape)

    # TODO: input should have trainable argument?
    input_preprocessing = InputModule(units=INPUT_LAYER_UNIT,
                                      tokenizer=tokenizer,
                                      embedding=embedding,
                                      mask_zero=mask_zero,
                                      max_len=max_context_length,
                                      rate=DROPOUT_RATE)

    # TODO: Maybe seperate the input module, we only need to share the tokenizer and embedding layer
    context, context_states = input_preprocessing(context)
    question, question_states = input_preprocessing(question)

    # TODO: this looks problematic. Maybe we should not pass m from the beginning to EPISODIC CELL?
    attention, m = EpisodicModule(units=EPISODIC_LAYER_UNITS,
                                  question=question,
                                  attention_layer_units=ATTENTION_LAYER_UNITS,
                                  m=question,
                                  reg_scale=REG_SCALE,
                                  trainable=trainable,
                                  return_sequences=False)(context_states)

    answer = AnswerModule(units=n_answer, question=question)(m)

    model = Model([context, question], answer)
    return model



