import sys
import os
import pickle
import time
import numpy as np

from tensorflow.keras.models import Model, load_model
from preprocessing.preprocessing import transform

MODEL_PATH = os.path.join(os.getcwd(), '/baseline_model')
DATA_PATH = os.path.join(os.getcwd(), '../data/merged')
MAX_CONTEXT = 50
MAX_QUESTION = 30

def to_task_description (num):
    switcher = {
        1: "Single Supporting Fact",
        2: "Two Supporting Facts",
        3: "Three Supporting Facts",
        4: "Two Argument Relations",
        5: "Three Argument Relations",
        6: "Yes/No Questions",
        7: "Counting",
        8: "Lists/Sets",
        9: "Simple Negotiation",
        10: "Indefinite Knowledge"
    }
    return switcher.get(num, "Invalid task number")


def main():
    # Get tokenizer
    with open(os.path.join(DATA_PATH, 'special/tokenizer.p'), 'rb') as f:
        tokenizer = pickle.load(f)
    # Build reverse dictionary
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    with open(os.path.join(DATA_PATH, 'special/embedding_matrix.npy'), 'rb') as f:
        embeddings = np.load(f, allow_pickle=True)

    print("Please enter task number")
    task_num = int(input())
    if task_num <= 0 or task_num > 20:
        print("Wrong task number")
    else:
        model = load_model('lstm_glove_train.h5')
        print(model.summary())
        running = True
        while running:
            # Get input from user
            print("Please enter context or type 'q' for quit the program: ")
            context = input ()
            if context == 'q':
                running = False
                break
            print("Please enter question based on the context: ")
            question = input ()

            # Transform the context and answer input
            context = transform(context, max_len=MAX_CONTEXT, tokenizer=tokenizer)
            question = transform(question, max_len=MAX_QUESTION, tokenizer=tokenizer)
            answer = model.predict([context, question])

            correct_tag_id = np.argmax(answer) # Turn one hot encoding to index
            # TODO: this only gives index 10 which is bedroom
            # pred_word_ind = model.predict_classes(pad_encoded,verbose=0)[0]
            # This doesnt work somehow saying there is no predict_classes
            word = reverse_word_map.get(correct_tag_id)
            print(word)


if __name__ == "__main__":
    main()