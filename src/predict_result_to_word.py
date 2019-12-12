import os
import pickle
DATA_PATH = os.path.join(os.getcwd(), '../data/merged')

with open(os.path.join(DATA_PATH, 'special/tokenizer.p'), 'rb') as f:
    tokenizer = pickle.load(f)
# Build reverse dictionary
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
print(type(reverse_word_map))
first2pairs = {k: reverse_word_map[k] for k in list(reverse_word_map)[:20]}
print(first2pairs)