# -*- coding: utf-8 -*-
from __future__ import division, print_function
from tensorflow.keras.preprocessing import sequence as seq
import collections
import re
import nltk
import numpy as np
import os
from functools import reduce

def get_files_for_task(task_nbr, babi_dir):
    filenames = os.listdir(babi_dir)
    task_files = list(filter(lambda x: re.search("qa%d_" % (task_nbr), x), filenames))
    assert(len(task_files) == 2)
    train_file = list(filter(lambda x: re.search("_train.txt", x), task_files))[0]
    test_file = list(filter(lambda x: re.search("_test.txt", x), task_files))[0]
    return train_file, test_file

def get_stories(taskfile, only_support=False):
    """
    :param taskfile: filepath to read
    :param only_support:
    :return: a list of tuples (strings)
     ex of element:
     (['Mary', 'moved', 'to', 'the', 'bathroom',  'Daniel', 'journeyed', 'to', 'the', 'office', '.'],
      ['Where', 'is', 'Daniel', '?'],
        'office')
    """
    data = []
    story_sents = []
    ftask = open(taskfile, "rb")
    for line in ftask:
        line = line.strip()
        nid, line = line.decode().split(" ", 1)
        if int(nid) == 1:
            # new story
            story_sents = []
        if "\t" in line:
            # capture question, answer and support
            q, a, support = line.split("\t")
            q = nltk.word_tokenize(q)
            if only_support:
                # only select supporting sentences
                support_idxs = [int(x)-1 for x in support.split(" ")]
                story_so_far = []
                for support_idx in support_idxs:
                    story_so_far.append(story_sents[support_idx])
            else:
                story_so_far = [x for x in story_sents]
            story = reduce(lambda a, b: a + b, story_so_far)
            data.append((story, q, a))
        else:
            # only capture story
            story_sents.append(nltk.word_tokenize(line))
    ftask.close()
    return data

def build_vocab(daten):
    counter = collections.Counter()
    for data in daten:
        for story, question, answer in data:
            for w in story:
                counter[w] += 1
            for w in question:
                counter[w] += 1
            for w in [answer]:
                counter[w] += 1
    # also we want to reserve 0 for pad character, so we offset the
    # indexes by 1.
    words = [wordcount[0] for wordcount in counter.most_common()]
    word2idx = {w: i+1 for i, w in enumerate(words)}
    return word2idx

def get_maxlens(daten):
    """ Return the max number of words in story and question """
    data_comb = []
    for data in daten:
        data_comb.extend(data)
    story_maxlen = max([len(x) for x, _, _ in data_comb])
    question_maxlen = max([len(x) for _, x, _ in data_comb])
    return story_maxlen, question_maxlen

def vectorize(data, word2idx, story_maxlen, question_maxlen):
    """ Create the story and question vectors and the label """
    Xs, Xq, Y = [], [], []
    for story, question, answer in data:
        xs = [word2idx[word] for word in story]
        xq = [word2idx[word] for word in question]
        y = np.zeros(len(word2idx) + 1)
        y[word2idx[answer]] = 1
        Xs.append(xs)
        Xq.append(xq)
        Y.append(y)
    return (seq.pad_sequences(Xs, maxlen=story_maxlen), 
            seq.pad_sequences(Xq, maxlen=question_maxlen),
            np.array(Y))

def separate_files_cqa (task_num, train, stories, save_dir):
    """
    Saves the story into
    :param task_num: task number
    :param train: whether this is training or testing data
    :param stories: the list of tuple of strings [(c1,q1,a1), ... , (cn, 1n, an)]
    :param save_dir: the path to save the txt file
    :return:
    """
    with open(os.path.join(save_dir, 'Context_{}.txt'.format(train + "_" + str(task_num))), 'w+') as f:
        for context in [story[0] for story in stories]:
            f.write(" ".join(context) + '\n')

    with open(os.path.join(save_dir, 'Question_{}.txt'.format(train + "_" + str(task_num))), 'w+') as f:
        for question in [story[1] for story in stories]:
            f.write(" ".join(question) + '\n')

    with open(os.path.join(save_dir, 'Answer_{}.txt'.format(train + "_" +  str(task_num))), 'w+') as f:
        for answer in [story[2] for story in stories]:
            f.write("".join(answer) + '\n')
    print("Finished saving "+str(task_num))


# saving file
if __name__ == "__main__":
    BABI_DIR = "/home/stellasylee/Documents/CSC395/question-answering/script/data/en"
    SAVE_DIR = "/home/stellasylee/Documents/CSC395/question-answering/script/data/merged/"

    # Saving each of tasks into Context, Question, and Answer
    for task_num in range(1,21):
        train_file, test_file = get_files_for_task(task_num, BABI_DIR)
        train = os.path.join(BABI_DIR, train_file)
        separate_files_cqa(task_num, "Train", get_stories(train), SAVE_DIR)
        test = os.path.join(BABI_DIR, test_file)
        separate_files_cqa(task_num, "Test", get_stories(test), SAVE_DIR)


