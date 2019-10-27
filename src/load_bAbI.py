import re
import os



def get_raw(data_folder, task_id):
    """
        Return the correct name of the task given task id
    :param task_num:
    :return: task name
    """
    mapping = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",
        "MCTest": "MCTest",
        "19changed": "19changed",
        "joint": "all_shuffled"
    }

    task_name = mapping[task_id]
    train_path = os.path.join(data_folder, "{}_train.txt".format(task_name))
    test_path = os.path.join(data_folder, "{}_test.txt".format(task_name))

    train_data = read_bAbI(train_path)
    test_data = read_bAbI(test_path)
    return train_data, test_data


def read_bAbI(file_path):
    """
        Read bAbI txt files and return a list of tasks.
        Each task is a dictionary of context, question and answer. Context contains
        partial contents that helps answer the question, not the whole story.
        Contexts and questions are tokenized into a list of words and punctuations.
    :param file_path:
    :return: tasks
    """
    print("Loading text ...")

    tasks = []
    curr = None

    for i, line in enumerate(open(file_path)):
        idx = int(line[0:line.find(' ')])

        # If start of a new story
        if idx == 1:
            curr = {"C": [], "Q": None, "A": None}

        line = line.strip()
        # Remove the initial story index
        line = line[line.find(' ') + 1:]
        # If the current line is not a QA pair
        if '\t' not in line:
            curr["C"] += tokenize(line)
        else:
            q_end = line.find('?')
            curr["Q"] = tokenize(line[:q_end])

            ls = line[q_end + 1:].strip().split('\t')
            curr["A"] = ls[0]
            # Add the current story, question and answer pair to task
            tasks.append(curr.copy())

    return tasks


def tokenize(sentence):
    """
    Split a sentence into tokens including punctuation.
    Args:
        sentence (string) : String of sentence to tokenize.
    Returns:
        list : List of tokens.
    """
    return [x.strip() for x in re.split('(\W+)?', sentence) if x.strip()]
