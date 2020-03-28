import pickle
from pathlib import Path
import numpy as np
MIN_FREQ = 3


def invert_dict(d):
    res = {}
    for k, v in d.items():
        res[v] = k
    return res


def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1


def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab


def reorganize_data(texts):
    data = []

    for sentence in texts:
        signs = []
        trans = []
        for sign, tran in sentence:
            signs.append(sign)
            trans.append(tran)
        data.append((signs, trans))

    return data


def give_idx(key, dict):
    if key not in dict:
        dict[key] = len(dict)


def rep_to_ix(data):
    sign_to_ix = {}
    tran_to_ix = {}

    for signs, trans in data:
        for sign in signs:
            give_idx(sign, sign_to_ix)
        for tran in trans:
            give_idx(tran, tran_to_ix)

    return sign_to_ix, tran_to_ix


def dump_object_to_file(object, object_name):
    with open(object_name, "wb") as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)


def load_object_from_file(object_name):
    with open(object_name, "rb") as input:
        object = pickle.load(input)

    return object


def logits_to_trans(tag_logits, model, id_to_tran):
    #print(len(tag_logits[0]))
    tag_ids = np.argmax(tag_logits, axis=-1)
    scores = []

    for i in range(len(tag_logits)):
        scores.append(tag_logits[i][tag_ids[i]])
        tag_logits[i][tag_ids[i]] = -1000000
    tag2_ids = np.argmax(tag_logits, axis=-1)
    scores2 = []

    for i in range(len(tag_logits)):
        scores2.append(tag_logits[i][tag2_ids[i]])
        tag_logits[i][tag2_ids[i]] = -1000000
    tag3_ids = np.argmax(tag_logits, axis=-1)
    scores3 = []

    for i in range(len(tag_logits)):
        scores3.append(tag_logits[i][tag3_ids[i]])

    prediction = []
    for id in [model.vocab.get_token_from_index(i, 'labels') for i in tag_ids]:
        tran = id_to_tran[int(id)]
        prediction.append(tran)

    prediction2 = []
    for id in [model.vocab.get_token_from_index(i, 'labels') for i in tag2_ids]:
        tran = id_to_tran[int(id)]
        prediction2.append(tran)

    prediction3 = []
    for id in [model.vocab.get_token_from_index(i, 'labels') for i in tag3_ids]:
        tran = id_to_tran[int(id)]
        prediction3.append(tran)

    return prediction, prediction2, prediction3, scores, scores2, scores3


def is_word_end(s):
    if s[-1] in "-.":
        return False
    return True


def compute_accuracy(texts, prediction_function, *args):
    correct = 0
    correct_without_segmentation = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for text in texts:
        prediction = prediction_function(text, *args)
        for i in range(len(prediction)):
            total += 1
            if text[i][1] == prediction[i]:
                correct += 1
            elif not is_word_end(text[i][1]) and text[i][1][:-1] == prediction[i]:
                correct_without_segmentation += 1
            elif not is_word_end(prediction[i]) and text[i][1] == prediction[i][:-1]:
                correct_without_segmentation += 1

            if is_word_end(prediction[i]) and is_word_end(text[i][1]):
                true_positives += 1
            if is_word_end(prediction[i]) and not is_word_end(text[i][1]):
                false_positives += 1
            if not is_word_end(prediction[i]) and is_word_end(text[i][1]):
                false_negatives += 1

    precision = float(true_positives) / (true_positives + false_positives)
    recall = float(true_positives) / (true_positives + false_negatives)

    F1 = (2 * precision * recall) / (precision + recall)

    accuracy = float(correct) / total
    accuracy_without_segmentation = float(correct + correct_without_segmentation) / total

    return accuracy, accuracy_without_segmentation, F1
