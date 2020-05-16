import pickle
import numpy as np
MIN_FREQ = 3


def invert_dict(d):
    """
    Exchanges keys and values in a dictionary
    :param d: dictionary for invertion
    :return: inverted dictionary
    """
    res = {}
    for k, v in d.items():
        res[v] = k
    return res

def add_to_dictionary(dictionary, key, value):
    """
    Add values to a dictionary when values are lists
    :param dictionary: the dictionary to be used
    :param key: the key to be appended
    :param value: the value to be added
    :return: nothing
    """
    if key not in dictionary:
        dictionary[key] = [value]
    else:
        dictionary[key].append(value)


def increment_count(count_dict, key):
    """
    Puts the key in the dictionary if does not exist or adds one if it does.
    :param count_dict: dictionary for the count performed
    :param key: key to add to dictionary
    :return: dictionary after addition of the key
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1


def compute_vocab_count(sents):
    """
    Takes a corpus and computes the frequency of each word's appearance
    :param sents: sentences for counting frequency
    :return: dictionary with count of words
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab


def reorganize_data(texts):
    """
    Reorganize data to contain tuples of a all signs combined and all trans combined
    :param texts: sentences in format of tuples of (sign, tran)
    :return: data reorganized
    """
    data = []

    for sentence in texts:
        signs = []
        trans = []
        for sign, tran in sentence:
            signs.append(sign)
            trans.append(tran)
        data.append((signs, trans))

    return data


def from_key_to_line_number(key):
    """
    Takes a key and returns the line number in it
    :param key: The key to parse
    :return: line number
    """
    n = key.split(".", 2)[1]

    # Sometimes line number contains a redundant "l" at the end ("Q005624.1l" for example), so we ignore it.
    if n[-1] == "l":
        n = n[:-1]

    if not n.isdigit():
        return -1

    line_number = int(n)

    return line_number


def from_key_to_text_and_line_numbers(key):
    """
    Takes a key and divides it into the text, start line and end line
    :param key: The key to parse
    :return: text, start line and end line
    """
    # Calculation of start_line and end_line for signs and transcriptions
    text = key[0].split(".", 2)[0]
    start_line = from_key_to_line_number(key[0])

    if start_line == -1:
        return text, -1, -1

    # Sometimes the end line is not specified when it's one line ("n057" for example), so we use the start line.
    if "." in key[1]:
        end_line = from_key_to_line_number(key[1])
    else:
        end_line = start_line

    return text, start_line, end_line


def give_idx(key, dict):
    """
    Gives unique value for every new key
    :param key: the key to be added to the dict
    :param dict: the dictionary that will be changed
    :return: nothing
    """
    if key not in dict:
        dict[key] = len(dict)


def rep_to_ix(data):
    """
    Builds 2 dictionaries with unique values for each sign/transliteration
    :param data: the data in a format of signs and transliterations
    :return: the dictionaries - one for signs, one for transliterations
    """
    sign_to_ix = {}
    tran_to_ix = {}

    for signs, trans in data:
        for sign in signs:
            give_idx(sign, sign_to_ix)
        for tran in trans:
            give_idx(tran, tran_to_ix)

    return sign_to_ix, tran_to_ix


def dump_object_to_file(object, object_name):
    """
    Dumps object to a file called object_name (usually learned stuff)
    :param object: learned data which will be saved
    :param object_name: file name to save the object
    :return: nothing
    """
    with open(object_name, "wb") as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)


def load_object_from_file(object_name):
    """
    Loads object from a file called object_name (usually learned stuff)
    :param object_name: file name to load the object from
    :return: the object which was learned and saved
    """
    with open(object_name, "rb") as input:
        object = pickle.load(input)

    return object


def logits_to_trans(tag_logits, model, id_to_tran):
    """
    Builds lists of the predicted tags and their scores according to BiLSTM (3 most reasonable tags)
    :param tag_logits: the tags' probabilities as predicted by BiLSTM
    :param model: the model which was learned by BiLSTM
    :param id_to_tran: dictionary that maps unique id to the corresponding transliteration
    :return: 3 lists of the top 3 predicted tags and lists of their scores
    """
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
    """
    Checks if a sign finishes a word
    :param s: the sign to check
    :return: true if the sign finishes the word
    """
    if s[-1] in "-.":
        return False
    return True


def compute_accuracy(texts, prediction_function, *args):
    """
    Computes the accuracy of the predicted tags by the model
    :param texts: the texts for the accuracy checks
    :param prediction_function: the function which predicts the corresponding tag and we want to check
    :*args: more arguements needed for prediction_function
    :return: returns the computed accuracy with segmentation, without segmentation and the F1 score
    """
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
