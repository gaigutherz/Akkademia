import pickle
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


def replace_word(word):
    """
        (numbers, dates, etc...)
    """
    ### YOUR CODE HERE
    ### END YOUR CODE
    return "UNK"


def preprocess_sent(vocab, sents):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
                new_sent.append(token)
            else:
                #print token[0] + " is rare!!!"
                #new_sent.append((replace_word(token[0]), token[1]))
                new_sent.append(token)
                replaced += 1
            total += 1
        res.append(new_sent)
    #print("would have replaced: " + str(float(replaced)/total))
    return res


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
    with open(object_name + ".pkl", "wb") as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)


def load_object_from_file(object_name):
    with open(object_name + ".pkl", "rb") as input:
        object = pickle.load(input)

    return object


def logits_to_trans(tag_logits, model, id_to_tran):
    tag_ids = np.argmax(tag_logits, axis=-1)

    for i in range(len(tag_logits)):
        tag_logits[i][tag_ids[i]] = -1000000
    tag2_ids = np.argmax(tag_logits, axis=-1)

    for i in range(len(tag_logits)):
        tag_logits[i][tag2_ids[i]] = -1000000
    tag3_ids = np.argmax(tag_logits, axis=-1)

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

    return prediction, prediction2, prediction3


def BiLSTM_compute_accuracy(texts, model, predictor, sign_to_id, id_to_tran):
    correct = 0
    total = 0

    for text in texts:
        allen_format = ""
        for sign, tran in text:
            allen_format += str(sign_to_id[sign]) + " "
        allen_format = allen_format[:-1]

        tag_logits = predictor.predict(allen_format)['tag_logits']
        prediction, prediction2, prediction3 = logits_to_trans(tag_logits, model, id_to_tran)
        for i in range(len(prediction)):
            total += 1
            if text[i][1] == prediction[i]:
                correct += 1
            else:
                #print(text[i][0] + " " + text[i][1])
                #print(prediction[i] + " " + prediction2[i] + " " + prediction3[i])
                pass

    return float(correct) / total
