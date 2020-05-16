from data import increment_count, invert_dict, compute_vocab_count, build_tag_to_idx_dict
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import numpy as np
from build_data import preprocess
from hmm import hmm_choose_best_lamdas, hmm_compute_q_e_S, hmm_viterbi, hmm_train


def build_extra_decoding_arguments(train_sents):
    """
    Builds arguements for HMM, MEMM and BiLSTM (unigram, bigram, trigram, etc)
    :param train_sents: all sentences from training set
    :return: all extra arguments which your decoding procedures requires
    """

    extra_decoding_arguments = {}

    START_WORD, STOP_WORD = '<st>', '</s>'
    START_TAG, STOP_TAG = '*', 'STOP'
    e_word_tag_counts, e_tag_counts = {}, {}

    possible_tags = {}
    for sentence in train_sents:
        for token in sentence:
            if token[0] in possible_tags:
                possible_tags[token[0]].add(token[1])
            else:
                possible_tags[token[0]] = {token[1]}

    extra_decoding_arguments['possible_tags'] = possible_tags

    # New update to enhance performance.
    global most_common_tag
    most_common_tag = {}
    for word, tag in e_word_tag_counts:
        if word not in most_common_tag:
            most_common_tag[word] = (tag, e_word_tag_counts[word, tag])
        elif e_word_tag_counts[word, tag] > most_common_tag[word][1]:
            most_common_tag[word] = (tag, e_word_tag_counts[word, tag])

    adjusted_sents = []
    for sentence in train_sents:
        adjusted_sentence = []
        adjusted_sentence.append((START_WORD, START_TAG))
        adjusted_sentence.append((START_WORD, START_TAG))
        for token in sentence:
            adjusted_sentence.append(token)
        adjusted_sentence.append((STOP_WORD, STOP_TAG))
        adjusted_sents.append(adjusted_sentence)

    q_tri_counts, q_bi_counts, q_uni_counts = {}, {}, {}
    # q_uni_counts
    for sentence in adjusted_sents:
        for token in sentence:
            key = token[1]
            increment_count(q_uni_counts, key)
    S = q_uni_counts.keys()

    # q_bi_counts
    for sentence in adjusted_sents:
        for i in range(1, len(sentence)):
            key = (sentence[i - 1][1], sentence[i][1])
            increment_count(q_bi_counts, key)

    # q_tri_counts
    for sentence in adjusted_sents:
        for i in range(2, len(sentence)):
            key = (sentence[i - 2][1], sentence[i - 1][1], sentence[i][1])
            increment_count(q_tri_counts, key)

    extra_decoding_arguments['S'] = S
    cache_probability = {}
    extra_decoding_arguments['cache'] = cache_probability

    return extra_decoding_arguments


def extract_features_base(curr_sign, next_sign, nextnext_sign, prev_sign, prevprev_sign, prev_trans, prevprev_trans):
    """
    Builds the features according to the sign context
    :param curr_sign: current sign
    :param next_sign: next sign
    :param nextnext_sign: the sign after the next sign
    :param prev_sign: previous sign
    :param prevprev_sign: the sign before the previous sign
    :param prev_trans: previous classified transliteration
    :param prevprev_trans: the classified transliteration before the previous
    :return: The word's features
    """
    features = {}
    features['sign'] = curr_sign

    features['prev_sign'] = prev_sign
    features['prevprev_sign'] = prevprev_sign
    features['next_sign'] = next_sign
    features["nextnext_sign"] = nextnext_sign

    features["lower"] = prev_trans[0].islower()
    features["deter"] = 1 if prev_trans[-1] == '}' else 0
    features["part"] = 1 if prev_trans[-1] == ')' else 0

    features["bigram"] = prev_trans
    features["trigram"] = prevprev_trans + ',' + prev_trans

    return features


def extract_features(sentence, i):
    """
    Builds the features according to the sign context
    :param sentence: the sentence to be extracted
    :param i: the sign index to be extracted
    :return: : The word's features
    """
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<st>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<st>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    nextnext_token = sentence[i + 2] if i < (len(sentence) - 2) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], nextnext_token[0], prev_token[0], prevprev_token[0],
                                 prev_token[1], prevprev_token[1])


def vectorize_features(vec, features):
    """
    This function prepares the feature vector for the sklearn solver, use it for tags prediction.
    :param vec: function of vectorization
    :param features: feature vector
    :return: vectorization of the features
    """
    example = [features]
    return vec.transform(example)


def create_examples(sents, tag_to_idx_dict):
    """
    Organizes the sentences by their features for later use by MEMM
    :param sents: sentences to organize
    :param tag_to_idx_dict: dictionary from tags to indices
    :return: examples of features and their tags(labels)
    """
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in range(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            if sent[i][1] in tag_to_idx_dict:
                labels.append(tag_to_idx_dict[sent[i][1]])
            else:
                labels.append(0)

    return examples, labels


def memm_greedy(sent, logreg, vec, index_to_tag_dict):
    """
    Tags a sentence according to parameters learned by MEMM by greedy algorithm
    :param sent: the sentence to tag
    :param logreg: the predictor that was learned
    :param vec: the vectorization function
    :param index_to_tag_dict: dictionary from index to tag
    :return: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    sent_tagged = [[token[0]] for token in sent]

    for i, word in enumerate(sent):
        predicted_tag = index_to_tag_dict[logreg.predict(vec.transform(extract_features(sent_tagged, i)))[0]]
        predicted_tags[i] = predicted_tag
        sent_tagged[i].append(predicted_tag)

    return predicted_tags


def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
    Tags a sentence according to parameters learned by MEMM by viterbi algorithm
    :param sent: the sentence to tag
    :param logreg: the predictor that was learned
    :param vec: the vectorization function
    :param index_to_tag_dict: dictionary from index to tag
    :param extra_decoding_arguments: extra arguements learned
    :return: predicted tags for the sentence
    """

    predicted_tags = [""] * (len(sent))
    cache_probability = extra_decoding_arguments['cache']

    START_WORD, STOP_WORD = '<st>', '</s>'
    START_TAG, STOP_TAG = '*', 'STOP'

    possible_tags = extra_decoding_arguments['possible_tags']
    S = extra_decoding_arguments['S']

    def viterbi_probability(sentence, index, prev_tag, prevprev_tag):
        curr_word = sentence[index][0]
        prev_word = sentence[index - 1][0] if index > 0 else START_WORD
        prevprev_word = sentence[index - 2][0] if index > 1 else START_WORD
        next_word = sentence[index + 1][0] if index < (len(sentence) - 1) else STOP_WORD
        nextnext_word = sentence[index + 2][0] if index < (len(sentence) - 2) else STOP_WORD
        params = (curr_word, next_word, nextnext_word, prev_word, prevprev_word, prev_tag, prevprev_tag)

        if params in cache_probability:
            return cache_probability[params]
        else:
            features_vector = vec.transform(extract_features_base(*params))
            probability = logreg.predict_proba(features_vector)[0]
            cache_probability[params] = probability
            return probability

    # n: sent length
    n = len(sent)

    # pi + bp
    pi = {}
    bp = {}
    pi[(0, START_TAG, START_TAG)] = 1

    # Viterbi algorithm
    for k in range(1, n + 1):
        try:
            S_u = possible_tags[sent[k - 2][0]]
        except:
            S_u = S
        if k == 1:
            S_u = [START_TAG]
        for u in S_u:
            try:
                S_v = possible_tags[sent[k - 1][0]]
            except:
                S_v = S
            for v in S_v:
                pi_tmp = -np.inf
                bp_tmp = ''
                try:
                    S_w = possible_tags[sent[k - 3][0]]
                except:
                    S_w = S
                if k == 1 or k == 2:
                    S_w = [START_TAG]
                for w in S_w:
                    try:
                        curr_prob = viterbi_probability(sent, k - 1, u, w)[tag_to_idx_dict[v]]
                        pi_val = pi[(k - 1, w, u)] + curr_prob
                    except:
                        continue
                    if pi_val > pi_tmp:
                        pi_tmp = pi_val
                        bp_tmp = w
                pi[(k, u, v)] = pi_tmp
                bp[(k, u, v)] = bp_tmp

    pi_tmp = -np.inf
    try:
        S_u = possible_tags[sent[n - 2][0]]
    except:
        S_u = S
    if n == 1:
        S_u = [START_TAG]
    for u in S_u:
        try:
            S_v = possible_tags[sent[n - 1][0]]
        except:
            S_v = S
        for v in S_v:
            try:
                pi_calc = pi[(n, u, v)]
            except:
                continue
            if pi_calc > pi_tmp:
                pi_tmp = pi_calc
                predicted_tags[n - 2] = u
                predicted_tags[n - 1] = v

    # If we enter here both predicted_tags[n-1] and predicted_tags[n-2] have not been assigned.
    if predicted_tags[n - 1] == "":
        predicted_tags[n - 1] = most_common_tag[sent[n - 1][0]][0]
        predicted_tags[n - 2] = most_common_tag[sent[n - 2][0]][0]

    for k in range(n - 2, 0, -1):
        try:
            predicted_tags[k - 1] = bp[(k + 2, predicted_tags[k], predicted_tags[k + 1])]
        except:
            predicted_tags[k - 1] = 'NNP'  # doesn't really matter, very rare.

    extra_decoding_arguments['cache'] = cache_probability

    return predicted_tags


def should_log(sentence_index):
    """
    Checks if should log
    :param sentence_index: index of the current sentence
    :return: true if should log
    """
    if sentence_index > 0 and sentence_index % 10 == 0:
        if sentence_index < 150 or sentence_index % 200 == 0:
            return True

    return False

def memm_hmm_eval(test_data, logreg, vec, index_to_tag_dict, extra_decoding_arguments, total_tokens, q_tri_counts,
                  q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Returns an evaluation of the accuracy of Viterbi & greedy memm and hmm
    :param test_data: the sentences to evaluate
    :param logreg: the predictor that was learned
    :param vec: the vectorization function
    :param index_to_tag_dict: dictionary from index to tag
    :param extra_decoding_arguments: extra arguements learned
    :param total_tokens: number of total tokens
    :param q_tri_counts: trigram counts
    :param q_bi_counts: bigram counts
    :param q_uni_counts: unigram counts
    :param e_word_tag_counts: word counts
    :param e_tag_counts: tag counts
    :return: the accuracies of hmm, memm greedy and memm viterbi
    """
    acc_viterbi, acc_greedy, acc_hmm = 0.0, 0.0, 0.0
    eval_start_timer = time.time()

    correct_hmm_preds = 0
    correct_greedy_preds = 0
    correct_viterbi_preds = 0
    total_words_count = 0


    hmm_compute_q_e_S(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts)
    lambda1, lambda2 = hmm_choose_best_lamdas(test_data[:25])

    for i, sen in enumerate(test_data):
        predicted_hmm = hmm_viterbi(sen, 0, {}, {}, lambda1, lambda2)
        predicted_greedy = memm_greedy(sen, logreg, vec, index_to_tag_dict)
        predicted_viterbi = memm_viterbi(sen, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        for j in range(len(sen)):
            if sen[j][1] == predicted_hmm[j]:
                correct_hmm_preds += 1
            else:
                print("hmm mistake: " + str(sen[j][0]) + " tagged as: " + str(predicted_hmm[j]) + " instead of: " + str(
                    sen[j][1]))
            if sen[j][1] == predicted_greedy[j]:
                correct_greedy_preds += 1
            else:
                print("greedy mistake: " + str(sen[j][0]) + " tagged as: " + str(
                    predicted_greedy[j]) + " instead of: " + str(sen[j][1]))
            if sen[j][1] == predicted_viterbi[j]:
                correct_viterbi_preds += 1
            else:
                print("viterbi mistake: " + str(sen[j][0]) + " tagged as: " + str(
                    predicted_viterbi[j]) + " instead of: " + str(sen[j][1]))

        total_words_count += len(sen)
        acc_hmm = float(correct_hmm_preds) / float(total_words_count)
        acc_greedy = float(correct_greedy_preds) / float(total_words_count)
        acc_viterbi = float(correct_viterbi_preds) / float(total_words_count)

        if should_log(i):
            if acc_greedy == 0 and acc_viterbi == 0:
                raise NotImplementedError
            eval_end_timer = time.time()
            print(str.format("Sentence index: {} hmm_acc: {} greedy_acc: {}    Viterbi_acc:{} , elapsed: {} ", str(i),
                             str(acc_hmm), str(acc_greedy), str(acc_viterbi), str(eval_end_timer - eval_start_timer)))
            eval_start_timer = time.time()
    acc_hmm = float(correct_hmm_preds) / float(total_words_count)
    acc_greedy = float(correct_greedy_preds) / float(total_words_count)
    acc_viterbi = float(correct_viterbi_preds) / float(total_words_count)

    return str(acc_viterbi), str(acc_greedy), str(acc_hmm)

def run_memm(train_sents, dev_sents):
    """
    Run the MEMM model and compute the logistic regression for the predictor
    :param train_sents: train sentences for the model
    :param dev_sents: dev sentences for the model
    :return: the parameters learned by MEMM
    """
    vocab = compute_vocab_count(train_sents)

    extra_decoding_arguments = build_extra_decoding_arguments(train_sents)

    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print("Create train examples")
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)

    num_train_examples = len(train_examples)
    print("#example: " + str(num_train_examples))
    print("Done")

    print("Create dev examples")
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print("#example: " + str(num_dev_examples))
    print("Done")

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print("Vectorize examples")
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print("Done")

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=1, solver='lbfgs', C=100000, verbose=1, n_jobs=2)
    print("Fitting...")
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print("End training, elapsed " + str(end - start) + " seconds")
    return logreg, vec, index_to_tag_dict

if __name__ == "__main__":
    """
    Test the run of MEMM and HMM
    :return: nothing
    """
    full_flow_start = time.time()

    train_sents, dev_sents, _, _, _, _ = preprocess()
    print(len(train_sents))
    print(len(dev_sents))

    vocab = compute_vocab_count(train_sents)

    extra_decoding_arguments = build_extra_decoding_arguments(train_sents)
    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    print("HMM trained")

    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print("Create train examples")
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)

    num_train_examples = len(train_examples)
    print("#example: " + str(num_train_examples))
    print("Done")

    print("Create dev examples")
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print("#example: " + str(num_dev_examples))
    print("Done")

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print("Vectorize examples")
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print("Done")

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=1, solver='lbfgs', C=100000, verbose=1, n_jobs=2)
    print("Fitting...")
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print("End training, elapsed " + str(end - start) + " seconds")
    # End of log linear model training


    start = time.time()
    print("Start evaluation on dev set")

    acc_viterbi, acc_greedy, acc_hmm = memm_hmm_eval(dev_sents, logreg, vec, index_to_tag_dict,
                                                     extra_decoding_arguments, total_tokens, q_tri_counts, q_bi_counts,
                                                     q_uni_counts, e_word_tag_counts, e_tag_counts)
    end = time.time()
    print("Dev: Accuracy hmm : " + acc_hmm)
    print("Dev: Accuracy greedy memm : " + acc_greedy)
    print("Dev: Accuracy Viterbi memm : " + acc_viterbi)

    print("Evaluation on dev set elapsed: " + str(end - start) + " seconds")