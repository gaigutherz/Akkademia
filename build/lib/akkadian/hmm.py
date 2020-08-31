from akkadian.data import increment_count, compute_vocab_count, compute_accuracy
from akkadian.build_data import preprocess
TEST_DATA_SIZE_FOR_LAMBDAS = 3


def hmm_preprocess(train_sents):
    """
    train the HMM model
    :param train_sents: train sentences for the model
    :return: counts of unigrams, bigrams and trigrams
    """

    print("Start training")
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = {}, {}, {}, {}, {}

    # e_tag_counts
    for sentence in train_sents:
        for token in sentence:
            key = token[1]
            increment_count(e_tag_counts, key)

    # e_word_tag_counts
    for sentence in train_sents:
        for token in sentence:
            key = token
            increment_count(e_word_tag_counts, key)

    # New update to enhance performance.
    most_common_tag = {}
    for word, tag in e_word_tag_counts:
        if word not in most_common_tag:
            most_common_tag[word] = (tag, e_word_tag_counts[word, tag])
        elif e_word_tag_counts[word, tag] > most_common_tag[word][1]:
                most_common_tag[word] = (tag, e_word_tag_counts[word, tag])
    most_common_tag["default"] = max(e_tag_counts, key=e_tag_counts.get)

    # Add *, * to beginning of every sentence and STOP to every end.
    adjusted_sents = []
    for sentence in train_sents:
        adjusted_sentence = []
        adjusted_sentence.append(('<s>', '<s>'))
        adjusted_sentence.append(('<s>', '<s>'))
        for token in sentence:
            adjusted_sentence.append(token)
        adjusted_sentence.append(('</s>', '</s>'))
        adjusted_sents.append(adjusted_sentence)

    # total_tokens
    for sentence in adjusted_sents:
        total_tokens += (len(sentence) - 2)

    # q_uni_counts
    for sentence in adjusted_sents:
        for token in sentence:
            key = token[1]
            increment_count(q_uni_counts, key)

    # q_bi_counts
    for sentence in adjusted_sents:
        for i in range(1, len(sentence)):
            key = (sentence[i-1][1], sentence[i][1])
            increment_count(q_bi_counts, key)

    # q_tri_counts
    for sentence in adjusted_sents:
        for i in range(2, len(sentence)):
            key = (sentence[i-2][1], sentence[i-1][1], sentence[i][1])
            increment_count(q_tri_counts, key)

    # possible tags
    possible_tags = {}
    for sentence in train_sents:
        for token in sentence:
            if token[0] in possible_tags:
                possible_tags[token[0]].add(token[1])
            else:
                possible_tags[token[0]] = {token[1]}

    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts, most_common_tag, \
           possible_tags


def hmm_compute_q_e_S(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Computes the frequencies needed for the HMM model
    :param total_tokens: total number of tokens
    :param q_tri_counts: dictionary of the frequency of each trigram in the text
    :param q_bi_counts: dictionary of the frequency of each bigram in the text
    :param q_uni_counts: dictionary of the frequency of each unigram in the text
    :param e_word_tag_counts: dictionary of the frequency of each (word, tag) couple in the text
    :param e_tag_counts: dictionary of the frequency of each tag in the text
    :return: nothing, q, e and S are saved as global parameters
    """
    # q
    q = {}
    for key in q_tri_counts:
        q[key] = (float(q_tri_counts[key])/q_bi_counts[(key[0], key[1])],
                 float(q_bi_counts[(key[1], key[2])])/q_uni_counts[key[1]],
                 float(q_uni_counts[key[2]])/total_tokens)

    # e
    e = {}
    for key in e_word_tag_counts:
        e[key] = float(e_word_tag_counts[key]) / e_tag_counts[key[1]]

    # S
    S = list(q_uni_counts.keys())

    return q, e, S


def hmm_viterbi(sent, total_tokens, q_bi_counts, q_uni_counts, q, e, S, most_common_tag, possible_tags, lambda1, lambda2):
    """
    Predict the transliteration of a sentence of signs
    :param sent: the sentence of signs to predict
    :param total_tokens: total tokens in the text
    :param q_bi_counts: frequency of bigrams in the text
    :param q_uni_counts: frequency of unigrams in the text
    :param lambda1: the weight for the unigrams
    :param lambda2: the weight for the bigrams
    :return: predicted transliteration for the sentence
    """
    predicted_tags = [""] * (len(sent))

    # n: sent length
    n = len(sent)

    # pi + bp
    pi = {}
    bp = {}
    pi[(0, '<s>', '<s>')] = 1
    S0 = []
    for s in S:
        if s != '<s>' and s != '<\s>':
            S0.append(s)

    # Viterbi algorithm
    for k in range(1, n+1):
        try:
            S_u = possible_tags[sent[k-2][0]]
        except:
            S_u = S0
        if k == 1:
            S_u = ['<s>']
        for u in S_u:
            try:
                S_v = possible_tags[sent[k-1][0]]
            except:
                S_v = S0
            for v in S_v:
                try:
                    e_calc = e[(sent[k-1][0], v)]
                except:
                    continue
                pi_tmp = -1
                bp_tmp = "DEFAULT"
                try:
                    S_w = possible_tags[sent[k-3][0]]
                except:
                    S_w = S0
                if k == 1 or k == 2:
                    S_w = ['<s>']
                for w in S_w:
                    try:
                        pi_calc = pi[(k - 1, w, u)]
                    except:
                        continue
                    try:
                        q_calc = lambda1*q[(w, u, v)][0] + lambda2*q[(w, u, v)][1] + (1-lambda1-lambda2)*q[(w, u, v)][2]
                    except:
                        if (u, v) in q_bi_counts:
                            q2 = float(q_bi_counts[(u, v)]) / q_uni_counts[u]
                        else:
                            q2 = 0
                        if v in q_uni_counts:
                            q1 = float(q_uni_counts[v]) / total_tokens
                        else:
                            continue
                        q_calc = lambda2 * q2 + (1 - lambda1 - lambda2) * q1
                    if pi_calc * q_calc * e_calc > pi_tmp:
                        pi_tmp = pi_calc * q_calc * e_calc
                        bp_tmp = w
                if bp_tmp != "DEFAULT":
                    pi[(k, u, v)] = pi_tmp
                    bp[(k, u, v)] = bp_tmp
                else:
                    pi[(k, u, v)] = -1
                    try:
                        bp[(k, u, v)] = most_common_tag[sent[k - 3][0]][0]
                    except:
                        bp[(k, u, v)] = "DEFAULT"

    pi_tmp = -1
    try:
        S_u = possible_tags[sent[n-2][0]]
    except:
        S_u = S0
    if n == 1:
        S_u = ['<s>']
    for u in S_u:
        try:
            S_v = possible_tags[sent[n-1][0]]
        except:
            S_v = S0
        for v in S_v:
            try:
                pi_calc = pi[(n, u, v)]
            except:
                continue
            try:
                q_calc = lambda1*q[(u, v, '</s>')][0] + lambda2*q[(u, v, '</s>')][1] + \
                         (1-lambda1-lambda2)*q[(u, v, '</s>')][2]
            except:
                if (v, '</s>') in q_bi_counts:
                    q2 = float(q_bi_counts[(v, '</s>')]) / q_uni_counts[v]
                else:
                    q2 = 0
                if '</s>' in q_uni_counts:
                    q1 = float(q_uni_counts['</s>']) / total_tokens
                else:
                    continue
                q_calc = lambda2 * q2 + (1 - lambda1 - lambda2) * q1
            if pi_calc * q_calc > pi_tmp:
                pi_tmp = pi_calc * q_calc
                predicted_tags[n-2] = u
                predicted_tags[n-1] = v

    # If we enter here both predicted_tags[n-1] and predicted_tags[n-2] have not been assigned.
    if predicted_tags[n-1] == "":
        try:
            predicted_tags[n - 1] = most_common_tag[sent[n - 1][0]][0]
        except:
            predicted_tags[n - 1] = most_common_tag["default"]
        try:
            predicted_tags[n - 2] = most_common_tag[sent[n - 2][0]][0]
        except:
            predicted_tags[n - 2] = most_common_tag["default"]
    for k in range(n-2, 0, -1):
        try:
            predicted_tags[k - 1] = bp[(k+2, predicted_tags[k], predicted_tags[k+1])]
        except:
            try:
                predicted_tags[k - 1] = most_common_tag[sent[k - 1][0]][0]
            except:
                predicted_tags[k - 1] = "NNP"

    return predicted_tags


def hmm_choose_best_lamdas(dev_data, total_tokens, q_bi_counts, q_uni_counts, q, e, S, most_common_tag, possible_tags):
    """
    Do grid search to find lambda1 and lambda2
    :param dev_data: dev sentences for the model
    :return: best lambda1 and best lambda2
    """
    best_lambda1 = -1
    best_lambda2 = -1
    best_accuracy = -1

    for i in range(0, 11, 1):
        for j in range(0, 10 - i, 1):
            lambda1 = i / 10.0
            lambda2 = j / 10.0
            accuracy, _, _ = compute_accuracy(dev_data, hmm_viterbi, total_tokens, q_bi_counts, q_uni_counts, q, e, S,
                                              most_common_tag, possible_tags, lambda1, lambda2)
            #print("For lambda1 = " + str(lambda1), ", lambda2 = " + str(lambda2), \
            #    ", lambda3 = " + str(1 - lambda1 - lambda2) + " got accuracy = " + str(accuracy))
            if accuracy > best_accuracy:
                best_lambda1 = lambda1
                best_lambda2 = lambda2
                best_accuracy = accuracy

    print("The setting that maximizes the accuracy on the test data is lambda1 = " + \
          str(best_lambda1), ", lambda2 = " + str(best_lambda2), \
        ", lambda3 = " + str(1 - best_lambda1 - best_lambda2) + " (accuracy = " + str(best_accuracy) + ")")

    return best_lambda1, best_lambda2


def hmm_train(train_sents, dev_sents):
    """
    Run the HMM model and compute the parameters learned by hmm
    :param train_sents: train sentences for the model
    :param dev_sents: dev sentences for the model
    :return: trained parameters
    """
    vocab = compute_vocab_count(train_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts, most_common_tag, \
            possible_tags = hmm_preprocess(train_sents)
    q, e, S = hmm_compute_q_e_S(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts)

    lambda1, lambda2 = hmm_choose_best_lamdas(dev_sents, total_tokens, q_bi_counts, q_uni_counts, q, e, S,
                                              most_common_tag, possible_tags)
    return most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2


def main():
    """
    Tests the run of HMM
    :return: nothing
    """
    train_texts, dev_texts, test_texts, sign_to_id, tran_to_id, id_to_sign, id_to_tran = \
        preprocess(['rinap/rinap1', 'rinap/rinap3', 'rinap/rinap4', 'rinap/rinap5'])
    most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2 = \
        hmm_train(train_texts, dev_texts)

    print("Done training, now computing accuracy!")
    print(compute_accuracy(train_texts, hmm_viterbi, total_tokens, q_bi_counts, q_uni_counts, q, e, S, most_common_tag,
                           possible_tags, lambda1, lambda2))
    print(compute_accuracy(dev_texts, hmm_viterbi, total_tokens, q_bi_counts, q_uni_counts, q, e, S, most_common_tag,
                           possible_tags, lambda1, lambda2))
    print(compute_accuracy(test_texts, hmm_viterbi, total_tokens, q_bi_counts, q_uni_counts, q, e, S, most_common_tag,
                           possible_tags, lambda1, lambda2))


if __name__ == "__main__":
    main()
