import numpy as np

from hmm import hmm_train, hmm_viterbi
from data import load_object_from_file, logits_to_trans
from memm import memm_greedy, build_extra_decoding_arguments


def sentence_to_HMM_format(sentence):
    """
    Transform the sentence to HMM format
    :param sentence: the sentence to transform
    :return: the HMM format
    """
    list = []
    for sign in sentence:
        if sign == " ":
            continue
        list.append((sign, ""))

    return list


def list_to_tran(list):
    """
    Transform the list of predicted tags to a printable way
    :param list: list of tags
    :return: string of transliteration
    """
    transcription = ""
    for tran in list:
        if tran[-3:] == "(0)":
            transcription += tran[:-3]
        elif tran[-4:] == "(0)-" or tran[-4:] == "(0).":
            transcription += tran[:-4] + tran[-1]
        elif tran[-1] == ")" or tran[-2:] == ")-" or tran[-2:] == ").":
            continue
        else:
            transcription += tran

        if tran[-1] != "-" and tran[-1] != ".":
            transcription += " "

    return transcription


def sentence_to_allen_format(sentence, sign_to_id, usingRealSigns):
    """
    Transform the sentence to AllenNLP format
    :param sentence: the sentence to transform
    :param sign_to_id: dictionary of sign to id
    :param usingRealSigns: whether using the signs as is
    :return: the AllenNLP format for BiLSTM
    """
    signs = ""

    if usingRealSigns:
        for sign in sentence:
            if sign == " " or sign == "\t" or sign == "\n":
                continue
            try:
                signs += str(sign_to_id[sign]) + " "
            except:
                signs += "0 " # default index

    else:
        for sign in sentence.split(","):
            signs += sign + " "

    return signs


def overall_choose_best_gammas(lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, predictor_from_file, model_from_file, id_to_tran, sign_to_id, dev_texts):
    """
    Choose the best gammas for combination of BiLSTM, MEMM and HMM (the strength of each algorithm)
    :param lambda1: lambda for HMM use
    :param lambda2: lambda for HMM use
    :param logreg: learned for MEMM use
    :param vec: vectorization function for MEMM use
    :param idx_to_tag_dict: dictionary of indices to tags
    :param extra_decoding_arguments: extra decoding arguments
    :param predictor_from_file: for BiLSTM use
    :param model_from_file: for BiLSTM use
    :param id_to_tran: dictionary of indices to tags
    :param sign_to_id: dictionary of signs to indices
    :param dev_texts: validation texts for hyperparams
    :return: the gammas
    """

    best_gamma1 = -1
    best_gamma2 = -1
    best_accuracy = -1

    for i in range(0, 5):
        for j in range(0, 5):
            gamma1 = i / 10.0
            gamma2 = j / 10.0
            accuracy = overall_compute_accuracy(dev_texts, gamma1, gamma2, lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, predictor_from_file, model_from_file, id_to_tran, sign_to_id)
            print("For gamma1 = " + str(gamma1), ", gamma2 = " + str(gamma2), " got accuracy = " + str(accuracy))
            if accuracy > best_accuracy:
                best_gamma1 = gamma1
                best_gamma2 = gamma2
                best_accuracy = accuracy
    print("The setting that maximizes the accuracy on the test data is gamma1 = " + \
          str(best_gamma1), ", gamma2 = " + str(best_gamma2), " (accuracy = " + str(best_accuracy) + ")")

    return best_gamma1, best_gamma2


def overall_compute_accuracy(test_data, gamma1, gamma2, lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, predictor_from_file, model_from_file, id_to_tran, sign_to_id):
    """
    Evaluate the best gammas for combination of BiLSTM, MEMM and HMM (the strength of each algorithm)
    :param test_data: data for evaluation
    :param gamma1: the strength that was learned
    :param gamma2: the strength that was learned
    :param lambda1: lambda for HMM use
    :param lambda2: lambda for HMM use
    :param logreg: learned for MEMM use
    :param vec: vectorization function for MEMM use
    :param idx_to_tag_dict: dictionary of indices to tags
    :param extra_decoding_arguments: extra decoding arguments
    :param predictor_from_file: for BiLSTM use
    :param model_from_file: for BiLSTM use
    :param id_to_tran: dictionary of indices to tags
    :param sign_to_id: dictionary of signs to indices
    :return: the accuracy for the learned gammas
    """

    correct = 0
    seps = 0
    total = 0

    for sentence in test_data:
        new_sentence = ''
        for i in sentence:
            new_sentence += i[0]

        predicted_tags = overall_classifier(new_sentence, gamma1, gamma2, total_tokens, q_bi_counts, q_uni_counts, q,
                    e, S, lambda1, lambda2, logreg, vec, idx_to_tag_dict, extra_decoding_arguments,
                                            predictor_from_file, model_from_file, id_to_tran, sign_to_id, False)

        for i in range(len(sentence)):
            total += 1

            if sentence[i][1] == predicted_tags[i]:
                correct += 1
            else:
                if len(sentence[i]) > 0 and sentence[i][-1] == "-" or \
                        len(predicted_tags[i]) > 0 and predicted_tags[i][-1] == "-":
                    seps += 1



    return float(correct) / total


def overall_classifier(sentence, gamma1, gamma2, total_tokens, q_bi_counts, q_uni_counts, q, e, S, most_common_tag,
                       possible_tags, lambda1, lambda2, logreg, vec, idx_to_tag_dict, predictor_from_file,
                       model_from_file, id_to_tran, sign_to_id, is_verbose):
    """
    Classify the tags with the best gammas for combination of BiLSTM, MEMM and HMM (the strength of each algorithm)
    :param sentence: sentence to tag
    :param gamma1: the strength that was learned
    :param gamma2: the strength that was learned
    :param lambda1: lambda for HMM use
    :param lambda2: lambda for HMM use
    :param logreg: learned for MEMM use
    :param vec: vectorization function for MEMM use
    :param idx_to_tag_dict: dictionary of indices to tags
    :param predictor_from_file: for BiLSTM use
    :param model_from_file: for BiLSTM use
    :param id_to_tran: dictionary of indices to tags
    :param sign_to_id: dictionary of signs to indices
    :param is_verbose: whether to print
    :return: the classified tags by the combination of the algorithms
    """

    if is_verbose:
        print(sentence)
    HMM_predicted_tags = hmm_viterbi(sentence_to_HMM_format(sentence), total_tokens, q_bi_counts, q_uni_counts, q, e,
                           S, most_common_tag, possible_tags, lambda1, lambda2)

    if is_verbose:
        print("HMM prediction: ")
        print(HMM_predicted_tags)

    MEMM_predicted_tags = memm_greedy(sentence_to_HMM_format(sentence), logreg, vec, idx_to_tag_dict)
    if is_verbose:
        print("MEMM prediction: ")
        print(MEMM_predicted_tags)

    # BiLSTM prediction
    tag_logits = predictor_from_file.predict(sentence_to_allen_format(sentence, sign_to_id, True))['tag_logits']
    biLSTM_predicted_tags, biLSTM_predicted2_tags, biLSTM_predicted3_tags, biLSTM_scores, biLSTM_scores2, biLSTM_scores3 = logits_to_trans(
        tag_logits, model_from_file, id_to_tran)
    if is_verbose:
        print("biLSTM prediction: ")
        print(biLSTM_predicted_tags)
        print("biLSTM scores: ")
        print(biLSTM_scores)
        print("biLSTM second option prediction: ")
        print(biLSTM_predicted2_tags)
        print("biLSTM second option scores: ")
        print(biLSTM_scores2)
        print("biLSTM third option prediction: ")
        print(biLSTM_predicted3_tags)
        print("biLSTM third option scores: ")
        print(biLSTM_scores3)

    HMM_tran = list_to_tran(HMM_predicted_tags)
    if is_verbose:
        print("HMM transcription: ")
        print(HMM_tran)

    MEMM_tran = list_to_tran(MEMM_predicted_tags)
    if is_verbose:
        print("MEMM transcription: ")
        print(MEMM_tran)

    biLSTM_tran = list_to_tran(biLSTM_predicted_tags)
    if is_verbose:
        print("biLSTM transcription: ")
        print(biLSTM_tran)

    algorithms_tags = (biLSTM_predicted_tags, biLSTM_predicted2_tags, biLSTM_predicted3_tags,
    biLSTM_scores, biLSTM_scores2, biLSTM_scores3,
    HMM_predicted_tags, MEMM_predicted_tags)
    overall_tran = list_to_tran(combine_tags(algorithms_tags, gamma1, gamma2))

    if is_verbose:
        print("Overall transcription: ")
        print(overall_tran)


def combine_tags(algorithms_tags, gamma1, gamma2):
    """
    Classify the tags with the best gammas for combination of BiLSTM, MEMM and HMM (the strength of each algorithm)
    :param algorithms_tags: the tags predicted by BiLSTM, MEMM and HMM
    :param gamma1: the strength that was learned
    :param gamma2: the strength that was learned
    :return: the classified tags by the combination of the algorithms
    """

    (biLSTM_predicted_tags, biLSTM_predicted2_tags, biLSTM_predicted3_tags,
    biLSTM_scores, biLSTM_scores2, biLSTM_scores3,
    HMM_predicted_tags, MEMM_predicted_tags) = algorithms_tags
    overall_predicted_tags = []
    tag_options = [biLSTM_predicted_tags, biLSTM_predicted2_tags, biLSTM_predicted3_tags]

    for i in range(len(biLSTM_predicted_tags)):
        if biLSTM_predicted_tags[i] == HMM_predicted_tags[i]:
            biLSTM_scores[i] += gamma1
        if biLSTM_predicted_tags[i] == MEMM_predicted_tags[i]:
            biLSTM_scores[i] += gamma2
        if biLSTM_predicted2_tags[i] == HMM_predicted_tags[i]:
            biLSTM_scores2[i] += gamma1
        if biLSTM_predicted2_tags[i] == MEMM_predicted_tags[i]:
            biLSTM_scores2[i] += gamma2
        if biLSTM_predicted3_tags[i] == HMM_predicted_tags[i]:
            biLSTM_scores3[i] += gamma1
        if biLSTM_predicted3_tags[i] == MEMM_predicted_tags[i]:
            biLSTM_scores3[i] += gamma2

        max_ind = np.argmax([biLSTM_scores[i], biLSTM_scores2[i], biLSTM_scores3[i]])
        overall_predicted_tags.append(tag_options[max_ind][i])

    return overall_predicted_tags
