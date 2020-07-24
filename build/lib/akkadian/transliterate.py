from akkadian.data import logits_to_trans, load_object_from_file
from akkadian.combine_algorithms import overall_classifier, list_to_tran, sentence_to_allen_format, sentence_to_HMM_format
from akkadian.hmm import hmm_viterbi
from akkadian.memm import memm_greedy
from akkadian.__init__ import hmm_path, memm_path, bilstm_path


def transliterate(sentence):
    """
    Transliterate signs using best transliteration algorithm so far
    :param sentence: signs to be transliterated
    :return: transliteration of the sentence
    """
    return transliterate_bilstm(sentence)


def transliterate_bilstm(sentence):
    """
    Transliterate signs using biLSTM
    :param sentence: signs to be transliterated
    :return: transliteration of the sentence
    """
    model, predictor, sign_to_id, id_to_tran, test_texts = load_object_from_file(bilstm_path)

    tag_logits = predictor.predict(sentence_to_allen_format(sentence, sign_to_id, True))['tag_logits']
    biLSTM_predicted_tags, _, _, _, _, _ = logits_to_trans(tag_logits, model, id_to_tran)
    return list_to_tran(biLSTM_predicted_tags)


def transliterate_bilstm_top3(sentence):
    """
    Transliterate signs using biLSTM
    :param sentence: signs to be transliterated
    :return: 3 top transliterations of the sentence with their scores
    """
    model, predictor, sign_to_id, id_to_tran, test_texts = load_object_from_file(bilstm_path)

    tag_logits = predictor.predict(sentence_to_allen_format(sentence, sign_to_id, True))['tag_logits']
    prediction1, prediction2, prediction3, score1, score2, score3 = logits_to_trans(tag_logits, model, id_to_tran)
    return list_to_tran(prediction1), list_to_tran(prediction2), list_to_tran(prediction3)


def transliterate_hmm(sentence):
    """
    Transliterate signs using HMM
    :param sentence: signs to be transliterated
    :return: transliteration of the sentence
    """
    most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2, test_texts = \
        load_object_from_file(hmm_path)

    HMM_predicted_tags = hmm_viterbi(sentence_to_HMM_format(sentence), total_tokens, q_bi_counts, q_uni_counts, q, e,
                           S, most_common_tag, possible_tags, lambda1, lambda2)
    return list_to_tran(HMM_predicted_tags)


def transliterate_memm(sentence):
    """
    Transliterate signs using MEMM
    :param sentence: signs to be transliterated
    :return: transliteration of the sentence
    """
    logreg, vec, idx_to_tag_dict, test_texts = load_object_from_file(memm_path)

    MEMM_predicted_tags = memm_greedy(sentence_to_HMM_format(sentence), logreg, vec, idx_to_tag_dict)

    return list_to_tran(MEMM_predicted_tags)


def main():
    """
    Loads all models' learned data and open an interpreter for transliterating sentences of signs from input
    :return: nothing, never stops
    """
    most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2, test_texts = \
        load_object_from_file(hmm_path)

    logreg, vec, idx_to_tag_dict, test_texts = load_object_from_file(memm_path)

    model, predictor, sign_to_id, id_to_tran, test_texts = load_object_from_file(bilstm_path)

    gamma1 = 0.4
    gamma2 = 0.2

    """
    Sennacherib = "𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"
    """

    while True:
        sentence = input("write here:")

        if sentence == "":
            continue

        overall_classifier(sentence, gamma1, gamma2, total_tokens, q_bi_counts, q_uni_counts,
            q, e, S, most_common_tag, possible_tags, lambda1, lambda2, logreg, vec, idx_to_tag_dict, predictor, model,
                                                    id_to_tran, sign_to_id, True)


if __name__ == '__main__':
    # main()
    print(transliterate("𒁕𒄿𒁍 𒂵𒊑𒂊𒋗 𒄨 𒃼𒁺 𒊓𒉿𒉡 𒈾𒆠𒊑 𒃻 𒄯𒊓𒀀𒉌"))
    print(transliterate_bilstm("𒁕𒄿𒁍 𒂵𒊑𒂊𒋗 𒄨 𒃼𒁺 𒊓𒉿𒉡 𒈾𒆠𒊑 𒃻 𒄯𒊓𒀀𒉌"))
    print(transliterate_bilstm_top3("𒁕𒄿𒁍 𒂵𒊑𒂊𒋗 𒄨 𒃼𒁺 𒊓𒉿𒉡 𒈾𒆠𒊑 𒃻 𒄯𒊓𒀀𒉌"))
    print(transliterate_hmm("𒁕𒄿𒁍 𒂵𒊑𒂊𒋗 𒄨 𒃼𒁺 𒊓𒉿𒉡 𒈾𒆠𒊑 𒃻 𒄯𒊓𒀀𒉌"))
    print(transliterate_memm("𒁕𒄿𒁍 𒂵𒊑𒂊𒋗 𒄨 𒃼𒁺 𒊓𒉿𒉡 𒈾𒆠𒊑 𒃻 𒄯𒊓𒀀𒉌"))
