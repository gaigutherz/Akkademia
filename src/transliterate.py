from pathlib import Path
from data import logits_to_trans, load_object_from_file
from combine_algorithms import overall_classifier, list_to_tran, sentence_to_allen_format, \
    sentence_to_HMM_format
from hmm import hmm_viterbi


def signs_to_transliteration(sentence):
    """
    Transliterate signs using biLSTM
    :param sentence: signs to be transliterated
    :return: transliteration of the sentence
    """
    model, predictor, sign_to_id, id_to_tran, test_texts = load_object_from_file(Path(r"../output/biLSTM_model.pkl"))

    tag_logits = predictor.predict(sentence_to_allen_format(sentence, sign_to_id, True))['tag_logits']
    biLSTM_predicted_tags, _, _, _, _, _ = logits_to_trans(tag_logits, model, id_to_tran)
    return list_to_tran(biLSTM_predicted_tags)


def signs_to_transliteration_hmm(sentence):
    """
    Transliterate signs using HMM
    :param sentence: signs to be transliterated
    :return: transliteration of the sentence
    """
    most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2, test_texts = \
        load_object_from_file(Path(r"../output/hmm_model.pkl"))

    HMM_predicted_tags = hmm_viterbi(sentence_to_HMM_format(sentence), total_tokens, q_bi_counts, q_uni_counts, q, e,
                           S, most_common_tag, possible_tags, lambda1, lambda2)
    return HMM_predicted_tags


def main():
    """
    Loads all models' learned data and open an interpreter for transliterating sentences from input
    :return: nothing, never stops
    """
    most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2, test_texts = \
        load_object_from_file(Path(r"../output/hmm_model.pkl"))

    logreg, vec, idx_to_tag_dict, test_texts = \
       load_object_from_file(Path(r"../output/memm_model.pkl"))

    model, predictor, sign_to_id, id_to_tran, test_texts = load_object_from_file(Path(r"../output/biLSTM_model.pkl"))

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
    main()
