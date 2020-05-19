from data import logits_to_trans
from combine_algorithms import overall_classifier, overall_choose_best_gammas, list_to_tran
from train import sentence_to_allen_format, load_trained_data


def signs_to_transliteration(sentence):
    most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2, \
    sign_to_id, tran_to_id, id_to_sign, id_to_tran, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, \
    model, predictor = load_trained_data()

    tag_logits = predictor.predict(sentence_to_allen_format(sentence, sign_to_id, True))['tag_logits']
    biLSTM_predicted_tags, _, _, _, _, _ = logits_to_trans(tag_logits, model, id_to_tran)
    return list_to_tran(biLSTM_predicted_tags)


def main():
    most_common_tag, possible_tags, q, e, S, total_tokens, q_bi_counts, q_uni_counts, lambda1, lambda2, \
    sign_to_id, tran_to_id, id_to_sign, id_to_tran, logreg, vec, idx_to_tag_dict, extra_decoding_arguments, \
    model, predictor = load_trained_data()
    gamma1 = 0.4
    gamma2 = 0.2

    """
    Sennacherib = "𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"
    """

    while True:
        sentence = input("write here:")

        if sentence == "":
            continue

        overall_predicted_tags = overall_classifier(sentence, gamma1, gamma2, lambda1, lambda2, logreg, vec,
                        idx_to_tag_dict, extra_decoding_arguments, predictor, model, id_to_tran, sign_to_id, True)

        overall_tran = list_to_tran(overall_predicted_tags)
        print("Overall transcription: \n" + overall_tran)


if __name__ == '__main__':
    main()
