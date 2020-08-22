from akkadian.data import logits_to_trans, load_object_from_file
from akkadian.combine_algorithms import overall_classifier, list_to_tran, sentence_to_allen_format, sentence_to_HMM_format
from akkadian.hmm import hmm_viterbi
from akkadian.memm import memm_greedy
from akkadian.__init__ import hmm_path, memm_path, bilstm_path


def sanitize(sentence):
    out_sentence = ""
    for sign in sentence:
        if 0x12000 <= ord(sign) or ord(sign) == 'x':
            out_sentence += sign

    return out_sentence


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
    sentence = sanitize(sentence)

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
    sentence = sanitize(sentence)

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
    sentence = sanitize(sentence)

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
    sentence = sanitize(sentence)

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
    example = """
1	
𒁹𒀭𒋾𒀪𒆪𒊻 𒈗 𒃲𒌑

 	
𒈗 𒆗𒉡 𒈗 𒊹 𒈗 𒂊𒆠 𒈗 𒆳𒆳

 	
𒍝𒉌𒅔 𒂍𒊕𒅍 𒅇 𒂍𒍣𒁕

 	
𒌉𒍑 𒊕𒆗 𒊭 𒁹𒋛𒇻𒊌𒆪 𒈗

5	
𒇽𒈠𒀝𒅗𒁺𒈾𒀀𒀀 𒈗 𒂊𒆠

 	
𒀀𒈾𒆪 𒄿𒉡𒈠 𒀀𒈾 𒂊𒁉𒅖

 	
𒂍𒊕𒅍 𒅇 𒂍𒍣𒁕

 	
𒊮𒁉 𒌒𒇴𒈠 𒋞𒄭𒀀

 	
𒂍𒊕𒅍 𒅇 𒂍𒍣𒁕

10
𒀸 𒆳𒄩𒀜𒁴 𒀸 𒋗𒈫𒐊 𒂖𒇷𒋾

 	
𒄿𒈾 𒉌𒄑 𒊒𒍑𒋾 𒀠𒁉𒅔𒈠

 	
𒀀𒈾 𒈾𒁲𒂊 𒍑𒋗 𒊭 𒂍𒊕𒅍

 	
𒅇 𒂍𒍣𒁕 𒌒𒁉𒅋 𒀸 𒌗𒊺 𒌓𒌋𒌋𒄰

 	
𒈬𒐏𒐈𒄰 𒍑𒋗 𒊭 𒂍𒍣𒁕

15	
𒂍 𒆠𒄿𒉌 𒂍 𒀭𒀝 𒃻 𒆠𒆗 𒁇𒍦𒆠

 	
𒀜𒁲𒂊 𒍑𒅆𒋗 𒀭𒀝 𒌉𒍑 𒍢𒄿𒊑

 	
𒅆𒅅𒆷 𒀭𒈨𒌍 𒈲𒋻𒄷

 	
𒊭 𒀀𒈾 𒋫𒈾𒁕𒀀𒋾

 	
𒋃𒆪𒉡 𒌉𒍑 𒊕𒌅𒌑

20	
𒊭 𒀭𒀫𒌓 𒄿𒀖𒋾 𒀭𒀀𒂔𒌑𒀀

 	
𒊬𒋥 𒉺𒋾𒋗𒈫 𒀮𒉌𒋾

 	
𒄩𒁹 𒀮𒇷𒄑𒈠

 	
𒄿𒈾 𒆠𒁉𒋾𒅗 𒍢𒅕𒋾

 	
𒊭 𒆷 𒅔𒊩𒌆𒉡𒌑 𒆠𒂍𒋢

25	
𒋗𒌝𒄣𒌓 𒈠𒀀𒋾 𒀀𒀀𒁉𒐊

 	
𒅗𒃻𒁺 𒅕𒉌𒀉𒋾𒐊

 	
𒌋𒅗 𒈾𒆠𒊑 𒌑𒋗𒊻𒍪 𒄿𒈾 𒇷𒄿𒋾

 	
𒈗𒌑𒌅 𒈪𒃻𒊑 𒉺𒇷𒂊

 	
𒁍𒀀𒊑 𒈬𒀭𒈾𒈨𒌍 𒂅𒌒 𒊮𒁉

30	
𒊺𒁉𒂊 𒀖𒌅𒌅 𒇻 𒅆𒊑𒅅𒋾

 	
𒈗𒌑𒋾 𒃻 𒁹𒀭𒋾𒀪𒆪𒊻

 	
𒅇 𒋛𒇻𒊌𒆪 𒈗 𒌉𒋙

 	
𒀀𒈾 𒁕𒊏𒀀𒋾 𒌉 𒊒𒁉𒂊

 	
𒀭𒀝 𒌉𒍑 𒂍𒊕𒅍

35	
𒁍𒉽 𒀭𒍂𒊑 𒊕𒌅𒌑

 	
𒄿𒀖𒋾 𒀭𒀀𒂔𒌑𒀀 𒊬𒋥

 	
𒀀𒈾 𒂍𒍣𒁕 𒂍 𒆠𒄿𒉌

 	
𒂍 𒀭𒀀𒉡𒋾𒅗 𒋗𒁁 𒂅𒌒 𒊮𒁉𒅗

 	
𒄿𒈾 𒄭𒁕𒀀𒌓 𒅇 𒊑𒃻𒀀𒌓

40	
𒄿𒈾 𒂊𒊑𒁉𒅗 𒄿𒈾 𒆠𒁉𒋾𒅗

 	
𒆤𒋾 𒊭 𒆷 𒍑𒌓𒊓𒆪 𒇷𒊑𒆪 𒌓𒈪𒐊

 	
𒇷𒈪𒁕 𒈬𒀭𒈾𒋾𒐊

 	
𒇷𒆲 𒄑𒄖𒍝𒌑𒀀 𒇷𒅋𒁉𒅕

 	
𒉺𒇻𒌑𒀀 𒄿𒈾 𒄑𒁕𒅗 𒍢𒄿𒊑

45	
𒈬𒆥 𒁄𒇻𒊌𒆪 𒀭𒂊 𒌋 𒆠𒁴

 	
𒄿𒈾 𒉿𒄿𒅗 𒂖𒇷 𒇺𒋳𒅗𒉡

 	
𒁺𒌦𒆠𒐊 𒆳𒆳𒈨𒌍 𒋫 𒍢𒀉 𒀭𒌓𒅆

 	
𒀀𒁲 𒂊𒊑𒅁 𒀭𒌓𒅆 𒌨𒋗𒁺

 	
𒋗𒈫𒀀𒀀 𒌋𒌋𒁕𒀜𒋾𒅆𒉡 𒇻𒊻𒉌𒅅𒈠

50	
𒀀𒈾 𒉻𒇻𒇻 𒂍𒊕𒅍

 	
𒅇 𒂍𒍣𒁕 𒇻𒁉𒅋 𒀭𒀝

 	
𒌉𒍑 𒊕𒆗 𒀀𒈾 𒂍𒍣𒁕

 	
𒂍 𒆠𒄿𒉌 𒄿𒈾 𒂊𒊑𒁉𒅗

 	
𒅆𒂟𒁴 𒁹𒀭𒋾𒀪𒆪𒊻 𒈗 𒆳𒆳

55	
𒁹𒋛𒇻𒊌𒆪 𒈗 𒌉𒋙

 	
𒊩𒊍𒋫𒅈𒋫𒉌𒅅𒆪

 	
𒄭𒋥𒋢 𒊬𒊏𒀜

 	
𒁕𒈪𒅅𒋾𒋙𒉡

 	
𒇷𒅖𒃻𒆥 𒄿𒈾 𒉿𒄿𒅗
"""
    print(transliterate(example))
    print(transliterate_bilstm(example))
    print(transliterate_bilstm_top3(example))
    print(transliterate_hmm(example))
    print(transliterate_memm(example))
    #main()
