import os
import random
from parse_json import parse_json
from data import reorganize_data, rep_to_ix, invert_dict
import platform
from pathlib import Path

def build_signs_and_transcriptions(corpora):
    base_directory = Path(r"../raw_data/")
    chars = {}
    translation = {}

    for corpus in corpora:
        directory = base_directory / corpus
        for r, d, f in os.walk(directory):
            for file in f:
                key = str(file[:-len(".json")])
                if key not in chars.keys():
                    c, t = parse_json(os.path.join(r, file))
                    if c is not None and t is not None:
                        chars[key], translation[key] = parse_json(os.path.join(r, file))

    return chars, translation


def break_into_sentences(chars):
    sentences = {}
    for key in chars:
        for c in chars[key]:
            line = ".".join(c[0].split(".", 2)[:2])
            if line not in sentences:
                sentences[line] = [c]
            else:
                sentences[line].append(c)
    return sentences


def write_data_to_file(chars):
    output_file = open("signs_and_transcriptions.txt", "w", encoding="utf8")
    #output_file = open("signs_and_transcriptions.txt", "w")

    sum = 0
    for key in chars:
        output_file.write(key + "\n")
        tran_number = len(chars[key])
        sum += tran_number
        output_file.write("number of transcriptions: " + str(tran_number) + "\n")
        for c in chars[key]:
            if c[2] is None:
                #output_file.write(c[1].encode('utf-8') + " ")
                output_file.write(c[1])
            else:
                #output_file.write(c[1].encode('utf-8'))
                output_file.write(c[1])
                #output_file.write(c[2])
                output_file.write(c[2])
        output_file.write("\n")
        for c in chars[key]:
            #output_file.write(c[3].encode('utf-8') + " ")
            output_file.write(c[3])
        output_file.write("\n\n")

    output_file.write("total number of transcriptions: " + str(sum) + "\n")
    output_file.close()


def build_dictionary(chars):
    d = {}

    for key in chars:
        for c in chars[key]:
            if c[3] not in d:
                d[c[3]] = set(c[1])
            else:
                d[c[3]].add(c[1])

    return d


def write_dictionary_to_file(d):
    if platform.system() == "Windows":
        output_file = open(r"..\output\dictionary.txt", "w", encoding="utf8")
    else:
        output_file = open(r"../output/dictionary.txt", "w", encoding="utf8")

    for sign in d:
        output_file.write(sign + "\n")
        output_file.write("size of sign: " + str(len(sign)) + "\n")
        output_file.write("number of transcriptions: " + str(len(d[sign])) + "\n")
        for tran in d[sign]:
            output_file.write(tran + " ")
        output_file.write("\n\n")

    output_file.close()


def build_data_for_hmm(sentences, isTrans):
    texts = []
    for key in sentences:
        text = []
        for c in sentences[key]:
            if len(c[3]) == 1:
                if isTrans:
                    # text.append((c[3], c[1]))
                    text.append((c[3], c[1] + c[2] if not c[2] is None else c[1]))
                else:
                    text.append((c[3], "-" if not c[2] is None else "END"))
                #text.append((c[3], c[1] + "-" if not c[2] is None else c[1]))
                continue
            for i in range(len(c[3])):
                if isTrans:
                    rep = c[1] + "(" + str(i) + ")"
                    text.append((c[3][i], rep + c[2] if not c[2] is None else rep))
                    # text.append((c[3][i], rep))
                else:
                    text.append((c[3][i], "-" if not c[2] is None else "END"))
                #if i < len(c[3]) - 1:
                #    text.append((c[3][i], rep + "-"))
                #else:
                #    text.append((c[3][i], rep + "-" if not c[2] is None else rep))
        if len(text) != 0:
            texts.append(text)

    return texts


def build_id_dicts(texts):
    sign_to_id, tran_to_id = rep_to_ix(reorganize_data(texts))
    #print(sign_to_id)
    #print(tran_to_id)
    id_to_sign = invert_dict(sign_to_id)
    id_to_tran = invert_dict(tran_to_id)
    return sign_to_id, tran_to_id, id_to_sign, id_to_tran


def write_data_for_allen_to_file(texts, file, sign_to_id, tran_to_id):
    #output_file = open(file, "w", encoding="utf8")
    output_file = open(file, "w")

    for text in texts:
        for obj in text:
            #output_file.write(str(obj[0].encode("utf-8")) + "###" + str(obj[1].encode("utf-8")) + " ")
            output_file.write(str(sign_to_id[obj[0]]) + "###" + str(tran_to_id[obj[1]]) + " ")
        output_file.write("\n")

    output_file.close()


def preprocess():
    chars, translation = build_signs_and_transcriptions(["rinap"])
    sentences = break_into_sentences(chars)
    #write_data_to_file(chars)
    d = build_dictionary(chars)
    write_dictionary_to_file(d)
    texts = build_data_for_hmm(sentences, True)
    sign_to_id, tran_to_id, id_to_sign, id_to_tran = build_id_dicts(texts)

    random.shuffle(texts)
    TEN_PERCENT = len(texts) // 10
    test_texts = texts[:TEN_PERCENT]
    dev_texts = texts[TEN_PERCENT : 2*TEN_PERCENT]
    train_texts = texts[2*TEN_PERCENT:]

    write_data_for_allen_to_file(dev_texts, Path(r"../BiLSTM_input/allen_dev_texts.txt"), sign_to_id, tran_to_id)
    write_data_for_allen_to_file(train_texts, Path(r"../BiLSTM_input/allen_train_texts.txt"), sign_to_id, tran_to_id)

    return train_texts, dev_texts, test_texts, sign_to_id, tran_to_id, id_to_sign, id_to_tran


def main():
    preprocess()


if __name__ == '__main__':
    main()
