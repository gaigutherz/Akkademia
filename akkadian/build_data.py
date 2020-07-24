import os
import random
import platform
from pathlib import Path
from akkadian.parse_json import parse_json
from akkadian.data import reorganize_data, rep_to_ix, invert_dict, add_to_dictionary
from akkadian.__init__ import dictionary_path


def build_signs_and_transcriptions(corpora, add_three_dots=False):
    """
    Builds the mappings of signs, transliterations and translations from the jsons of the corpus
    :param corpora: the corpora which we want to learn
    :param add_three_dots: whether we should add three dots out of the .json file or not
    :return: all values we are interested in from the .json file
    """
    base_directory = Path(r"../raw_data/")
    chars = {}
    translation = {}
    mapping = {}
    lines_cut_by_translation = []

    for corpus in corpora:
        directory = base_directory / corpus
        for r, d, f in os.walk(directory):
            for file in f:
                key = str(file[:-len(".json")])
                if key not in chars.keys():
                    c, t, m, l = parse_json(os.path.join(r, file), add_three_dots)
                    if c is not None and t is not None and m is not None and l is not None:
                        chars[key] = c
                        translation[key] = t
                        mapping[(corpus, key)] = m
                        for line in l:
                            lines_cut_by_translation.append(line)

    return chars, translation, mapping, lines_cut_by_translation

def break_into_sentences(chars, lines_cut_by_translation):
    """
    Breaks the data read from files into sentences for learning
    :param chars: all chars read from files to learn
    :param lines_cut_by_translation: lines that are partially translated
    :return: sentences from the file
    """

    sentences = {}
    for key in chars:
        for c in chars[key]:
            values = c[0].split(".")
            line = values[0] + "." + values[1]
            word = int(values[2])

            if lines_cut_by_translation is not None:
                detected_line = False

                for cut_line, threshold in lines_cut_by_translation:
                    if line == cut_line:
                        detected_line = True
                        if word < threshold:
                            add_to_dictionary(sentences, line + "(part 1)", c)
                        else:
                            add_to_dictionary(sentences, line + "(part 2)", c)

                if not detected_line:
                    add_to_dictionary(sentences, line, c)

            else:
                add_to_dictionary(sentences, line, c)

    return sentences


def write_data_to_file(chars):
    """
    Saves data read from file in a new file (signs_and_transcriptions.txt) after organization
    :param chars: all chars read from files to learn
    :return: nothing
    """
    output_file = open("signs_and_transcriptions.txt", "w", encoding="utf8")

    sum = 0
    for key in chars:
        output_file.write(key + "\n")
        tran_number = len(chars[key])
        sum += tran_number
        output_file.write("number of transcriptions: " + str(tran_number) + "\n")
        for c in chars[key]:
            if c[2] is None:
                output_file.write(c[1])
            else:
                output_file.write(c[1])
                output_file.write(c[2])
        output_file.write("\n")
        for c in chars[key]:
            output_file.write(c[3])
        output_file.write("\n\n")

    output_file.write("total number of transcriptions: " + str(sum) + "\n")
    output_file.close()


def build_dictionary(chars):
    """
    Organizes data read from files in a dictionary
    :param chars: all chars read from files to learn
    :return: the dictionary
    """

    d = {}

    for key in chars:
        for c in chars[key]:
            if c[3] not in d:
                d[c[3]] = set(c[1])
            else:
                d[c[3]].add(c[1])

    return d


def write_dictionary_to_file(d):
    """
    Saves data the dictionary in a new file (output\dictionary.txt) after organization
    :param d: the dictionary to save
    :return: nothing
    """
    if platform.system() == "Windows":
        output_file = open(dictionary_path, "w", encoding="utf8")
    else:
        output_file = open(dictionary_path, "w", encoding="utf8")

    for sign in d:
        output_file.write(sign + "\n")
        output_file.write("size of sign: " + str(len(sign)) + "\n")
        output_file.write("number of transcriptions: " + str(len(d[sign])) + "\n")
        for tran in d[sign]:
            output_file.write(tran + " ")
        output_file.write("\n\n")

    output_file.close()


def build_data_for_hmm(sentences):
    """
    Builds the data needed for HMM from the sentences
    :param sentences: the sentences read from files
    :return: data organized for HMM learn
    """

    texts = []
    for key in sentences:
        text = []
        for c in sentences[key]:
            if len(c[3]) == 1:
                text.append((c[3], c[1] + c[2] if not c[2] is None else c[1]))
                continue
            for i in range(len(c[3])):
                rep = c[1] + "(" + str(i) + ")"
                text.append((c[3][i], rep + c[2] if not c[2] is None else rep))
        if len(text) != 0:
            texts.append(text)

    return texts


def build_id_dicts(texts):
    """
    Builds 4 dictionaries that connects between sign, trans and their ids (both ways)
    :param texts: the texts in a format for HMM
    :return: 4 dictionaries that connects between sign, trans and their ids (both ways)
    """

    sign_to_id, tran_to_id = rep_to_ix(reorganize_data(texts))
    id_to_sign = invert_dict(sign_to_id)
    id_to_tran = invert_dict(tran_to_id)
    return sign_to_id, tran_to_id, id_to_sign, id_to_tran


def write_data_for_allen_to_file(texts, file, sign_to_id, tran_to_id):
    """
    Saves data needed for allenNLP BiLSTM algorithm
    :param texts: the texts in a format for HMM
    :param file: file name to save
    :param sign_to_id: dictionary from sign to id
    :param tran_to_id: dictionary from transliteration to id
    :return: nothing
    """

    output_file = open(file, "w")

    for text in texts:
        for obj in text:
            output_file.write(str(sign_to_id[obj[0]]) + "###" + str(tran_to_id[obj[1]]) + " ")
        output_file.write("\n")

    output_file.close()


def preprocess():
    """
    Does all the preparations needed for HMM, MEMM, BiLSTM - reads and organizes the train, validation and test data
    :return: nothing
    """
    chars, _, _, _ = build_signs_and_transcriptions(["rinap/rinap5"])
    sentences = break_into_sentences(chars, None)
    #write_data_to_file(chars)
    d = build_dictionary(chars)
    write_dictionary_to_file(d)
    texts = build_data_for_hmm(sentences)
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
    """
    Test the preprocess for all the algorithms
    :return: nothing
    """
    preprocess()


if __name__ == '__main__':
    main()
