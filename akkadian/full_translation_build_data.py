from pathlib import Path
import os
from statistics import mean
import matplotlib.pyplot as plt
from random import randint
import pickle
from akkadian.build_data import build_signs_and_transcriptions, break_into_sentences
from akkadian.data import from_key_to_text_and_line_numbers
from akkadian.parse_xml import parse_xml
from akkadian.data import increment_count


def write_sentences_to_file(chars_sentences, translation_sentences, signs_path, transcription_path, translation_path):
    """
    Write the data of word by word translations to files (different files for signs, transliterations and translations)
    :param chars_sentences: sentences with the signs and transliterations
    :param translation_sentences: translations done word by word for the corresponding chars_sentences
    :return: nothing, signs, transliterations and translations written to corresponding files
    """
    signs_file = open(signs_path, "w", encoding="utf8")
    transcription_file = open(transcription_path, "w", encoding="utf8")
    translation_file = open(translation_path, "w", encoding="utf8")

    translation_lengths = []
    for key in translation_sentences:
        signs_file.write(key + ": ")
        transcription_file.write(key + ": ")
        translation_file.write(key + ": ")

        for c in chars_sentences[key]:
            signs_file.write(c[3])
            delim = c[2] if not c[2] is None else " "
            transcription_file.write(c[1] + delim)

        translation_lengths.append(len(translation_sentences[key]))
        for t in translation_sentences[key]:
            translation_file.write(t[1] + " ")

        signs_file.write("\n")
        transcription_file.write("\n")
        translation_file.write("\n")

    print("Number of word translations in a line is: " + str(len(translation_lengths)))
    print("Mean word translations in a line length is: " + str(mean(translation_lengths)))
    # build_graph(translation_lengths, "word translations in a line")

    signs_file.close()
    transcription_file.close()
    translation_file.close()


def write_sentences_to_file_no_translation(chars_sentences, transcription_path):
    """
    Write the data of word by word translations to files (different files for signs, transliterations and translations)
    :param chars_sentences: sentences with the signs and transliterations
    :return: nothing, signs, transliterations and translations written to corresponding files
    """
    transcription_file = open(transcription_path, "w", encoding="utf8")

    for key in chars_sentences:
        transcription_file.write(key + ": ")

        for c in chars_sentences[key]:
            delim = c[2] if not c[2] is None else " "
            transcription_file.write(c[1] + delim)

        transcription_file.write("\n")

    transcription_file.close()


def build_translations(corpora, mapping):
    """
    Build translations for preprocess
    :param corpora: corpora to use for building the data for full translation
    :param mapping: mapping between different numbering of lines
    :return: translations
    """
    base_directory = Path(r"../raw_data/tei/")
    all_translations = {}

    for corpus in corpora:
        directory = base_directory / corpus
        for r, d, f in os.walk(directory):
            for file in f:
                translation = parse_xml(os.path.join(r, file), mapping, corpus)
                all_translations.update(translation)

    return all_translations


def build_full_line_translation_process(corpora, has_translation, signs_path, transcription_path, translation_path):
    """
    Do first part of preprocess, build signs and transliterations
    :param corpora: corpora to use for building the data for full translation
    :return: signs, transliterations and mapping between different numbering of lines
    """
    chars, translation, mapping, lines_cut_by_translation = build_signs_and_transcriptions(corpora, True)
    chars_sentences = break_into_sentences(chars, lines_cut_by_translation)
    if has_translation:
        translation_sentences = break_into_sentences(translation, lines_cut_by_translation)
        write_sentences_to_file(chars_sentences, translation_sentences, signs_path, transcription_path,
                                translation_path)
    else:
        write_sentences_to_file_no_translation(chars_sentences, transcription_path)
    return chars_sentences, mapping


def build_graph(translation_lengths, name):
    """
    Build a graph to show different translation lengths and their frequencies
    :param translation_lengths: list of all translation lengths
    :param name: name for the graph
    :return: nothing, a graph is saved to a file
    """
    # matplotlib histogram
    plt.hist(translation_lengths, color='blue', edgecolor='black', bins=100)

    # Add labels
    plt.title('Histogram of Translation Lengths - ' + name)
    plt.xlabel('Number of Words in a Sentence')
    plt.ylabel('Number of Sentences')

    plt.savefig(Path(r".output/" + name))


def get_dict_sorted(d):
    """
    Sort a dictionary
    :param d: dictionary to be sorted
    :return: the dictionary after sorting
    """
    return str({k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)})


def get_rare_elements_number(d, n):
    """
    Count the number of rare elements
    :param d: dictionary to use
    :param n: the threshold for rarity
    :return: the number of rare elements as a string
    """
    i = 0
    for k, v in d.items():
        if v < n:
            i += 1

    return str(i)


def print_statistics(translation_lengths, long_trs, very_long_trs, signs_vocab, transcription_vocab, translation_vocab,
                     could_divide_by_three_dots, could_not_divide):
    """
    Print all the statistics computed
    :param translation_lengths: list of all translation lengths
    :param long_trs: counter for long translations
    :param very_long_trs: counter for very long translations
    :param signs_vocab: vocabulary of all the signs
    :param transcription_vocab: vocabulary of all the transliterations
    :param translation_vocab: vocabulary of all the words in different translations
    :param could_divide_by_three_dots: counter for translations possible to divide based on three dots
    :param could_not_divide: counter for translations not possible to divide based on three dots
    :return: nothing, all data is printed to stdout
    """
    print("Number of real translations is: " + str(len(translation_lengths)))
    print("Mean real translations length is: " + str(mean(translation_lengths)))
    print("Number of real translations longer than 50 is: " + str(long_trs))
    print("Number of real translations longer than 200 is: " + str(very_long_trs))

    print("Size of signs vocabulary is: " + str(len(signs_vocab)))
    print("Number of signs with less than 5 occurrences is: " + get_rare_elements_number(signs_vocab, 5))
    print("The signs vocabulary is: " + get_dict_sorted(signs_vocab))

    print("Size of transliteration vocabulary is: " + str(len(transcription_vocab)))
    print("Number of transliterations with less than 5 occurrences is: " +
          get_rare_elements_number(transcription_vocab, 5))
    print("The transliteration vocabulary is: " + get_dict_sorted(transcription_vocab))

    print("Size of translation (English) vocabulary is: " + str(len(translation_vocab)))
    print("Number of translations (English) with less than 5 occurrences is: " +
          get_rare_elements_number(translation_vocab, 5))
    print("The translation (English) vocabulary is: " + get_dict_sorted(translation_vocab))

    print("Number of sentences that were divided by three dots is: " + str(could_divide_by_three_dots))
    print("Number of sentences that were not able to be divided is: " + str(could_not_divide))

    # build_graph(translation_lengths, "real translations")


def compute_translation_statistics(tr, translation_lengths, long_trs, very_long_trs, translation_vocab):
    """
    Compute statistics related to translation
    :param tr: current translation
    :param translation_lengths: list of all translation lengths
    :param long_trs: counter for long translations
    :param very_long_trs: counter for very long translations
    :param translation_vocab: vocabulary of all the words in different translations
    :return: the four last parameters to the function after updated for current translation
    """
    translation_lengths.append(len(tr.split()))

    if len(tr.split()) > 50:
        long_trs += 1

    if len(tr.split()) > 200:
        very_long_trs += 1

    for word in tr.split():
        word = word.replace(",", "").replace("!", "").replace("?", "").replace(":", "").replace(";", "")
        if word.replace(".", "") == "":
            word = "..."
        else:
            word = word.replace(".", "")
        increment_count(translation_vocab, word)

    return translation_lengths, long_trs, very_long_trs, translation_vocab


def clean_signs_transcriptions(signs, is_signs):
    """
    Clean the signs and transcriptions and canonize them
    :param signs: signs / transliterations
    :param is_signs: True if we are dealing with signs
    :return: signs / transliterations after clean is done
    """
    start_index = 0

    while start_index < len(signs):
        index1 = signs.find(".", start_index, len(signs))
        index2 = signs.find("x", start_index, len(signs))

        if index1 != -1 or index2 != -1:
            if index1 != -1 and index2 == -1:
                index = index1
            elif index1 == -1 and index2 != -1:
                index = index2
            else:
                index = min(index1, index2)

            end_index = index
            if is_signs:
                while end_index < len(signs) and (signs[end_index] == "." or signs[end_index] == "x"):
                    end_index += 1

                signs = signs[:index] + "..." + signs[end_index:]
                start_index = index + 3

            else:
                while end_index < len(signs) and (signs[end_index] == "." or signs[end_index] == "x"
                                                  or signs[end_index] == " " or signs[end_index] == "-"
                                                  or signs[end_index] == "+" or signs[end_index] == "—"
                                                  or signs[end_index] == "ₓ"):
                    end_index += 1

                sub_signs = signs[index:end_index]
                if sub_signs == ".":
                    start_index = index + 1
                elif sub_signs == ". ":
                    start_index = index + 2
                elif sub_signs == ".-":
                    start_index = index + 2
                elif sub_signs == ".—":
                    start_index = index + 2
                elif sub_signs == "xₓ":
                    start_index = index + 2
                elif sub_signs == "xₓ—":
                    start_index = index + 3
                else:
                    signs = signs[:index] + "... " + signs[end_index:]
                    start_index = index + 4

        else:
            start_index = len(signs)

    return signs


def add_translation_to_file(prev_signs, signs_vocab, prev_transcription, transcription_vocab, prev_tr,
                            translation_lengths, long_trs, very_long_trs, translation_vocab, prev_text,
                            prev_start_line, prev_end_line, signs_file, transcription_file, translation_file,
                            could_divide_by_three_dots, could_not_divide, metadata=False, divide_by_three_dots=True):
    """
    Add a translation with corresponding signs and transliterations to files
    :param prev_signs: previous signs written to file
    :param signs_vocab: vocabulary of all the signs
    :param prev_transcription: previous transliterations written to file
    :param transcription_vocab: vocabulary of all the transliterations
    :param prev_tr: previous translation written to file
    :param translation_lengths: list of all translation lengths
    :param long_trs: counter for long translations
    :param very_long_trs: counter for very long translations
    :param translation_vocab: vocabulary of all the words in different translations
    :param prev_text: previous text written to file
    :param prev_start_line: previous start line written to file
    :param prev_end_line: previous end line written to file
    :param signs_file: file of all signs, being built as input for translation algorithms
    :param transcription_file: file of all transliterations, being built as input for translation algorithms
    :param translation_file: file of all translations, being built as input for translation algorithms
    :param could_divide_by_three_dots: counter for translations possible to divide based on three dots
    :param could_not_divide: counter for translations not possible to divide based on three dots
    :param metadata: should add the id of each sample to the files
    :return: some of the parameters to the function, after update
    """
    signs = ""
    transcription = ""

    for sign in prev_signs:
        signs += sign
        increment_count(signs_vocab, sign)

    for t, delim in prev_transcription:
        transcription += t + delim
        increment_count(transcription_vocab, t)

    signs = clean_signs_transcriptions(signs, True)
    transcription = clean_signs_transcriptions(transcription, False)

    real_key = [prev_text + "." + str(prev_start_line), prev_text + "." + str(prev_end_line)]

    splitted_signs = [s for s in signs.split("...") if s != "" and s != " "]
    splitted_transcription = [t for t in transcription.split("... ") if t != "" and t != " "]
    splitted_translation = [tr for tr in prev_tr.split("... ") if tr != "" and tr != " "]

    # Write to files
    if len(splitted_signs) == len(splitted_transcription) and len(splitted_transcription) == len(splitted_translation) \
            and divide_by_three_dots:
        could_divide_by_three_dots += 1

        for i in range(len(splitted_signs)):
            if metadata:
                signs_file.write(str(real_key) + "[" + str(i + 1) + "]: " + splitted_signs[i] + "\n")
                transcription_file.write(str(real_key) + "[" + str(i + 1) + "]: " + splitted_transcription[i] + "\n")
                translation_file.write(str(real_key) + "[" + str(i + 1) + "]: " + splitted_translation[i] + "\n")
            else:
                signs_file.write(splitted_signs[i] + "\n")
                transcription_file.write(splitted_transcription[i] + "\n")
                translation_file.write(splitted_translation[i] + "\n")

            translation_lengths, long_trs, very_long_trs, translation_vocab = \
                compute_translation_statistics(splitted_translation[i], translation_lengths, long_trs, very_long_trs,
                                               translation_vocab)

    else:
        could_not_divide += 1
        if metadata:
            signs_file.write(str(real_key) + ": " + signs + "\n")
            transcription_file.write(str(real_key) + ": " + transcription + "\n")
            translation_file.write(str(real_key) + ": " + prev_tr + "\n")
        else:
            signs_file.write(signs + "\n")
            transcription_file.write(transcription + "\n")
            translation_file.write(prev_tr + "\n")

        translation_lengths, long_trs, very_long_trs, translation_vocab = \
            compute_translation_statistics(prev_tr, translation_lengths, long_trs, very_long_trs, translation_vocab)

    return signs_vocab, transcription_vocab, translation_lengths, long_trs, very_long_trs, translation_vocab, \
           could_divide_by_three_dots, could_not_divide


def write_translations_to_file(chars_sentences, translations, signs_path, transcription_path, translation_path, divide_by_three_dots):
    """
    Write all the data we collected (signs, transliterations and translations) to proper files
    :param chars_sentences: sentences of the signs ans transliterations
    :param translations: translations corresponding to the signs and transliterations
    :return: nothing, the signs, transliterations and translations are written to different files
    """
    signs_file = open(signs_path, "w", encoding="utf8")
    transcription_file = open(transcription_path, "w", encoding="utf8")
    translation_file = open(translation_path, "w", encoding="utf8")

    translation_lengths = []
    long_trs = 0
    very_long_trs = 0
    signs_vocab = {}
    transcription_vocab = {}
    translation_vocab = {}
    could_divide_by_three_dots = 0
    could_not_divide = 0

    prev_text = ""
    prev_start_line = ""
    prev_end_line = ""
    prev_key = ""
    prev_signs = []
    prev_transcription = []
    prev_tr = ""
    prev_should_add = False
    for key in translations.keys():
        text, start_line, end_line = from_key_to_text_and_line_numbers(key)

        if start_line == -1:
            if prev_should_add == True and len(prev_signs) != 0:
                signs_vocab, transcription_vocab, translation_lengths, long_trs, very_long_trs, translation_vocab, \
                could_divide_by_three_dots, could_not_divide = \
                    add_translation_to_file(prev_signs, signs_vocab, prev_transcription, transcription_vocab, prev_tr,
                                    translation_lengths, long_trs, very_long_trs, translation_vocab, prev_text,
                                    prev_start_line, prev_end_line, signs_file, transcription_file,
                                    translation_file, could_divide_by_three_dots, could_not_divide, False, divide_by_three_dots)
            prev_should_add = False
            continue

        cur_signs = []
        cur_transcription = []
        for n in range(start_line, end_line + 1):
            k = text + "." + str(n)
            if k not in chars_sentences.keys():
                # Handle lines divided between sentences.
                if start_line == end_line:
                    if prev_key[1] == key[0]:
                        if k + "(part 2)" in chars_sentences.keys():
                            k = k + "(part 2)"
                            start_line = str(start_line) + "(part 2)"
                            end_line = start_line
                        else:
                            continue
                    else:
                        if k + "(part 1)" in chars_sentences.keys():
                            k = k + "(part 1)"
                            start_line = str(start_line) + "(part 1)"
                            end_line = start_line
                        else:
                            continue

                elif n == start_line and k + "(part 2)" in chars_sentences.keys():
                    k = k + "(part 2)"
                    start_line = str(start_line) + "(part 2)"

                elif n == end_line and k + "(part 1)" in chars_sentences.keys():
                    k = k + "(part 1)"
                    end_line = str(end_line) + "(part 1)"

                else:
                    continue

            for c in chars_sentences[k]:
                cur_signs.append(c[3])
                delim = c[2] if not c[2] is None else " "
                cur_transcription.append((c[1], delim))

        cur_tr = translations[key]

        if text == prev_text and start_line == prev_end_line:
            # The translation is not accurate, because it didn't give exact division point, so we don't use it.
            prev_should_add = False
        else:
            if prev_should_add == True and len(prev_signs) != 0:
                signs_vocab, transcription_vocab, translation_lengths, long_trs, very_long_trs, translation_vocab, \
                could_divide_by_three_dots, could_not_divide = \
                    add_translation_to_file(prev_signs, signs_vocab, prev_transcription, transcription_vocab, prev_tr,
                                        translation_lengths, long_trs, very_long_trs, translation_vocab, prev_text,
                                        prev_start_line, prev_end_line, signs_file, transcription_file,
                                        translation_file, could_divide_by_three_dots, could_not_divide, False, divide_by_three_dots)

            prev_should_add = True

        prev_text = text
        prev_start_line = start_line
        prev_end_line = end_line
        prev_key = key
        prev_signs = cur_signs
        prev_transcription = cur_transcription
        prev_tr = cur_tr

    if prev_should_add == True and len(prev_signs) != 0:
        signs_vocab, transcription_vocab, translation_lengths, long_trs, very_long_trs, translation_vocab, \
        could_divide_by_three_dots, could_not_divide = \
            add_translation_to_file(prev_signs, signs_vocab, prev_transcription, transcription_vocab, prev_tr,
                                    translation_lengths, long_trs, very_long_trs, translation_vocab, prev_text,
                                    prev_start_line, prev_end_line, signs_file, transcription_file,
                                    translation_file, could_divide_by_three_dots, could_not_divide, False, divide_by_three_dots)

    print_statistics(translation_lengths, long_trs, very_long_trs, signs_vocab, transcription_vocab,
                         translation_vocab, could_divide_by_three_dots, could_not_divide)

    signs_file.close()
    transcription_file.close()
    translation_file.close()


def preprocess(corpora, divide_by_three_dots):
    """
    Process corpora for the input of the translation algorithms
    :param corpora: corpora to process
    :return: nothing
    """
    chars_sentences, mapping = build_full_line_translation_process(corpora,
                                                                   True,
                                                                   Path(r"../NMT_input/signs_per_line.txt"),
                                                                   Path(r"../NMT_input/transcriptions_per_line.txt"),
                                                                   Path(r"../NMT_input/translation_per_line.txt"))
    translations = build_translations(corpora, mapping)
    if divide_by_three_dots:
        write_translations_to_file(chars_sentences,
                                   translations,
                                   Path(r"../NMT_input/signs.txt"),
                                   Path(r"../NMT_input/transcriptions.txt"),
                                   Path(r"../NMT_input/translation.txt"),
                                   True)
    else:
        write_translations_to_file(chars_sentences,
                                   translations,
                                   Path(r"../NMT_input/not_divided_by_three_dots/signs.txt"),
                                   Path(r"../NMT_input/not_divided_by_three_dots/transcriptions.txt"),
                                   Path(r"../NMT_input/not_divided_by_three_dots/translation.txt"),
                                   False)


def preprocess_not_translated_corpora(corpora):
    chars_sentences, mapping = build_full_line_translation_process(corpora,
                                                                   False,
                                                                   None,
                                                                   Path(r"../NMT_input/for_translation.tr"),
                                                                   None)


def write_train_valid_test_files(file_type, lang, valid_lines, test_lines, divide_by_three_dots):
    if divide_by_three_dots:
        f = open(Path(r"../NMT_input/" + file_type + ".txt"), "r", encoding="utf8")
        train = open(Path(r"../NMT_input/train." + lang), "w", encoding="utf8")
        valid = open(Path(r"../NMT_input/valid." + lang), "w", encoding="utf8")
        test = open(Path(r"../NMT_input/test." + lang), "w", encoding="utf8")
    else:
        f = open(Path(r"../NMT_input/not_divided_by_three_dots/" + file_type + ".txt"), "r", encoding="utf8")
        train = open(Path(r"../NMT_input/not_divided_by_three_dots/train." + lang), "w", encoding="utf8")
        valid = open(Path(r"../NMT_input/not_divided_by_three_dots/valid." + lang), "w", encoding="utf8")
        test = open(Path(r"../NMT_input/not_divided_by_three_dots/test." + lang), "w", encoding="utf8")

    for i, line in enumerate(f):
        if i in valid_lines:
            valid.write(line)
        elif i in test_lines:
            test.write(line)
        else:
            train.write(line)

    f.close()
    train.close()
    valid.close()
    test.close()


def divide_to_train_valid_test(divide_by_three_dots):
    if divide_by_three_dots:
        file = Path(r"../NMT_input/signs.txt")
    else:
        file = Path(r"../NMT_input/not_divided_by_three_dots/signs.txt")

    with open(file, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            pass
    line_number = i + 1

    train_lines = []
    valid_lines = []
    test_lines = []

    for j in range(line_number):
        random_number = randint(1, 20)
        if random_number == 1:
            valid_lines.append(j)
        elif random_number == 2:
            test_lines.append(j)
        else:
            train_lines.append(j)

    return valid_lines, test_lines


def build_train_valid_test(divide_by_three_dots):
    if divide_by_three_dots:
        valid_lines_pkl = Path(r"../NMT_input/valid_lines.pkl")
        test_lines_pkl = Path(r"../NMT_input/test_lines.pkl")
    else:
        valid_lines_pkl = Path(r"../NMT_input/not_divided_by_three_dots/valid_lines.pkl")
        test_lines_pkl = Path(r"../NMT_input/not_divided_by_three_dots/test_lines.pkl")

    valid_lines, test_lines = divide_to_train_valid_test(divide_by_three_dots)
    with open(valid_lines_pkl, "wb") as f:
        pickle.dump(valid_lines, f)
    with open(valid_lines_pkl, "rb") as f:
        valid_lines_file = pickle.load(f)
    assert valid_lines == valid_lines_file

    with open(test_lines_pkl, "wb") as f:
        pickle.dump(test_lines, f)
    with open(test_lines_pkl, "rb") as f:
        test_lines_file = pickle.load(f)
    assert test_lines == test_lines_file

    write_train_valid_test_files("signs", "ak", valid_lines_file, test_lines_file, divide_by_three_dots)
    write_train_valid_test_files("transcriptions", "tr", valid_lines_file, test_lines_file, divide_by_three_dots)
    write_train_valid_test_files("translation", "en", valid_lines_file, test_lines_file, divide_by_three_dots)


def main():
    """
    Builds data for translation algorithms
    :return: nothing
    """
    corpora = ["rinap", "riao", "ribo", "saao", "suhu"]
    not_translated_corpora = ["atae"]
    divide_by_three_dots = False
    preprocess(corpora, divide_by_three_dots)
    build_train_valid_test(divide_by_three_dots)
    preprocess_not_translated_corpora(not_translated_corpora)



if __name__ == '__main__':
    main()
