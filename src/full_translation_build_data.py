from build_data import build_signs_and_transcriptions, break_into_sentences
from pathlib import Path
import os
from data import from_key_to_text_and_line_numbers
from parse_xml import parse_xml
from statistics import mean
import matplotlib.pyplot as plt
from data import increment_count


def write_sentences_to_file(chars_sentences, translation_sentences):
    signs_file = open(Path(r"../NMT_input/signs_per_line.txt"), "w", encoding="utf8")
    transcription_file = open(Path(r"../NMT_input/transcriptions_per_line.txt"), "w", encoding="utf8")
    translation_file = open(Path(r"../NMT_input/translation_per_line.txt"), "w", encoding="utf8")

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
    build_graph(translation_lengths, "word translations in a line")

    signs_file.close()
    transcription_file.close()
    translation_file.close()


def build_translations(corpora, mapping):
    base_directory = Path(r"../raw_data/tei/")
    all_translations = {}

    for corpus in corpora:
        directory = base_directory / corpus
        for r, d, f in os.walk(directory):
            for file in f:
                translation = parse_xml(os.path.join(r, file), mapping, corpus)
                all_translations.update(translation)

    return all_translations


def build_full_line_translation_process(corpora):
    chars, translation, mapping, lines_cut_by_translation = build_signs_and_transcriptions(corpora, True)
    chars_sentences = break_into_sentences(chars, lines_cut_by_translation)
    translation_sentences = break_into_sentences(translation, lines_cut_by_translation)
    write_sentences_to_file(chars_sentences, translation_sentences)
    return chars_sentences, mapping


def build_graph(translation_lengths, name):
    # matplotlib histogram
    plt.hist(translation_lengths, color='blue', edgecolor='black', bins=100)

    # Add labels
    plt.title('Histogram of Translation Lengths - ' + name)
    plt.xlabel('Number of Words in a Sentence')
    plt.ylabel('Number of Sentences')

    plt.savefig(Path(r"../output/" + name))


def get_dict_sorted(d):
    return str({k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)})


def get_rare_elements_number(d, n):
    i = 0
    for k, v in d.items():
        if v < n:
            i += 1

    return str(i)


def print_statistics(translation_lengths, long_trs, very_long_trs, signs_vocab, transcription_vocab, translation_vocab,
                     could_divide_by_three_dots, could_not_divide):
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

    build_graph(translation_lengths, "real translations")


def compute_translation_statistics(tr, translation_lengths, long_trs, very_long_trs, translation_vocab):
    # Statistics of translation lengths
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
                            could_divide_by_three_dots, could_not_divide):
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
    if len(splitted_signs) == len(splitted_transcription) and len(splitted_transcription) == len(splitted_translation):
        could_divide_by_three_dots += 1

        for i in range(len(splitted_signs)):
            signs_file.write(str(real_key) + "[" + str(i + 1) + "]: " + splitted_signs[i] + "\n")
            transcription_file.write(str(real_key) + "[" + str(i + 1) + "]: " + splitted_transcription[i] + "\n")
            translation_file.write(str(real_key) + "[" + str(i + 1) + "]: " + splitted_translation[i] + "\n")

            translation_lengths, long_trs, very_long_trs, translation_vocab = \
                compute_translation_statistics(splitted_translation[i], translation_lengths, long_trs, very_long_trs,
                                               translation_vocab)

    else:
        could_not_divide += 1
        signs_file.write(str(real_key) + ": " + signs + "\n")
        transcription_file.write(str(real_key) + ": " + transcription + "\n")
        translation_file.write(str(real_key) + ": " + prev_tr + "\n")

        translation_lengths, long_trs, very_long_trs, translation_vocab = \
            compute_translation_statistics(prev_tr, translation_lengths, long_trs, very_long_trs, translation_vocab)

    return signs_vocab, transcription_vocab, translation_lengths, long_trs, very_long_trs, translation_vocab, \
           could_divide_by_three_dots, could_not_divide


def write_translations_to_file(chars_sentences, translations):
    signs_file = open(Path(r"../NMT_input/signs.txt"), "w", encoding="utf8")
    transcription_file = open(Path(r"../NMT_input/transcriptions.txt"), "w", encoding="utf8")
    translation_file = open(Path(r"../NMT_input/translation.txt"), "w", encoding="utf8")

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
                                    translation_file, could_divide_by_three_dots, could_not_divide)
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
                                        translation_file, could_divide_by_three_dots, could_not_divide)

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
                                    translation_file, could_divide_by_three_dots, could_not_divide)

    print_statistics(translation_lengths, long_trs, very_long_trs, signs_vocab, transcription_vocab,
                         translation_vocab, could_divide_by_three_dots, could_not_divide)

    signs_file.close()
    transcription_file.close()
    translation_file.close()


def preprocess(corpora):
    chars_sentences, mapping = build_full_line_translation_process(corpora)
    translations = build_translations(corpora, mapping)
    write_translations_to_file(chars_sentences, translations)


def main():
    corpora = ["rinap", "riao", "ribo", "saao", "suhu"]
    preprocess(corpora)


if __name__ == '__main__':
    main()
