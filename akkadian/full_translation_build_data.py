from pathlib import Path
import os
from statistics import mean
import matplotlib.pyplot as plt
from random import randint
from akkadian.build_data import build_signs_and_transcriptions, break_into_sentences
from akkadian.data import from_key_to_text_and_line_numbers
from akkadian.parse_xml import parse_xml
from akkadian.data import increment_count


def write_sentences_to_file(chars_sentences, translation_sentences):
    """
    Write the data of word by word translations to files (different files for signs, transliterations and translations)
    :param chars_sentences: sentences with the signs and transliterations
    :param translation_sentences: translations done word by word for the corresponding chars_sentences
    :return: nothing, signs, transliterations and translations written to corresponding files
    """
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
    # build_graph(translation_lengths, "word translations in a line")

    signs_file.close()
    transcription_file.close()
    translation_file.close()


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


def build_full_line_translation_process(corpora):
    """
    Do first part of preprocess, build signs and transliterations
    :param corpora: corpora to use for building the data for full translation
    :return: signs, transliterations and mapping between different numbering of lines
    """
    chars, translation, mapping, lines_cut_by_translation = build_signs_and_transcriptions(corpora, True)
    chars_sentences = break_into_sentences(chars, lines_cut_by_translation)
    translation_sentences = break_into_sentences(translation, lines_cut_by_translation)
    write_sentences_to_file(chars_sentences, translation_sentences)
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
                            could_divide_by_three_dots, could_not_divide, metadata=False):
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
    if len(splitted_signs) == len(splitted_transcription) and len(splitted_transcription) == len(splitted_translation):
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


def write_translations_to_file(chars_sentences, translations):
    """
    Write all the data we collected (signs, transliterations and translations) to proper files
    :param chars_sentences: sentences of the signs ans transliterations
    :param translations: translations corresponding to the signs and transliterations
    :return: nothing, the signs, transliterations and translations are written to different files
    """
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
                                    translation_file, could_divide_by_three_dots, could_not_divide, False)
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
                                        translation_file, could_divide_by_three_dots, could_not_divide, False)

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
                                    translation_file, could_divide_by_three_dots, could_not_divide, False)

    print_statistics(translation_lengths, long_trs, very_long_trs, signs_vocab, transcription_vocab,
                         translation_vocab, could_divide_by_three_dots, could_not_divide)

    signs_file.close()
    transcription_file.close()
    translation_file.close()


def preprocess(corpora):
    """
    Process corpora for the input of the translation algorithms
    :param corpora: corpora to process
    :return: nothing
    """
    chars_sentences, mapping = build_full_line_translation_process(corpora)
    translations = build_translations(corpora, mapping)
    write_translations_to_file(chars_sentences, translations)


def write_train_valid_test_files(file_type, lang, valid_lines, test_lines):
    f = open(Path(r"../NMT_input/" + file_type + ".txt"), "r", encoding="utf8")
    train = open(Path(r"../NMT_input/train/train." + lang), "w", encoding="utf8")
    valid = open(Path(r"../NMT_input/valid/valid." + lang), "w", encoding="utf8")
    test = open(Path(r"../NMT_input/test/test." + lang), "w", encoding="utf8")

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


def divide_to_train_valid_test():
    with open(Path(r"../NMT_input/signs.txt"), "r", encoding="utf8") as f:
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


def build_train_valid_test():
    # valid_lines, test_lines = divide_to_train_valid_test()
    valid_lines = [11, 26, 34, 60, 89, 139, 168, 192, 194, 203, 255, 256, 259, 312, 348, 353, 354, 434, 447, 459, 496, 503, 505, 513, 519, 558, 598, 617, 622, 667, 768, 790, 823, 848, 850, 859, 897, 923, 930, 975, 979, 984, 1006, 1031, 1045, 1068, 1111, 1173, 1183, 1211, 1234, 1236, 1273, 1289, 1305, 1322, 1337, 1339, 1365, 1370, 1384, 1400, 1401, 1411, 1496, 1574, 1575, 1582, 1603, 1631, 1632, 1640, 1648, 1653, 1667, 1669, 1675, 1698, 1743, 1763, 1778, 1792, 1802, 1806, 1828, 1846, 1869, 1873, 1916, 1923, 1927, 1960, 1975, 1978, 2051, 2083, 2106, 2111, 2129, 2171, 2180, 2184, 2197, 2199, 2219, 2240, 2261, 2275, 2279, 2311, 2345, 2399, 2426, 2431, 2452, 2459, 2469, 2478, 2519, 2561, 2569, 2593, 2594, 2611, 2670, 2671, 2698, 2746, 2752, 2788, 2795, 2822, 2825, 2841, 2907, 2932, 2934, 2951, 2961, 3008, 3020, 3050, 3124, 3131, 3135, 3139, 3142, 3152, 3173, 3182, 3203, 3212, 3214, 3240, 3242, 3277, 3282, 3289, 3292, 3295, 3304, 3350, 3387, 3420, 3443, 3469, 3516, 3553, 3555, 3587, 3592, 3614, 3632, 3643, 3680, 3710, 3731, 3774, 3787, 3790, 3796, 3798, 3807, 3825, 3871, 3953, 3958, 3966, 4015, 4044, 4071, 4090, 4092, 4097, 4165, 4186, 4212, 4253, 4293, 4361, 4366, 4376, 4418, 4446, 4468, 4475, 4488, 4493, 4501, 4506, 4508, 4515, 4521, 4540, 4579, 4634, 4661, 4666, 4703, 4707, 4751, 4808, 4823, 4838, 4839, 4851, 4855, 4889, 4897, 4932, 4933, 4937, 4953, 4961, 4980, 4996, 5005, 5031, 5074, 5090, 5106, 5111, 5112, 5122, 5179, 5184, 5219, 5255, 5268, 5324, 5334, 5351, 5388, 5408, 5410, 5427, 5437, 5446, 5457, 5462, 5463, 5467, 5482, 5483, 5490, 5509, 5523, 5548, 5582, 5600, 5608, 5618, 5676, 5693, 5697, 5703, 5707, 5715, 5717, 5720, 5727, 5744, 5783, 5800, 5809, 5819, 5850, 5867, 5880, 5883, 5891, 5912, 5919, 5924, 5934, 5941, 5988, 5994, 6008, 6016, 6023, 6047, 6065, 6068, 6073, 6079, 6099, 6100, 6214, 6227, 6228, 6237, 6275, 6352, 6404, 6424, 6466, 6529, 6547, 6565, 6570, 6648, 6652, 6675, 6677, 6703, 6729, 6741, 6764, 6818, 6824, 6850, 6906, 6952, 6975, 7007, 7022, 7036, 7046, 7070, 7071, 7076, 7091, 7127, 7148, 7164, 7176, 7179, 7182, 7189, 7209, 7234, 7274, 7277, 7298, 7316, 7320, 7334, 7351, 7354, 7364, 7367, 7372, 7382, 7397, 7402, 7410, 7416, 7423, 7454, 7468, 7474, 7501, 7514, 7536, 7563, 7573, 7587, 7608, 7615, 7632, 7655, 7667, 7669, 7675, 7677, 7679, 7690, 7700, 7749, 7806, 7816, 7820, 7826, 7884, 7890, 7917, 7954, 7965, 8002, 8005, 8018, 8024, 8038, 8061, 8072, 8162, 8165, 8179, 8221, 8243, 8285, 8286, 8299, 8302, 8326, 8405, 8442, 8465, 8468, 8509, 8555, 8571, 8596, 8599, 8617, 8656, 8671, 8860, 8883, 8894, 8918, 8932, 8959, 8973, 9038, 9048, 9052, 9066, 9077, 9123, 9124, 9162, 9171, 9175, 9223, 9280, 9282, 9286, 9357, 9358, 9366, 9379, 9437, 9477, 9502, 9505, 9546, 9626, 9742, 9775, 9790, 9793, 9807, 9850, 9851, 9917, 9919, 9944, 9955, 9964, 10021, 10071, 10098, 10138, 10158, 10178, 10183, 10215, 10216, 10252, 10262, 10290, 10302, 10305, 10317, 10328, 10338, 10347, 10416, 10444, 10467, 10471, 10484, 10503, 10521, 10545, 10564, 10566, 10636, 10638, 10683, 10689, 10690, 10703, 10731, 10743, 10770, 10806, 10823, 10837, 10844, 10879, 10885, 10889, 10893, 10898, 10906, 10910, 10943, 10947, 10950, 10961, 10978, 10998, 11051, 11097, 11134, 11157, 11205, 11236, 11291, 11305, 11312, 11327, 11354, 11357, 11367, 11402, 11409, 11414, 11423, 11431, 11436, 11440, 11494, 11537, 11538, 11567, 11595, 11626, 11638, 11653, 11675, 11686, 11697, 11715, 11735, 11752, 11757, 11847, 11850, 11882, 11922, 11923, 11937, 11938, 11960, 12007, 12064, 12091, 12109, 12118, 12157, 12170, 12176, 12178, 12180, 12184, 12201, 12210, 12225, 12240, 12245, 12262, 12295, 12346, 12388, 12394, 12420, 12430, 12439, 12452, 12467, 12471, 12493, 12511, 12546, 12581, 12635, 12643, 12655, 12699, 12712, 12724, 12735, 12743, 12765, 12769, 12775, 12776, 12835, 12836, 12903, 12925, 12946, 12951, 12964, 12975, 13020, 13023, 13041, 13072, 13092, 13121, 13124, 13142, 13156, 13162, 13177, 13214, 13240, 13249, 13261, 13326, 13372, 13373, 13442, 13523, 13541, 13561, 13571, 13587, 13603, 13617, 13619, 13620, 13622, 13634, 13644, 13653, 13657, 13668, 13707, 13714, 13719, 13734, 13754, 13791, 13806, 13810, 13814, 13816, 13838, 13865, 13866, 13869, 13876, 13904, 13957, 13963, 13977, 13997, 14050, 14067, 14082, 14103, 14113, 14119, 14121, 14123, 14194, 14279, 14291, 14293, 14296, 14302, 14317, 14322, 14348, 14357, 14361, 14365, 14370, 14373, 14387, 14395, 14399, 14435, 14446, 14579, 14604, 14627, 14647, 14653, 14658, 14666, 14669, 14687, 14702, 14717, 14737, 14738, 14767, 14789, 14797, 14802, 14832, 14834, 14865, 14923, 14939, 14942, 14974, 15037, 15046, 15056, 15110, 15125, 15140, 15150, 15175, 15177, 15196, 15199, 15225, 15232, 15318, 15322, 15346, 15362, 15396, 15411, 15488, 15495, 15524, 15534, 15572, 15634, 15654, 15663, 15677, 15718, 15719, 15753, 15755, 15763, 15776, 15809, 15879, 15912, 15942, 15963, 15971, 15995, 16007, 16025, 16035, 16039, 16052, 16057, 16066, 16085, 16112, 16116, 16126, 16129, 16133, 16139, 16202, 16216, 16279, 16286, 16302, 16328, 16333, 16339, 16361, 16374, 16378, 16381, 16406, 16413, 16422, 16423, 16436, 16458, 16524, 16525, 16531, 16549, 16603, 16659, 16661, 16670, 16675, 16680, 16723, 16728, 16735, 16748, 16759, 16764, 16768, 16771, 16775, 16795, 16797, 16834, 16838, 16880, 16881, 16896, 16915, 16961, 16981, 17022, 17038, 17051, 17066, 17073, 17084, 17095, 17126, 17144, 17162, 17173, 17183, 17193, 17224, 17231, 17236, 17244, 17270, 17289, 17332, 17352, 17365, 17372, 17388, 17391, 17454, 17458, 17480, 17534, 17536, 17539, 17543, 17601, 17681, 17704, 17717, 17720, 17729, 17733, 17774, 17788, 17806, 17839, 17853, 17856, 17874, 17893, 17895, 17904, 17952, 17956, 18013, 18021, 18031, 18061, 18077, 18105, 18110, 18130, 18132, 18166, 18236, 18248, 18265, 18266, 18271, 18309, 18351, 18389, 18408, 18419, 18428, 18432, 18452, 18462, 18490, 18492, 18519, 18564, 18580, 18584, 18587, 18613, 18645, 18654, 18674, 18786, 18803, 18857, 18907, 18918, 18922, 18923, 18929, 18993, 18995, 19003, 19048, 19065, 19068, 19077, 19081, 19092, 19096, 19104, 19107, 19121, 19144, 19154, 19192, 19206, 19220, 19234, 19239, 19241, 19261, 19270, 19275, 19320, 19353, 19368, 19382, 19390, 19400, 19408, 19414, 19454, 19458, 19466, 19467, 19511, 19519, 19523, 19547, 19552, 19591, 19645, 19674, 19778, 19779, 19787, 19804, 19813, 19829, 19838, 19842, 19844, 19853, 19864, 19891, 19926, 19927, 19984, 19999, 20053, 20084, 20113, 20175, 20181, 20190, 20213, 20219, 20238, 20243, 20248, 20263, 20294, 20345, 20364, 20370, 20374, 20379, 20399, 20430, 20443, 20455, 20470, 20545, 20562, 20606, 20620, 20625, 20635, 20646, 20665, 20697, 20707, 20733, 20756, 20765, 20812, 20843, 20851, 20866, 20869, 20880, 20902, 20905, 20937, 20941, 20953, 20977, 21005, 21044, 21062, 21073, 21075, 21084, 21085, 21106, 21125, 21140, 21158, 21175, 21184, 21212, 21216, 21225, 21293, 21308, 21321, 21328, 21348, 21353, 21410, 21415, 21476, 21480, 21557, 21572, 21584, 21590, 21591, 21601, 21606, 21618, 21629, 21643, 21701, 21758, 21797, 21831, 21861, 21862, 21883, 21896, 21943, 21945, 21946, 21964, 21996, 22010, 22024, 22041, 22061, 22066, 22092, 22130, 22148, 22154, 22155, 22161, 22193, 22196, 22197, 22221, 22224, 22226, 22277, 22297, 22337, 22360, 22451, 22460, 22483, 22501, 22526, 22556, 22570, 22577, 22584, 22609, 22626, 22653, 22660, 22661, 22668, 22726, 22730, 22742, 22773, 22780, 22816, 22830, 22835, 22842, 22855, 22888, 22929, 22935, 22959, 22969, 22971, 22982, 23007, 23018, 23019, 23022, 23028, 23031, 23067, 23090, 23092, 23117, 23120, 23128, 23129, 23135, 23181, 23182, 23193, 23241, 23274, 23290, 23351, 23388, 23405, 23424, 23457, 23481, 23499, 23525, 23538, 23561, 23564, 23576, 23621, 23662, 23693, 23713, 23714, 23727, 23729, 23737, 23740, 23741, 23745, 23768, 23800, 23812, 23831, 23860, 23869, 23898, 23902, 23904, 23933, 23944, 23956, 24035, 24055, 24079, 24091, 24093, 24101, 24103, 24104, 24113, 24117, 24207, 24211, 24227, 24239, 24265, 24338, 24360, 24389, 24401, 24462, 24483, 24541, 24546, 24566, 24572, 24597, 24629, 24680, 24686, 24695, 24703, 24729, 24736, 24740, 24747, 24749, 24751, 24757, 24772, 24785, 24830, 24850, 24857, 24859, 24875, 24890, 24892, 24935, 24963, 24999, 25006, 25021, 25031, 25074, 25092, 25094, 25110, 25165, 25182, 25189, 25245, 25251, 25280, 25307, 25309, 25344, 25348, 25353, 25377, 25390, 25406, 25411, 25426, 25432, 25446, 25454, 25495, 25527, 25557, 25560, 25567, 25585, 25598, 25604, 25621, 25641, 25678, 25696, 25700, 25711, 25741, 25782, 25821, 25824, 25834, 25867, 25873, 25888, 25891, 25908, 25926, 25934, 25991, 25992, 26026, 26045, 26057, 26106, 26124, 26144, 26145, 26161, 26185, 26199, 26210, 26227, 26237, 26240, 26313, 26316, 26325, 26347, 26392, 26412, 26415, 26422, 26433, 26449, 26457, 26458, 26463, 26465, 26504, 26507, 26510, 26520, 26535, 26602, 26637, 26647, 26648, 26668, 26669, 26704, 26713, 26716, 26750, 26779, 26781, 26821, 26848, 26881, 26901, 26907, 26924, 26929, 26932, 26968, 26984, 26989, 26993, 27032, 27036, 27043, 27074, 27098, 27133, 27135, 27146, 27167, 27178, 27187, 27191, 27212, 27224, 27227, 27229, 27234, 27266, 27281, 27300, 27330, 27347, 27348, 27370, 27389, 27395, 27438, 27442, 27453, 27480, 27507, 27534, 27544, 27577, 27600, 27604, 27610, 27619, 27637, 27652, 27656, 27675, 27677, 27756, 27797, 27807, 27810, 27842, 27844, 27856, 27863, 27865, 27867, 27934, 27993, 28017, 28063, 28068, 28080, 28087, 28092, 28121, 28123, 28124, 28137, 28149, 28161, 28174, 28177, 28214, 28244, 28299, 28327, 28365, 28374, 28410, 28430, 28437, 28536, 28539, 28544, 28552, 28580, 28606, 28613, 28615, 28621, 28640, 28653, 28687, 28733, 28735, 28769, 28779, 28783, 28819, 28869, 28919, 28950, 28982, 28998, 29029, 29034, 29036, 29076, 29089, 29101, 29109, 29111, 29115, 29125, 29130, 29148, 29153, 29155, 29162, 29164, 29173, 29187, 29198, 29204, 29211, 29232, 29233, 29257, 29271, 29321, 29343, 29368, 29417, 29419, 29421, 29469, 29477, 29501, 29533, 29562, 29575, 29576, 29597, 29616, 29625, 29663, 29688, 29700, 29770, 29799, 29804, 29878, 29902, 29918, 29931, 29939, 29940, 29950, 30012, 30030, 30032, 30042, 30132, 30188, 30202, 30209, 30293, 30294, 30300, 30314, 30331, 30357, 30382, 30384, 30409, 30434, 30448, 30459, 30487, 30533, 30543, 30559, 30573, 30610, 30619, 30697, 30755, 30757, 30766, 30797, 30821, 30834, 30852, 30855, 30870, 30885, 30907, 30917, 30950, 30988, 31016, 31018, 31054, 31061, 31064, 31131, 31236, 31273, 31298, 31317, 31321, 31377, 31413, 31441, 31473, 31477, 31539, 31546, 31547, 31571, 31581, 31598, 31627, 31632, 31659, 31672, 31702, 31738, 31768, 31780, 31792, 31801, 31842, 31856, 31863, 31883, 31895, 31912, 31937, 32006, 32011, 32055, 32077, 32085, 32111, 32154, 32179, 32193, 32206, 32211, 32221, 32329, 32333, 32359, 32367, 32395, 32399, 32408, 32410, 32516, 32532, 32572, 32601, 32614, 32672, 32674, 32691, 32713, 32766, 32770, 32781, 32786, 32800, 32802, 32879, 32902, 32903, 32919, 32941, 32960, 32971, 33017, 33023, 33103, 33138, 33164, 33167, 33176, 33190, 33195, 33209, 33229, 33267, 33273, 33308, 33335, 33348, 33370, 33383, 33391, 33443, 33445, 33501, 33512, 33516, 33523, 33528, 33538, 33545, 33565, 33634, 33635, 33660, 33672, 33675, 33681, 33690, 33700, 33713, 33716, 33730, 33741, 33768, 33787, 33793, 33796, 33838, 33844, 33857, 33866, 33878, 33879, 33888, 33915, 33941, 33960, 33976, 34076, 34105, 34110, 34114, 34137, 34160, 34190, 34204, 34262, 34264, 34299, 34307, 34348, 34365, 34416, 34419, 34436, 34479, 34483, 34503, 34536, 34539, 34578, 34581, 34614, 34642, 34645, 34666, 34691, 34702, 34729, 34751, 34755, 34773, 34777, 34797, 34814, 34819, 34821, 34830, 34852, 34883, 34897, 34900, 34910, 34917, 34957, 35002, 35030, 35033, 35041, 35042, 35060, 35098, 35113, 35127, 35135, 35136, 35171, 35188, 35192, 35223, 35266, 35278, 35324, 35345, 35404, 35406, 35425, 35428, 35455, 35510, 35517, 35529, 35532, 35539, 35540, 35575, 35583, 35625, 35649, 35675, 35678, 35679, 35683, 35735, 35763, 35775, 35792, 35804, 35805, 35833, 35840, 35851, 35873, 35913, 35979, 35980, 35995, 36003, 36008, 36012, 36049, 36077, 36080, 36103, 36104, 36113, 36116, 36180, 36184, 36207, 36211, 36230, 36245, 36250, 36314, 36320, 36335, 36336, 36363, 36380, 36392, 36393, 36397, 36432, 36464, 36527, 36548, 36560, 36561, 36563, 36586, 36595, 36641, 36645, 36649, 36672, 36675, 36679, 36697, 36711, 36719, 36721, 36753, 36755, 36756, 36763, 36782, 36792, 36813, 36817, 36854, 36881, 36882, 36889, 36897, 36903, 36920, 36946, 36948, 36981, 37010, 37139, 37165, 37184, 37192, 37196, 37231, 37234, 37236, 37269, 37287, 37302, 37308, 37312, 37321, 37337, 37349, 37360, 37386, 37387, 37409, 37485, 37527, 37529, 37554, 37594, 37595, 37617, 37687, 37715, 37772, 37796, 37811, 37906, 37907, 37915, 37916, 37936, 37983, 38009, 38054, 38057, 38070, 38076, 38138, 38161, 38181, 38207, 38236, 38249, 38263, 38282, 38294, 38337, 38342, 38370, 38378, 38426, 38457, 38467, 38506, 38519, 38526, 38532, 38588, 38609, 38676, 38697, 38698, 38714, 38734, 38747, 38799, 38805, 38879, 38894, 38903, 38906, 38914, 38916, 38926, 38971, 39008, 39051, 39065, 39080, 39111, 39114, 39132, 39133, 39139, 39146, 39172, 39198, 39238, 39248, 39275, 39301, 39342, 39349, 39356, 39361, 39375, 39458, 39474, 39480, 39491, 39525, 39578, 39583, 39596, 39614, 39641, 39675, 39677, 39697, 39720, 39750, 39765, 39803, 39813, 39851, 39900, 39903, 39904, 39954, 39959, 39966, 40001, 40023, 40043, 40068, 40077, 40078, 40154, 40180, 40192, 40196, 40225, 40249, 40284, 40330, 40362, 40393, 40408, 40410, 40432, 40458, 40467, 40493, 40506, 40511, 40516, 40537, 40543, 40577, 40596, 40610, 40653, 40696, 40713, 40722, 40727, 40754, 40767, 40781, 40822, 40827, 40829, 40862, 40870, 40882, 40921, 40929, 40943, 40944, 40972, 41026, 41040, 41057, 41072, 41074, 41104, 41109, 41140, 41168, 41192, 41201, 41205, 41208, 41218, 41268, 41271, 41354, 41358, 41402, 41405, 41467, 41516, 41526, 41545, 41568, 41613, 41644, 41649, 41681, 41692, 41759, 41803, 41856, 41862, 41876, 41967, 42009, 42053, 42115, 42121, 42144, 42170, 42175, 42196, 42212, 42226, 42230, 42257, 42286, 42290, 42309, 42359, 42399, 42402, 42404, 42451, 42468, 42495, 42529, 42535, 42538, 42553, 42584, 42585, 42597, 42606, 42620, 42629, 42659, 42679, 42700, 42709, 42722, 42727, 42729, 42731, 42733, 42737, 42838, 42844, 42884, 42914, 42941, 42953, 42964, 43007, 43021, 43026, 43037, 43058, 43063, 43071, 43073, 43079, 43086, 43090, 43095, 43120, 43163, 43176, 43183, 43215, 43224, 43250, 43263, 43277, 43287, 43297, 43307, 43332, 43338, 43341, 43350, 43360, 43378, 43396, 43404, 43422, 43439, 43470, 43485, 43501, 43538, 43540, 43565, 43570, 43598, 43602, 43603, 43604, 43825, 43840, 43852, 43860, 43869, 43901, 43907, 43912, 43917, 43921, 43970, 43974, 43980, 44013, 44088, 44103, 44118, 44129, 44149, 44155, 44216, 44245, 44252, 44264, 44287, 44337, 44362, 44381, 44382, 44402, 44413, 44414, 44498, 44570, 44578, 44579, 44595, 44635, 44637, 44646, 44655, 44677, 44690, 44701, 44724, 44745, 44748, 44778, 44786, 44803, 44815, 44837, 44848, 44866, 44903, 44922, 44934, 44963, 44975, 44977, 45014, 45033, 45072, 45103, 45124, 45141, 45152, 45220, 45237, 45242, 45263, 45270, 45308, 45352, 45354, 45366, 45419, 45427, 45446, 45447, 45451, 45532, 45558, 45580, 45600, 45602, 45613, 45640, 45702, 45711, 45786, 45820, 45843, 45865, 45874, 45942, 45948, 46016, 46045, 46049, 46072, 46097, 46108, 46125, 46149, 46161, 46178, 46200, 46221, 46231, 46253, 46294, 46300, 46303, 46391, 46395, 46457, 46465, 46478, 46496, 46548, 46569, 46576, 46592, 46615, 46656, 46667, 46683, 46707, 46728, 46748, 46790, 46793, 46797, 46833, 46848, 46865, 46869, 46898, 46911, 46925, 46955, 46982, 47005, 47022, 47053, 47054, 47065, 47071, 47077, 47154, 47176, 47216, 47222, 47237, 47263, 47266, 47310, 47327, 47411, 47451, 47476, 47500, 47508, 47512, 47529, 47537, 47589, 47615, 47649, 47650, 47699, 47707, 47718, 47764, 47773, 47792, 47797, 47832, 47857, 47859, 47866, 47914, 47946, 47954, 47957, 47966, 47989, 47994, 48009, 48036, 48042, 48111, 48146, 48160, 48168, 48187, 48192, 48218, 48245, 48261, 48300, 48301, 48311, 48320, 48327, 48333, 48334, 48414, 48506, 48517, 48534, 48601, 48694, 48741, 48742, 48751, 48759, 48776, 48802, 48810, 48831, 48850, 48854, 48862, 48867, 48872, 48877, 48887, 48890, 48949, 48957, 48959, 48966, 48984, 49016, 49028, 49047, 49094, 49113, 49121, 49131, 49141, 49152, 49174, 49189, 49212, 49240, 49249, 49287, 49305, 49306, 49326, 49351, 49368, 49397, 49426, 49451, 49463, 49471, 49494, 49496, 49505, 49567, 49575, 49578, 49630, 49678, 49705, 49759, 49774, 49777, 49823, 49834, 49838, 49876, 49880, 49894, 49920, 49958, 50026, 50034, 50059, 50067, 50071, 50091, 50109, 50219, 50228, 50232, 50239, 50292, 50293, 50307, 50326, 50333, 50335, 50371, 50410, 50428, 50436, 50471, 50486, 50497, 50514, 50549, 50562, 50570, 50600, 50603, 50615, 50646, 50657, 50707, 50714, 50751, 50807, 50827, 50830, 50847, 50853, 50874, 50898, 50936, 51007, 51108, 51129, 51147, 51157, 51191, 51230, 51283, 51303, 51338, 51350, 51389, 51441, 51443, 51464, 51475, 51513, 51535, 51541, 51543, 51562, 51604, 51635, 51687, 51704, 51711, 51719, 51742, 51765, 51783, 51786, 51794, 51800, 51821, 51857, 51911, 51937, 51940, 51942, 51946, 51983, 51992, 52005, 52011, 52031, 52090, 52107, 52109, 52144, 52153, 52193, 52205, 52215, 52254, 52267, 52268, 52292, 52304, 52384, 52387, 52426, 52429, 52460, 52463, 52473, 52508, 52543, 52552, 52566, 52585, 52588, 52596, 52599, 52632, 52633, 52635, 52639, 52645, 52682, 52687, 52700, 52711, 52730, 52745, 52754, 52756, 52761, 52766, 52777, 52780, 52819, 52856, 52859, 52862, 52883, 52896, 52904, 52908, 52918, 52929, 52938, 52946, 52951, 52957, 52964, 53001, 53016, 53073, 53129, 53150, 53161, 53164, 53166, 53173, 53208, 53213, 53225, 53262, 53268, 53290, 53312, 53332, 53345, 53387, 53414, 53416, 53458, 53460, 53476, 53549, 53589, 53590, 53617, 53642, 53650, 53661, 53701, 53785, 53815, 53844, 53911, 53939, 53974, 53985, 54028, 54032, 54034, 54073, 54078, 54095, 54107, 54155, 54157, 54170, 54189, 54204, 54206, 54230, 54232, 54243, 54270, 54286, 54290, 54297, 54320, 54333, 54339, 54402, 54406, 54408, 54410, 54411, 54418, 54436, 54480, 54516, 54577, 54617, 54638, 54719, 54734, 54745, 54763, 54779, 54780, 54811, 54826, 54842, 54843, 54854, 54870, 54909, 54998, 55004, 55026, 55048, 55049, 55057, 55097, 55125, 55149, 55151, 55170, 55181, 55189, 55211, 55253, 55264, 55266, 55311, 55342]
    test_lines = [20, 35, 130, 170, 176, 181, 212, 251, 265, 274, 301, 309, 344, 346, 379, 396, 414, 433, 439, 443, 445, 480, 494, 533, 539, 544, 546, 607, 618, 640, 679, 692, 695, 723, 733, 735, 755, 757, 780, 798, 864, 896, 920, 922, 941, 954, 994, 1021, 1053, 1062, 1087, 1136, 1169, 1176, 1181, 1220, 1298, 1318, 1364, 1387, 1413, 1415, 1437, 1450, 1472, 1530, 1533, 1604, 1612, 1614, 1625, 1693, 1720, 1746, 1748, 1760, 1779, 1790, 1810, 1907, 1945, 1973, 1974, 1980, 2033, 2043, 2060, 2067, 2075, 2101, 2109, 2118, 2128, 2147, 2173, 2194, 2223, 2231, 2232, 2262, 2289, 2308, 2319, 2416, 2448, 2450, 2456, 2465, 2477, 2536, 2581, 2584, 2619, 2626, 2631, 2641, 2642, 2643, 2674, 2682, 2707, 2737, 2756, 2800, 2816, 2824, 2888, 2890, 2915, 2921, 2953, 2964, 2970, 2995, 3007, 3031, 3065, 3067, 3100, 3101, 3108, 3122, 3171, 3227, 3236, 3254, 3308, 3312, 3390, 3395, 3424, 3475, 3503, 3508, 3527, 3554, 3559, 3605, 3638, 3644, 3711, 3714, 3718, 3756, 3810, 3867, 3868, 3921, 3923, 3924, 3962, 3964, 3967, 3968, 3980, 3986, 4005, 4006, 4007, 4050, 4053, 4070, 4080, 4119, 4154, 4224, 4244, 4273, 4276, 4289, 4297, 4356, 4373, 4386, 4463, 4465, 4472, 4477, 4489, 4507, 4535, 4549, 4561, 4596, 4619, 4635, 4704, 4712, 4721, 4773, 4776, 4783, 4796, 4805, 4834, 4835, 4852, 4892, 4910, 4930, 4935, 4968, 4974, 4999, 5008, 5012, 5013, 5030, 5092, 5133, 5144, 5145, 5163, 5210, 5220, 5230, 5233, 5290, 5321, 5333, 5339, 5358, 5366, 5381, 5394, 5399, 5417, 5449, 5450, 5481, 5489, 5493, 5498, 5500, 5568, 5602, 5621, 5671, 5684, 5721, 5749, 5751, 5769, 5787, 5801, 5817, 5833, 5844, 5848, 5892, 5893, 5903, 5927, 5966, 5978, 5986, 6003, 6004, 6014, 6017, 6025, 6048, 6062, 6101, 6103, 6107, 6167, 6206, 6210, 6229, 6234, 6255, 6336, 6341, 6356, 6400, 6408, 6411, 6428, 6439, 6447, 6459, 6473, 6491, 6501, 6503, 6505, 6517, 6522, 6532, 6542, 6575, 6583, 6584, 6617, 6627, 6633, 6644, 6645, 6663, 6673, 6685, 6695, 6701, 6707, 6709, 6731, 6758, 6792, 6848, 6934, 6937, 6945, 6954, 6963, 6966, 7002, 7015, 7027, 7044, 7079, 7090, 7094, 7111, 7131, 7139, 7142, 7146, 7155, 7167, 7178, 7191, 7246, 7251, 7253, 7260, 7283, 7288, 7333, 7362, 7369, 7390, 7399, 7403, 7414, 7418, 7437, 7439, 7475, 7478, 7480, 7492, 7499, 7532, 7533, 7561, 7575, 7576, 7581, 7646, 7665, 7678, 7746, 7771, 7825, 7839, 7881, 7918, 7952, 8009, 8020, 8032, 8065, 8070, 8073, 8125, 8126, 8129, 8131, 8176, 8177, 8227, 8232, 8255, 8264, 8313, 8314, 8323, 8327, 8400, 8511, 8519, 8537, 8538, 8546, 8548, 8554, 8564, 8582, 8584, 8606, 8613, 8637, 8650, 8684, 8718, 8773, 8821, 8891, 8898, 8901, 8903, 8911, 8912, 8972, 8985, 9007, 9018, 9035, 9060, 9107, 9109, 9118, 9119, 9167, 9201, 9213, 9252, 9343, 9378, 9387, 9388, 9391, 9421, 9432, 9441, 9448, 9467, 9503, 9511, 9519, 9524, 9530, 9537, 9545, 9565, 9566, 9575, 9603, 9612, 9636, 9655, 9661, 9724, 9730, 9776, 9795, 9819, 9849, 9891, 9910, 9911, 9947, 9958, 9976, 9992, 9995, 10008, 10056, 10072, 10075, 10119, 10121, 10137, 10172, 10283, 10289, 10331, 10333, 10354, 10366, 10398, 10423, 10434, 10457, 10480, 10488, 10499, 10514, 10528, 10569, 10575, 10580, 10590, 10613, 10623, 10628, 10645, 10648, 10649, 10666, 10668, 10670, 10688, 10707, 10708, 10716, 10721, 10759, 10769, 10785, 10801, 10803, 10846, 10847, 10915, 10929, 10951, 10974, 10979, 10986, 10987, 10988, 11011, 11019, 11044, 11054, 11061, 11079, 11138, 11150, 11168, 11182, 11195, 11203, 11208, 11212, 11215, 11221, 11325, 11353, 11369, 11377, 11407, 11466, 11524, 11529, 11597, 11607, 11628, 11639, 11657, 11672, 11688, 11690, 11708, 11711, 11718, 11746, 11751, 11788, 11828, 11833, 11836, 11857, 11873, 11879, 11915, 11917, 11967, 11996, 12003, 12011, 12024, 12049, 12083, 12125, 12151, 12168, 12179, 12192, 12193, 12194, 12212, 12222, 12224, 12238, 12258, 12284, 12285, 12331, 12339, 12354, 12422, 12433, 12444, 12464, 12519, 12531, 12537, 12544, 12615, 12634, 12637, 12659, 12662, 12728, 12759, 12784, 12844, 12875, 12883, 12887, 12892, 12913, 12935, 12939, 13039, 13043, 13059, 13112, 13119, 13122, 13158, 13159, 13165, 13169, 13175, 13178, 13193, 13203, 13226, 13244, 13257, 13284, 13291, 13327, 13328, 13338, 13376, 13387, 13392, 13425, 13432, 13444, 13451, 13496, 13516, 13540, 13546, 13552, 13554, 13556, 13563, 13600, 13731, 13768, 13784, 13830, 13831, 13852, 13884, 13933, 13956, 13962, 13974, 13976, 14029, 14030, 14038, 14056, 14066, 14107, 14242, 14284, 14337, 14344, 14477, 14504, 14505, 14514, 14541, 14550, 14555, 14559, 14566, 14585, 14588, 14626, 14640, 14655, 14705, 14742, 14765, 14776, 14783, 14813, 14835, 14875, 14896, 14905, 14910, 14926, 14937, 14945, 14962, 14991, 14998, 15001, 15006, 15009, 15014, 15039, 15041, 15042, 15043, 15058, 15067, 15070, 15078, 15107, 15122, 15144, 15176, 15179, 15193, 15206, 15239, 15255, 15259, 15262, 15264, 15275, 15296, 15313, 15336, 15375, 15421, 15424, 15436, 15515, 15517, 15558, 15621, 15636, 15639, 15648, 15655, 15656, 15675, 15721, 15735, 15742, 15744, 15773, 15784, 15803, 15822, 15848, 15866, 15876, 15884, 15887, 15903, 15914, 15915, 15940, 15954, 15968, 15984, 16000, 16003, 16014, 16044, 16055, 16080, 16093, 16095, 16140, 16170, 16173, 16206, 16233, 16242, 16247, 16259, 16275, 16290, 16301, 16325, 16334, 16342, 16344, 16350, 16385, 16392, 16456, 16492, 16516, 16574, 16629, 16638, 16643, 16657, 16701, 16710, 16751, 16789, 16807, 16856, 16868, 16905, 16917, 16936, 16956, 17015, 17024, 17044, 17076, 17089, 17096, 17099, 17109, 17121, 17208, 17214, 17218, 17254, 17262, 17279, 17321, 17350, 17358, 17360, 17430, 17436, 17441, 17451, 17488, 17492, 17505, 17518, 17528, 17557, 17585, 17602, 17626, 17665, 17713, 17785, 17809, 17830, 17878, 17881, 17950, 17998, 18004, 18011, 18024, 18070, 18086, 18108, 18129, 18154, 18155, 18211, 18218, 18221, 18284, 18290, 18319, 18324, 18333, 18343, 18375, 18454, 18468, 18478, 18515, 18581, 18596, 18597, 18624, 18643, 18715, 18738, 18760, 18764, 18798, 18827, 18832, 18862, 18876, 18887, 18904, 18932, 18937, 18988, 19026, 19067, 19088, 19091, 19131, 19132, 19153, 19175, 19201, 19205, 19249, 19262, 19285, 19286, 19312, 19324, 19335, 19360, 19369, 19411, 19416, 19434, 19475, 19489, 19512, 19533, 19587, 19601, 19609, 19628, 19640, 19659, 19665, 19717, 19806, 19809, 19822, 19827, 19878, 19882, 19902, 19935, 19972, 20041, 20117, 20184, 20273, 20297, 20300, 20380, 20392, 20442, 20472, 20497, 20509, 20511, 20519, 20548, 20554, 20577, 20589, 20610, 20617, 20628, 20673, 20681, 20682, 20720, 20745, 20779, 20787, 20806, 20808, 20810, 20829, 20850, 20890, 20893, 20904, 20942, 20952, 20955, 20959, 20964, 21001, 21004, 21006, 21040, 21046, 21055, 21058, 21090, 21103, 21114, 21137, 21157, 21215, 21278, 21284, 21342, 21371, 21388, 21421, 21424, 21426, 21446, 21453, 21479, 21484, 21506, 21519, 21521, 21554, 21565, 21581, 21587, 21592, 21630, 21639, 21725, 21744, 21778, 21802, 21817, 21891, 21892, 21905, 21928, 21933, 21934, 22058, 22070, 22106, 22116, 22142, 22180, 22189, 22213, 22236, 22239, 22251, 22259, 22288, 22304, 22322, 22330, 22401, 22425, 22500, 22529, 22535, 22546, 22589, 22640, 22644, 22680, 22689, 22727, 22756, 22806, 22811, 22815, 22819, 22824, 22880, 22887, 22898, 22912, 22920, 23002, 23042, 23044, 23063, 23101, 23109, 23110, 23126, 23127, 23145, 23170, 23174, 23179, 23206, 23212, 23270, 23275, 23288, 23357, 23360, 23373, 23381, 23403, 23409, 23411, 23455, 23492, 23536, 23549, 23570, 23654, 23669, 23698, 23734, 23742, 23749, 23765, 23788, 23789, 23797, 23819, 23825, 23832, 23848, 23882, 23896, 23916, 23930, 23998, 24011, 24024, 24029, 24044, 24099, 24108, 24114, 24139, 24140, 24167, 24170, 24179, 24195, 24230, 24237, 24251, 24277, 24313, 24315, 24337, 24387, 24411, 24413, 24447, 24455, 24482, 24487, 24571, 24575, 24582, 24586, 24604, 24661, 24681, 24693, 24700, 24705, 24706, 24745, 24752, 24783, 24791, 24799, 24840, 24867, 24887, 25018, 25052, 25078, 25091, 25105, 25122, 25134, 25162, 25264, 25269, 25276, 25302, 25318, 25337, 25340, 25389, 25412, 25413, 25428, 25434, 25500, 25502, 25504, 25524, 25582, 25594, 25623, 25636, 25637, 25654, 25664, 25666, 25694, 25697, 25701, 25744, 25829, 25836, 25840, 25846, 25906, 25924, 25941, 25944, 25969, 26007, 26012, 26036, 26048, 26056, 26062, 26070, 26084, 26088, 26104, 26132, 26136, 26180, 26222, 26266, 26273, 26302, 26305, 26307, 26413, 26429, 26430, 26434, 26478, 26482, 26506, 26514, 26525, 26543, 26568, 26574, 26575, 26624, 26626, 26627, 26665, 26688, 26709, 26721, 26728, 26747, 26819, 26826, 26850, 26866, 26872, 26875, 26904, 26921, 26967, 26975, 26977, 26983, 27028, 27106, 27151, 27157, 27158, 27210, 27279, 27283, 27285, 27286, 27324, 27345, 27346, 27365, 27369, 27386, 27392, 27399, 27429, 27433, 27452, 27461, 27472, 27502, 27513, 27519, 27537, 27538, 27549, 27573, 27653, 27668, 27685, 27699, 27710, 27735, 27775, 27816, 27850, 27858, 27897, 27915, 27947, 28018, 28031, 28055, 28091, 28134, 28183, 28188, 28199, 28218, 28251, 28260, 28303, 28317, 28326, 28356, 28359, 28386, 28399, 28408, 28435, 28454, 28480, 28490, 28505, 28560, 28605, 28610, 28617, 28671, 28674, 28676, 28696, 28700, 28701, 28736, 28739, 28755, 28781, 28792, 28814, 28855, 28896, 28923, 28930, 28936, 29009, 29015, 29038, 29047, 29063, 29078, 29095, 29104, 29112, 29114, 29123, 29138, 29156, 29206, 29219, 29222, 29279, 29289, 29298, 29316, 29319, 29389, 29398, 29430, 29485, 29509, 29527, 29540, 29549, 29561, 29570, 29614, 29617, 29637, 29644, 29649, 29650, 29678, 29689, 29706, 29774, 29787, 29796, 29809, 29835, 29838, 29845, 29882, 29883, 29910, 29916, 29927, 29936, 29953, 29975, 30008, 30026, 30057, 30063, 30102, 30120, 30130, 30139, 30154, 30160, 30196, 30199, 30205, 30241, 30248, 30256, 30373, 30377, 30407, 30447, 30472, 30494, 30498, 30499, 30524, 30545, 30554, 30584, 30603, 30606, 30612, 30614, 30620, 30641, 30660, 30681, 30694, 30731, 30758, 30830, 30849, 30850, 30857, 30902, 30910, 30977, 30986, 30994, 31010, 31056, 31083, 31084, 31102, 31145, 31163, 31248, 31254, 31274, 31294, 31345, 31363, 31406, 31411, 31451, 31460, 31463, 31503, 31519, 31521, 31551, 31570, 31574, 31587, 31588, 31617, 31629, 31667, 31676, 31685, 31757, 31786, 31787, 31795, 31833, 31862, 31909, 31917, 31925, 31941, 31955, 31961, 31969, 31973, 31975, 32023, 32051, 32057, 32086, 32107, 32108, 32135, 32192, 32203, 32226, 32241, 32308, 32326, 32340, 32385, 32419, 32424, 32442, 32451, 32457, 32479, 32501, 32524, 32547, 32548, 32558, 32582, 32590, 32716, 32763, 32833, 32864, 32897, 32898, 32922, 32925, 32927, 32944, 32955, 32983, 32996, 33000, 33011, 33051, 33072, 33073, 33150, 33165, 33191, 33201, 33233, 33237, 33250, 33253, 33279, 33280, 33344, 33373, 33384, 33413, 33449, 33453, 33456, 33474, 33535, 33568, 33572, 33576, 33579, 33601, 33602, 33611, 33615, 33616, 33621, 33676, 33683, 33717, 33718, 33724, 33729, 33752, 33755, 33756, 33758, 33765, 33767, 33795, 33825, 33828, 33833, 33886, 33899, 33918, 33935, 33951, 33962, 33979, 34015, 34031, 34033, 34035, 34049, 34082, 34083, 34090, 34143, 34240, 34318, 34325, 34329, 34334, 34343, 34347, 34351, 34398, 34423, 34438, 34460, 34504, 34527, 34551, 34565, 34574, 34576, 34592, 34600, 34601, 34628, 34636, 34653, 34721, 34734, 34754, 34758, 34760, 34766, 34876, 34898, 34920, 34921, 34938, 34941, 34964, 34969, 34982, 34995, 34998, 35001, 35083, 35088, 35099, 35121, 35125, 35133, 35149, 35194, 35220, 35297, 35315, 35332, 35365, 35375, 35403, 35450, 35496, 35516, 35567, 35568, 35581, 35591, 35603, 35604, 35606, 35628, 35640, 35648, 35659, 35660, 35710, 35711, 35712, 35740, 35823, 35849, 35867, 35875, 35877, 35893, 35920, 35935, 35946, 35951, 35987, 36019, 36023, 36036, 36060, 36069, 36084, 36124, 36129, 36136, 36138, 36147, 36199, 36212, 36213, 36225, 36228, 36234, 36269, 36279, 36281, 36296, 36319, 36333, 36434, 36502, 36590, 36603, 36610, 36642, 36671, 36678, 36738, 36741, 36747, 36775, 36780, 36797, 36861, 36880, 36931, 36944, 36973, 36977, 36998, 37013, 37032, 37055, 37060, 37075, 37084, 37095, 37101, 37126, 37128, 37129, 37130, 37132, 37151, 37170, 37182, 37202, 37207, 37208, 37210, 37237, 37283, 37290, 37331, 37335, 37358, 37385, 37405, 37430, 37434, 37435, 37495, 37507, 37550, 37551, 37559, 37598, 37602, 37609, 37621, 37624, 37633, 37640, 37646, 37652, 37657, 37686, 37708, 37719, 37757, 37841, 37862, 37870, 37912, 37954, 37960, 38012, 38077, 38088, 38105, 38108, 38131, 38140, 38172, 38209, 38211, 38214, 38245, 38248, 38257, 38301, 38303, 38307, 38373, 38382, 38391, 38400, 38451, 38474, 38503, 38564, 38575, 38584, 38587, 38596, 38660, 38666, 38708, 38739, 38744, 38752, 38769, 38781, 38811, 38814, 38881, 38882, 38885, 38937, 38944, 38947, 38979, 38989, 39026, 39030, 39047, 39057, 39070, 39104, 39156, 39157, 39162, 39174, 39187, 39214, 39221, 39262, 39276, 39293, 39328, 39332, 39345, 39357, 39371, 39378, 39382, 39398, 39413, 39420, 39423, 39425, 39435, 39486, 39564, 39566, 39568, 39581, 39587, 39617, 39620, 39623, 39640, 39647, 39666, 39701, 39714, 39726, 39740, 39764, 39769, 39778, 39793, 39856, 39870, 39897, 39906, 39923, 39938, 39960, 39964, 39985, 40003, 40009, 40035, 40037, 40038, 40088, 40107, 40117, 40129, 40131, 40136, 40161, 40165, 40181, 40184, 40187, 40209, 40220, 40227, 40258, 40261, 40281, 40299, 40303, 40304, 40326, 40339, 40354, 40356, 40357, 40377, 40386, 40387, 40391, 40402, 40403, 40417, 40425, 40451, 40453, 40487, 40551, 40564, 40598, 40622, 40629, 40662, 40667, 40691, 40708, 40751, 40756, 40765, 40769, 40777, 40799, 40898, 40965, 40973, 40984, 41011, 41054, 41117, 41119, 41136, 41167, 41182, 41217, 41225, 41227, 41278, 41286, 41290, 41313, 41318, 41323, 41340, 41361, 41378, 41379, 41400, 41447, 41482, 41499, 41531, 41532, 41534, 41564, 41627, 41669, 41710, 41723, 41727, 41748, 41809, 41815, 41820, 41849, 41852, 41910, 41924, 41964, 41991, 41995, 41998, 41999, 42023, 42027, 42041, 42067, 42071, 42079, 42083, 42089, 42101, 42114, 42135, 42186, 42192, 42218, 42221, 42239, 42281, 42289, 42296, 42302, 42316, 42318, 42346, 42356, 42396, 42453, 42474, 42498, 42534, 42568, 42590, 42602, 42618, 42619, 42625, 42640, 42654, 42660, 42663, 42694, 42741, 42768, 42802, 42828, 42882, 42885, 42888, 42894, 42906, 42956, 42966, 42982, 42987, 42995, 43023, 43057, 43072, 43093, 43134, 43137, 43143, 43156, 43187, 43209, 43234, 43280, 43288, 43290, 43313, 43315, 43354, 43370, 43397, 43438, 43454, 43478, 43514, 43529, 43536, 43537, 43566, 43643, 43666, 43667, 43668, 43685, 43706, 43711, 43713, 43726, 43735, 43757, 43766, 43769, 43770, 43797, 43835, 43838, 43918, 43938, 43941, 43991, 44000, 44006, 44020, 44022, 44025, 44035, 44054, 44055, 44067, 44070, 44081, 44112, 44114, 44132, 44135, 44178, 44202, 44204, 44248, 44285, 44330, 44348, 44363, 44384, 44406, 44458, 44469, 44481, 44499, 44500, 44510, 44569, 44584, 44587, 44588, 44601, 44612, 44643, 44644, 44660, 44702, 44727, 44728, 44734, 44739, 44740, 44749, 44757, 44760, 44772, 44782, 44796, 44797, 44805, 44808, 44824, 44829, 44835, 44863, 44869, 44882, 45000, 45008, 45032, 45039, 45050, 45093, 45100, 45111, 45129, 45137, 45138, 45153, 45160, 45239, 45258, 45259, 45282, 45283, 45323, 45328, 45343, 45349, 45373, 45382, 45390, 45407, 45439, 45452, 45467, 45471, 45482, 45503, 45509, 45633, 45647, 45656, 45668, 45705, 45729, 45748, 45751, 45773, 45787, 45798, 45800, 45803, 45858, 45864, 45891, 45937, 45944, 45964, 45975, 46038, 46067, 46082, 46106, 46129, 46138, 46144, 46159, 46185, 46213, 46262, 46263, 46307, 46313, 46314, 46332, 46345, 46377, 46398, 46419, 46452, 46476, 46477, 46481, 46486, 46494, 46512, 46517, 46520, 46521, 46531, 46550, 46559, 46574, 46599, 46600, 46680, 46704, 46705, 46743, 46785, 46794, 46816, 46837, 46838, 46860, 46893, 46979, 46989, 46994, 47008, 47155, 47164, 47172, 47219, 47231, 47278, 47299, 47307, 47319, 47325, 47326, 47332, 47359, 47367, 47390, 47398, 47409, 47422, 47437, 47455, 47460, 47467, 47506, 47509, 47546, 47550, 47590, 47597, 47604, 47666, 47673, 47677, 47698, 47710, 47727, 47775, 47785, 47798, 47830, 47835, 47840, 47841, 47901, 47904, 47944, 47969, 47974, 47984, 48029, 48035, 48085, 48096, 48098, 48100, 48135, 48170, 48175, 48215, 48216, 48241, 48243, 48291, 48302, 48330, 48331, 48341, 48342, 48358, 48365, 48400, 48436, 48453, 48476, 48491, 48492, 48500, 48529, 48545, 48569, 48587, 48588, 48604, 48606, 48653, 48657, 48672, 48674, 48677, 48685, 48690, 48744, 48761, 48801, 48811, 48827, 48858, 48865, 48866, 48884, 48910, 48912, 48921, 48922, 48927, 48996, 49008, 49048, 49052, 49068, 49082, 49098, 49104, 49163, 49182, 49197, 49256, 49271, 49275, 49281, 49292, 49297, 49347, 49366, 49402, 49416, 49428, 49433, 49447, 49458, 49479, 49495, 49501, 49517, 49558, 49560, 49620, 49638, 49649, 49651, 49677, 49679, 49681, 49722, 49730, 49771, 49773, 49781, 49786, 49797, 49847, 49857, 49884, 49885, 49887, 49899, 49906, 49910, 49913, 49935, 49945, 49959, 50005, 50169, 50205, 50225, 50236, 50266, 50271, 50278, 50279, 50295, 50329, 50349, 50353, 50373, 50393, 50403, 50422, 50441, 50453, 50488, 50510, 50518, 50521, 50614, 50624, 50654, 50656, 50672, 50698, 50742, 50756, 50777, 50803, 50844, 50858, 50867, 50868, 50872, 50877, 50890, 50916, 50960, 50980, 50986, 50993, 51017, 51044, 51054, 51055, 51101, 51142, 51156, 51159, 51163, 51190, 51197, 51203, 51207, 51219, 51224, 51231, 51277, 51290, 51334, 51343, 51408, 51415, 51417, 51435, 51439, 51452, 51490, 51492, 51530, 51565, 51571, 51573, 51586, 51594, 51624, 51640, 51688, 51691, 51708, 51717, 51725, 51741, 51778, 51781, 51810, 51876, 51880, 51913, 51915, 51925, 51951, 51989, 52020, 52022, 52043, 52047, 52068, 52069, 52080, 52087, 52099, 52118, 52128, 52131, 52134, 52160, 52211, 52273, 52301, 52302, 52309, 52322, 52337, 52344, 52372, 52376, 52406, 52420, 52430, 52442, 52461, 52469, 52517, 52542, 52610, 52615, 52626, 52685, 52686, 52694, 52721, 52742, 52743, 52744, 52760, 52772, 52801, 52823, 52825, 52840, 52849, 52851, 52913, 52931, 52936, 52937, 52942, 52949, 52979, 53048, 53105, 53169, 53187, 53224, 53229, 53256, 53286, 53298, 53343, 53357, 53378, 53389, 53466, 53467, 53498, 53518, 53543, 53560, 53571, 53620, 53633, 53663, 53681, 53702, 53706, 53707, 53710, 53714, 53720, 53774, 53801, 53812, 53813, 53906, 53908, 53918, 53931, 53936, 53940, 53948, 53960, 53972, 54015, 54057, 54074, 54084, 54102, 54108, 54151, 54164, 54166, 54235, 54282, 54310, 54312, 54324, 54378, 54385, 54413, 54466, 54487, 54503, 54505, 54551, 54602, 54620, 54632, 54665, 54685, 54716, 54722, 54723, 54792, 54793, 54797, 54810, 54836, 54908, 54924, 54932, 54933, 54958, 55012, 55024, 55045, 55052, 55083, 55088, 55090, 55138, 55160, 55162, 55178, 55182, 55201, 55220, 55240, 55260, 55265, 55268, 55283, 55284, 55307, 55318]

    write_train_valid_test_files("signs", "ak", valid_lines, test_lines)
    write_train_valid_test_files("transcriptions", "tr", valid_lines, test_lines)
    write_train_valid_test_files("translation", "en", valid_lines, test_lines)


def main():
    """
    Builds data for translation algorithms
    :return: nothing
    """
    corpora = ["rinap", "riao", "ribo", "saao", "suhu"]
    preprocess(corpora)
    build_train_valid_test()


if __name__ == '__main__':
    main()
