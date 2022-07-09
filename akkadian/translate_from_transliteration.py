import subprocess
from pathlib import Path
import string
from translation_tokenize import tokenize
from translate_common import source, translation, detokenize_transliteration, detokenize_translation

letter_substitutions = {
    "ḫ": "h"
}

number_substitutions = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉"
}

acute_grave_substitutions = {
    "á": "a₂", "é": "e₂", "í": "i₂", "ó": "o₂", "ú":"u₂",
    "à": "a₃", "è": "e₃", "ì": "i₃", "ò": "o₃", "ù":"u₃"
}

logogram_substitutions = {
    "--": "-", "- ": "-",
    "{KI}-": "{KI} ", "{M}": "{m}", "{D}": "{d}",
    "₂}": "2}", "₃}": "3}"
}

exception_substitutions = {
    "aš₂": "aš2", "kas₂": "kas2", "kul₂": "kul2", "dab₂": "dab2",
    "šul₃": "šul3", "ti₃": "ti3", "kat₃": "kat3", "lib₃": "lib3",
    "tu₄": "tu4", "u₄": "u4",
    "LIL2": "LIL₂", "DU8": "DU₈", "LU2": "LU₂", "ŠA3": "ŠA₃",
    "ŠAR2": "ŠAR₂", "SIG5": "SIG₅", "Ku3": "KU₃", "Du3": "DU₃"
}

def find_all_occurences(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def fix_logogram(line):
    left_brackets = find_all_occurences(line, "{")
    right_brackets = find_all_occurences(line, "}")

    if len(left_brackets) == 0:
        return line

    if len(left_brackets) != len(right_brackets):
        return line

    for i in range(len(left_brackets)):
        if left_brackets[i] >= right_brackets[i]:
            return line

    new_line = ""
    for i in range(len(left_brackets)):
        if i == 0:
            new_line += line[0:left_brackets[0]]
        else:
            new_line += line[right_brackets[i-1]+1:left_brackets[i]]
        new_line += line[left_brackets[i]:right_brackets[i]+1].upper() + "-"
    new_line += line[right_brackets[len(left_brackets)-1]+1:]

    new_line = substitute_phrase(new_line, logogram_substitutions)

    return new_line


def substitute_phrase(line, substitution_dict):
    for key, value in substitution_dict.items():
        if key in line:
            line = line.replace(key, value)

    return line


def is_constant(l):
    if l in string.ascii_letters and not l in "aeiouAEIOU":
        return True
    return False


def fix_acute_grave(line):
    new_line = ""

    i = 0
    while i < len(line):
        if line[i] in acute_grave_substitutions:
            if i + 1 < len(line) and is_constant(line[i+1]):
                new_line += acute_grave_substitutions[line[i]][0] + line[i+1] + acute_grave_substitutions[line[i]][1]
                i += 1
            else:
                new_line += acute_grave_substitutions[line[i]]
        else:
            new_line += line[i]
        i += 1

    return new_line


def fix_numbers(line):
    new_line = ""

    prev_l = ""
    for l in line:
        if prev_l and prev_l in string.ascii_letters and l in number_substitutions:
            new_line += number_substitutions[l]
        else:
            new_line += l
        prev_l = l

    return new_line


def organize_transliteration_line(line):
    new_line = ""

    for l in line:
        if l in letter_substitutions:
            new_line += letter_substitutions[l]
        else:
            new_line += l

    new_line = fix_numbers(new_line)
    new_line = fix_acute_grave(new_line)
    new_line = fix_logogram(new_line)
    new_line = substitute_phrase(new_line, exception_substitutions)

    return new_line


def organize_transliteration_input(file):
    tmp_file_name = "tmp_" + file
    tmp_file = open(tmp_file_name, "w", encoding="utf8")

    with open(file, "r", encoding="utf8") as f:
        for line in f:
            tmp_file.write(organize_transliteration_line(line))

    tmp_file.close()
    return tmp_file_name


def translate_transliteration_base(file, capture_output=False):
    tmp_file = organize_transliteration_input(file)
    tokenize("transliteration_bpe", tmp_file, False, Path("NMT_input/tokenization"), Path(""), Path("/tmp"))
    cmd = "../fairseq/fairseq_cli/interactive.py " \
          "data-bin-transliteration/ " \
          "--path trans_result.LR_0.1.MAX_TOKENS_4000/checkpoint_best.pt " \
          "--beam 5 " \
          "--input /tmp/" + tmp_file

    if capture_output:
        return subprocess.run(cmd.split(), capture_output=True)
    return subprocess.run(cmd.split())


def translate_transliteration_raw(file):
    translate_transliteration_base(file)


def translate_transliteration_file(file):
    raw_result = translate_transliteration_base(file, True).stdout
    for line in raw_result.decode().split('\n'):
        if source(line):
            print(detokenize_transliteration(line, "NMT_input/tokenization/transliteration_bpe.model"))
        if translation(line):
            print(detokenize_translation(line, "NMT_input/tokenization/translation_bpe.model", True) + "\n")


if __name__ == '__main__':
    transliteration_file = input("Please enter the name of the transliteration file for translation\n")
    translate_transliteration_file(transliteration_file)
