import subprocess
from pathlib import Path
from translation_tokenize import tokenize
from translate_common import source, translation, detokenize_transliteration, detokenize_translation

def subscript(l):
    if l == "0":
        return "₀"
    elif l == "1":
        return "₁"
    elif l == "2":
        return "₂"
    elif l == "3":
        return "₃"
    elif l == "4":
        return "₄"
    elif l == "5":
        return "₅"
    elif l == "6":
        return "₆"
    elif l == "7":
        return "₇"
    elif l == "8":
        return "₈"
    elif l == "9":
        return "₉"
    else:
        return l


def find_all_occurences(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def fix_logogram(line):
    left_brackets = find_all_occurences(line, "{")
    right_brackets = find_all_occurences(line, "}")

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
    new_line += line[right_brackets[len(left_brackets)-1]:]

    new_line.replace("{KI}-", "{KI} ").replace("{M}", "{m}").replace("{D}", "{d}")

    return new_line


def organize_transliteration_line(line):
    new_line = ""

    for l in line:
        if l == "ḫ":
            new_line += "h"
        elif l == "◌́ ":
            new_line += "₂"
        elif l == "◌`":
            new_line += "₃"
        elif l.isdigit():
            new_line += subscript(l)
        else:
            new_line += l

    new_line.replace("aš₂", "aš2").replace("kas₂", "kas2").replace("kul₂", "kul2").replace("dab₂", "dab2")
    new_line.replace("šul₃", "šul3").replace("ti₃", "ti3").replace("kat₃", "kat3").replace("lib₃", "lib3")
    new_line.replace("tu₄", "tu4").replace("u₄", "u4")
    new_line.replace("₂}", "2}").replace("₃}", "3}")

    new_line = fix_logogram(new_line)

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
            print(detokenize_transliteration(line))
        if translation(line):
            print(detokenize_translation(line, True) + "\n")


if __name__ == '__main__':
    transliteration_file = input("Please enter the name of the transliteration file for translation\n")
    translate_transliteration_file(transliteration_file)
